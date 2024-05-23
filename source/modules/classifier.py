import os
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# from modules import embeddings
from modules import iepirkumi

WINNER_EMBEDDINGS_FNAME = "./tmp/winner_embeds.npy"
ACTIVE_EMBEDDINGS_FNAME = "./tmp/active_embeds.npy"

SKIP_EMBEDS = os.environ.get("SKIP_CREATING_EMBEDDINGS") == "1"

OPTIMAL_K = 300
RANDOM_SATE = 36
PCA_COMPONENTS = 50


class Classifier:

    def __init__(self):
        self.lg = logging.getLogger("classifier")
        self.lg.info("Loading data...")
        self.winners_df = pd.read_pickle(iepirkumi.prepare_winners())
        self.active_df = pd.read_pickle(iepirkumi.prepare_iepirkumi())
        if SKIP_EMBEDS:
            self.winner_embeds = np.load(WINNER_EMBEDDINGS_FNAME)
            self.active_embeds = np.load(ACTIVE_EMBEDDINGS_FNAME)
        else:
            pca = PCA(n_components=PCA_COMPONENTS)
            self.winner_embeds = embeddings.generate_embeddings_from_texts(
                self.titles_from_df(self.winners_df))
            self.winner_embeds = pca.fit_transform(self.winner_embeds)
            self.active_embeds = embeddings.generate_embeddings_from_texts(
                self.titles_from_df(self.active_df))
            self.active_embeds = pca.transform(self.active_embeds)
        self.create_clusters()
        self.company_profiles = self.create_company_profiles()

    @staticmethod
    def titles_from_df(df: pd.DataFrame) -> list[str]:
        titles = []
        for _, row in df.iterrows():
            if row["Ir_dalijums_dalas"] == "Nē":
                titles.append(row["Iepirkuma_nosaukums"])
            else:
                titles.append(row["Iepirkuma_dalas_nosaukums"])
        return titles

    def create_clusters(self):
        self.lg.info("Creating clusters...")
        kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_SATE)
        clusters = kmeans.fit_predict(self.winner_embeds)
        self.winners_df["cluster"] = clusters
        clusters = kmeans.predict(self.active_embeds)
        self.active_df["cluster"] = clusters

    def create_company_profiles(self) -> dict:
        self.lg.info("Creating company profiles...")
        company_profiles = {}
        for reg_no in self.winners_df[
                'Uzvaretaja_registracijas_numurs'].unique():
            company_wins = self.winners_df[
                self.winners_df['Uzvaretaja_registracijas_numurs'] == reg_no]
            cluster_counts = company_wins['cluster'].value_counts().to_dict()
            company_profiles[reg_no] = cluster_counts
        return company_profiles

    @staticmethod
    def score_procurement_for_company(company_profile, procurement_cluster):
        return company_profile.get(procurement_cluster, 0)

    def get_recommendations(self, reg_no, n_recommendations=25):
        self.lg.info(f"Getting recommendations for {reg_no} ...")
        company_profile = self.company_profiles.get(reg_no, {})
        scores = []
        ids = set()
        for _, procurement in self.active_df.iterrows():
            proc_id = procurement['Iepirkuma_ID']
            if proc_id in ids:
                continue
            ids.add(proc_id)
            score = self.score_procurement_for_company(company_profile,
                                                       procurement['cluster'])
            scores.append((procurement.to_dict(), score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[:n_recommendations]
        recommendations = [score[0] for score in scores]
        for i in range(len(recommendations)):
            recommendations[i]["Piedavajumu_iesniegsanas_datums"] = recommendations[i]["Piedavajumu_iesniegsanas_datums"].strftime("%d.%m.%Y")
        return recommendations

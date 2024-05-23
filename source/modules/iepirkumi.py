import logging
import requests
import datetime
import os

import pandas as pd

TMPDIR = "./tmp"
IEPIRKUMI_URL = "https://data.gov.lv/dati/dataset/f78dd9df-25fb-4e3e-8247-1db02684819a/resource/d7204b7f-0767-472e-b1c7-b85816992885/download/eis_e_iepirkumi_izsludinatie_{year}.csv"
WINNERS_URL = "https://data.gov.lv/dati/dataset/e909312a-61c9-4cde-a72a-0a09dd75ef43/resource/5040e052-58e0-4aca-bf6e-cf2d58e9c50c/download/eis_e_iepirkumi_rezultati_{year}.csv"

IEPIRKUMI_DF_FNAME = f"{TMPDIR}/iepirkumi_main.pkl"
WINNERS_DF_FNAME = f"{TMPDIR}/winners_main.pkl"
UNIQUE_WINNERS_NO_TO_NAME_FNAME = f"{TMPDIR}/unique_winners_no_to_name.pkl"
UNIQUE_WINNERS_NAME_TO_NO_FNAME = f"{TMPDIR}/unique_winners_name_to_no.pkl"

lg = logging.getLogger("iepirkumi")

iepirkumi_fields = [
    'Iepirkuma_ID',
    'Iepirkuma_nosaukums',
    'Iepirkuma_identifikacijas_numurs',
    'Pasutitaja_nosaukums',
    'Pasutitaja_registracijas_numurs',
    'CPV_kods_galvenais_prieksmets',
    'Iepirkuma_statuss',
    'Iepirkuma_izsludinasanas_datums',
    'Piedavajumu_iesniegsanas_datums',
    'Piedavajumu_iesniegsanas_laiks',
    'Hipersaite_EIS_kura_pieejams_zinojums',
    'Hipersaite_uz_IUB_publikaciju',
    'Ir_dalijums_dalas',
    'Iepirkuma_dalas_nr',
    'Iepirkuma_dalas_nosaukums',
]

winner_fields = [
    'Iepirkuma_ID',
    'Iepirkuma_nosaukums',
    'Iepirkuma_identifikacijas_numurs',
    'Proceduras_veids',
    'Hipersaite_EIS_kura_pieejams_zinojums',
    'Hipersaite_uz_IUB_publikaciju',
    'Ir_dalijums_dalas',
    'Iepirkuma_dalas_nr',
    'Iepirkuma_dalas_nosaukums',
    'Uzvaretaja_nosaukums',
    'Uzvaretaja_registracijas_numurs',
]


def download_iepirkumi() -> str:
    current_year = datetime.date.today().year
    today = datetime.date.today().strftime("%Y-%m-%d")
    fname = f"{TMPDIR}/iepirkumi_{today}.csv"
    url = IEPIRKUMI_URL.format(year=current_year)
    with requests.get(url, stream=True) as r:
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return fname


def prepare_iepirkumi() -> str:
    fname = download_iepirkumi()
    df = pd.read_csv(fname, sep=';', encoding='utf-8')
    df = df[iepirkumi_fields]
    strip_fields(df)
    df['Piedavajumu_iesniegsanas_datums'] = pd.to_datetime(
        df['Piedavajumu_iesniegsanas_datums'], format='%d.%m.%Y')
    today = datetime.datetime.now()
    df = df[df['Piedavajumu_iesniegsanas_datums'] > today]
    df = df[df['Iepirkuma_statuss'] == 'Izsludināts']
    df.to_pickle(IEPIRKUMI_DF_FNAME)
    return IEPIRKUMI_DF_FNAME


def download_winners(year: int) -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    url = WINNERS_URL.format(year=year)
    fname = f"{TMPDIR}/winners_{today}.csv"
    with requests.get(url, stream=True) as r:
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return fname


def prepare_winners() -> str:
    if os.environ.get("SKIP_DOWNLOAD_FILES") == "1":
        return WINNERS_DF_FNAME
    current_year = datetime.date.today().year
    years = (current_year - 2, current_year - 1, current_year)
    fnames = []
    for year in years:
        fnames.append(download_winners(year))
    dataframes = [
        pd.read_csv(filename, sep=';', encoding='utf-8') for filename in fnames
    ]
    # Concatenate the DataFrames into one
    df = pd.concat(dataframes, ignore_index=True)
    strip_fields(df)
    df = df[df["Liguma_dok_veids"] == "Līgums"]
    df = df[df["Uzvaretaja_registracijas_numurs"] != ""]
    df = df[winner_fields]
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(WINNERS_DF_FNAME)
    return WINNERS_DF_FNAME


def unique_winners(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    target_column = 'Uzvaretaja_registracijas_numurs'
    df_sorted = df.sort_values(by=[target_column])
    duplicates_mask = df_sorted.duplicated(subset=[target_column], keep='last')
    result_df = df_sorted[~duplicates_mask]
    result_df = result_df[[
        "Uzvaretaja_registracijas_numurs", "Uzvaretaja_nosaukums"
    ]]
    result_df = result_df[result_df["Uzvaretaja_registracijas_numurs"] != ""]
    result_df.reset_index(drop=True, inplace=True)

    reg_no_to_name: dict[str, str] = {}
    name_to_reg_no: dict[str, str] = {}
    for _, row in result_df.iterrows():
        reg_no_to_name[row["Uzvaretaja_registracijas_numurs"]] = row[
            "Uzvaretaja_nosaukums"]
        name_to_reg_no[row["Uzvaretaja_nosaukums"]] = row[
            "Uzvaretaja_registracijas_numurs"]
    return reg_no_to_name, name_to_reg_no


def strip_fields(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lstrip("=\"")
            df[col] = df[col].str.rstrip("\"")

# Piemēroto iepirkumu meklēšana izmantojot [LVBERT](https://github.com/LUMII-AILab/LVBERT)

> Projekts tika veidots Rīgas Tehniskās universitātes mācību kursa "Datu integrācijas un mākoņdatošanas seminārs" ietvaros.

!`requirements.txt` datne satur datus MacOS (Apple silicon) sistēmai. Uz citām platformām programmatūra netika testēta!

## .env datnes piemērs
```bash
export PORT=8059
export LOGLEVEL=DEBUG
export DEVMODE=1
export SKIP_DOWNLOAD_FILES=0 # Ja "1", izlaist iepirkumu datu lejupielādi
export SKIP_CREATING_EMBEDDINGS=0 # Ja "1", izlaist embedding izveidi
```

## Risinājuma apraksts
Datu ielasīšanai no eis.gov.lv portāla ir nepieciešama oficiāla atļauja no uzturētāja, tāpēc dati par aktīviem iepirkumiem un rezultātiem tika nolasīti no data.gov.lv portāla, kur reizi dienā tiek publicēt dati no EIS.
- https://data.gov.lv/dati/dataset/f78dd9df-25fb-4e3e-8247-1db02684819a/resource/d7204b7f-0767-472e-b1c7-b85816992885/download/eis_e_iepirkumi_izsludinatie_2024.csv
- https://data.gov.lv/dati/dataset/e909312a-61c9-4cde-a72a-0a09dd75ef43/resource/5040e052-58e0-4aca-bf6e-cf2d58e9c50c/download/eis_e_iepirkumi_rezultati_2024.csv

Iepirkumu analīzei kā aprakstošais atribūts tiek ņemts iepirkuma nosaukums. Atribūtu izgūšana (feature extraction) notiek izmantojot [LVBERT](https://github.com/LUMII-AILab/LVBERT) modeli, kas ir pieejams [Hugginface](https://huggingface.co/AiLab-IMCS-UL/lvbert) vietnē.
Tiek nolasīti dati par iepirkumu uzvarētājiem pēdējo 3 gadu laikā. Visiem iepirkumu nosaukumiem tiek izveidots vārdlietojuma kartējums (embedding), izmantojot LVBERT. Tiek samazināta kartējuma vektora dimensionalitāte, sākumā pielietojot tf.keras.layers.GlobalAveragePooling1D slāni, tad izmanto Principal Component Analysis (no Scikit learn bibliotēkas), lai samazinātu dimensiju daudzumu līdz vēlamajam rādītājam.
Tālāk iepirkumu nosaukumi no abām datu kopām tiek klasterēti, izmantojot K-vidējo algoritmu. Katram uzņēmumam no uzvarētāju datu kopas tiek izveidots uzvarētāja profils (cik uzvaras katrā klasterā).
Lietotājam pieprasot iepirkumu rekomendācijas konkrētam uzņēmumam, tiek atlasīts noteikts iepirkumu skaits balstoties uz uzvarētāja profilu.
Saskarnei tika izmantota vienkārša Python Flask lietotne ar vienu skatu, kur notiek gan meklēšana, gan rezultātu attēlošana, gan kļūdas attēlošana.

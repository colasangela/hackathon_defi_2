#!/usr/bin/env python3

import xarray as xr
import requests
import pandas as pd

# 1️⃣ Récupérer les données depuis l'API
url = "https://hubeau.eaufrance.fr/api/v1/temperature/station?code_departement=82&pretty&page=1&size=1000"
response = requests.get(url)
data_json = response.json()

# 2️⃣ Extraire la liste de stations
stations = data_json["data"]

# 3️⃣ Convertir en DataFrame
df = pd.DataFrame(stations)
df.to_csv("stations_toutes.csv", index=False, encoding="utf-8")


# Supposons que tu as déjà ton DataFrame des stations
# df_stations = pd.DataFrame(...)  

# Chercher les stations du Rhône
rhone_stations = df[df['libelle_cours_eau'].str.contains("Rhône", case=False, na=False)]

# Chercher les stations de la Garonne
garonne_stations = df[df['libelle_cours_eau'].str.contains("Garonne", case=False, na=False)]

# Corrélation entre Tair et débit eau

# Convertir en DataFrame
df = pd.DataFrame(stations)

# Filtrer sur tes codes
codes = [2900020, 9000010, 4010010, 1630020, 3030020, 4530010]
df = df[df["code_station"].isin(codes)]

# Exporter
df.to_csv("stations_loiret.csv", index=False, encoding="utf-8")
print(df.columns)

print(len(df))
print("Fichier CSV généré !")



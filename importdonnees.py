import pandas as pd

# --- Import des tables ---
temp = pd.read_csv("temperatures_journalieres_5_stations.csv")
temp_eau = pd.read_csv("tas_moyen_par_station_journalier.csv")
debit = pd.read_csv("debit_2009_2010.csv")

print("\nColonnes de temp :")
print(temp.columns.tolist())

print("\nColonnes de debit :")
print(debit.columns.tolist())


# --- Renommage des colonnes ---
temp = temp.rename(columns={
    'libelle_station': 'site_id',
    'date_mesure_temp': 'date',
    'resultat': 'T_fleuve'
})

debit = debit.rename(columns={
    'station': 'site_id',
    'time': 'date',
    'debit': 'Q'
})

temp_eau = temp_eau.rename(columns={
    'libelle_station': 'site_id',
    'time': 'date',
    'tas': 'T_fleuve'
})

# --- Dates ---
temp['date'] = pd.to_datetime(temp['date'])
debit['date'] = pd.to_datetime(debit['date'])
temp_eau['date'] = pd.to_datetime(temp_eau['date'])

# --- Vérification structure ---
print("\n=== Colonnes TEMP ===")
print(temp.columns.tolist())
print("Nb lignes temp :", len(temp))

print("\nStations dans temp :")
print(sorted(temp['site_id'].unique()))

print("\n=== Colonnes DEBIT ===")
print(debit.columns.tolist())
print("Nb lignes debit :", len(debit))

print("\nStations dans debit :")
print(sorted(debit['site_id'].unique()))

print("\n=== Colonnes TEMP_EAU ===")
print(temp_eau.columns.tolist())
print("Nb lignes temp_eau :", len(temp_eau))

print("\nStations dans temp_eau :")
print(sorted(temp_eau['site_id'].unique()))

print("\n=== Intersection des site_id entre TEMP et DEBIT ===")
print(set(temp['site_id']).intersection(set(debit['site_id'])))


print("\n=== Intersection des site_id entre TEMP_EAU et TEMP ===")
print(set(temp_eau['site_id']).intersection(set(debit['site_id'])))

print("\n=== site_id présents uniquement dans TEMP ===")
print(set(temp['site_id']) - set(debit['site_id']))

print("\n=== site_id présents uniquement dans TEMP_EAU ===")
print(set(temp_eau['site_id']) - set(temp['site_id']))

print("\n=== site_id présents uniquement dans DEBIT ===")
print(set(debit['site_id']) - set(temp['site_id']))

# --- Merge TEMP + DEBIT ---
df1 = pd.merge(temp, debit, on=['site_id', 'date'], how='left', indicator=True)

# --- Merge temp_eau + df1 ---
df = pd.merge(temp_eau, df1, on=['site_id', 'date'], how='left', indicator='merge_indicator')

# ----- 2) Vérification des valeurs manquantes -----
print("\n=== Nombre de valeurs manquantes par colonne ===")
print(df.isna().sum())

print("\n=== Nombre total de lignes contenant au moins un NaN ===")
print(df.isna().any(axis=1).sum())

print("\n=== Aperçu des lignes contenant au moins un NaN ===")
print(df[df.isna().any(axis=1)].head(20))

# ----- 3) Analyse de la provenance des lignes (indicator=True) -----
print("\n=== Répartition des lignes selon leur origine (merge_indicator) ===")
print(df['merge_indicator'].value_counts())

print("\n=== Lignes présentes uniquement dans temp_eau (pas dans df1) ===")
print(df[df['merge_indicator'] == 'left_only'].head(20))

print("\n=== Lignes présentes uniquement dans df1 (pas dans temp_eau) ===")
print(df[df['merge_indicator'] == 'right_only'].head(20))

# Supprimer toutes les lignes avec au moins un NaN
df = df.dropna()

# Vérification
print("Nb lignes après suppression des NaN :", len(df))
print("\nNombre de valeurs manquantes par colonne :")
print(df.isna().sum())

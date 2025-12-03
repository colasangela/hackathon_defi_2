import pandas as pd

# --- Import des tables ---
temp = pd.read_csv("temperatures_journalieres_5_stations.csv")
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


temp['date'] = pd.to_datetime(temp['date'])
debit['date'] = pd.to_datetime(debit['date'])


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


print("\n=== Intersection des site_id ===")
print(set(temp['site_id']).intersection(set(debit['site_id'])))

print("\n=== site_id présents uniquement dans TEMP ===")
print(set(temp['site_id']) - set(debit['site_id']))

print("\n=== site_id présents uniquement dans DEBIT ===")
print(set(debit['site_id']) - set(temp['site_id']))

df = pd.merge(temp, debit, on=['site_id', 'date'], how='left', indicator=True)


# ----- 2) Vérification des valeurs manquantes -----
print("\n=== Nombre de valeurs manquantes par colonne ===")
print(df.isna().sum())

print("\n=== Nombre total de lignes contenant au moins un NaN ===")
print(df.isna().any(axis=1).sum())

print("\n=== Aperçu des lignes contenant au moins un NaN ===")
print(df[df.isna().any(axis=1)].head(20))

# ----- 3) Analyse de la provenance des lignes (indicator=True) -----
print("\n=== Répartition des lignes selon leur origine ===")
print(df['_merge'].value_counts())

print("\n=== Lignes présentes uniquement dans temp (pas dans debit) ===")
print(df[df['_merge'] == 'left_only'].head(20))

print("\n=== Lignes présentes uniquement dans debit (pas dans temp) ===")
print(df[df['_merge'] == 'right_only'].head(20))


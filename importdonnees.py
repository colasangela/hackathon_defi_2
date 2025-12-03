import pandas as pd

# --- Import des tables ---
temp = pd.read_csv("temperatures_journalieres.csv")
debit = pd.read_csv("debit_2009_2010.csv")

# --- Renommage des colonnes ---
temp = temp.rename(columns={
    'libelle_station': 'site_id',
    'date_mesure_temps': 'date',
    'resultat': 'T_fleuve'
})

debit = debit.rename(columns={
    'station': 'site_id',
    'time': 'date',
    'debit': 'Q'
})


temp['date'] = pd.to_datetime(temp['date'])
debit['date'] = pd.to_datetime(debit['date'])


df = pd.merge(temp, debit, on=['site_id', 'date'], how='outer', indicator=True)


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
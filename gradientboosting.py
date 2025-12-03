# ---------------------------
# 0) Imports
# ---------------------------
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# --- Import des tables ---
temp = pd.read_csv("temperatures_journalieres_5_stations.csv")
temp_air = pd.read_csv("tas_moyen_par_station_journalier.csv")
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

temp_air = temp_air.rename(columns={
    'libelle_station': 'site_id',
    'time': 'date',
    'tas': 'T_air'
})

# --- Dates ---
temp['date'] = pd.to_datetime(temp['date'])
debit['date'] = pd.to_datetime(debit['date'])
temp_air['date'] = pd.to_datetime(temp_air['date'])

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

print("\n=== Colonnes temp_air ===")
print(temp_air.columns.tolist())
print("Nb lignes temp_air :", len(temp_air))

print("\nStations dans temp_air :")
print(sorted(temp_air['site_id'].unique()))

print("\n=== Intersection des site_id entre TEMP et DEBIT ===")
print(set(temp['site_id']).intersection(set(debit['site_id'])))


print("\n=== Intersection des site_id entre temp_air et TEMP ===")
print(set(temp_air['site_id']).intersection(set(debit['site_id'])))

print("\n=== site_id présents uniquement dans TEMP ===")
print(set(temp['site_id']) - set(debit['site_id']))

print("\n=== site_id présents uniquement dans temp_air ===")
print(set(temp_air['site_id']) - set(temp['site_id']))

print("\n=== site_id présents uniquement dans DEBIT ===")
print(set(debit['site_id']) - set(temp['site_id']))

# --- Merge TEMP + DEBIT ---
df1 = pd.merge(temp, debit, on=['site_id', 'date'], how='left', indicator=True)

# --- Merge temp_air + df1 ---
df = pd.merge(temp_air, df1, on=['site_id', 'date'], how='left', indicator='merge_indicator')

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

print("\n=== Lignes présentes uniquement dans temp_air (pas dans df1) ===")
print(df[df['merge_indicator'] == 'left_only'].head(20))

print("\n=== Lignes présentes uniquement dans df1 (pas dans temp_air) ===")
print(df[df['merge_indicator'] == 'right_only'].head(20))

# Supprimer toutes les lignes avec au moins un NaN
df = df.dropna()

# Vérification
print("Nb lignes après suppression des NaN :", len(df))
print("\nNombre de valeurs manquantes par colonne :")
print(df.isna().sum())



# ---------------------------
# 1) Chargement des données
# ---------------------------
# ATTENTION : adapte le chemin/lecture à tes données
# Ex : df = pd.read_csv("data_fleuve.csv", parse_dates=["date"])
if 'df' not in globals():
    raise RuntimeError("Définis d'abord le DataFrame `df` contenant les colonnes nécessaires.")

# Vérification colonnes obligatoires
required_cols = {'date','site_id','T_fleuve','T_air','Q'}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Colonnes manquantes dans df : {missing}")

# Formatage / tri par dates et stations
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date','site_id']).reset_index(drop=True)

# Si pas de 'scenario', on considère tout comme historique et on définira une date de split
if 'scenario' not in df.columns:
    df['scenario'] = 'historique'

# Ajouter colonnes temporelles auxiliaires
df['year'] = df['date'].dt.year # Extrait l’année de la date
df['dayofyear'] = df['date'].dt.dayofyear # Extrait le numéro du jour dans l’année (1 à 365 ou 366)

# ---------------------------
# 2) Feature engineering (lags, rolling, interactions)
# ---------------------------
def add_features(df):
    """
    Ajoute features utiles (opère sur l'ensemble multi-site).
    """
    df = df.copy()
    # Saisonnalité sin/cos
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.0)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.0)
    
    # Groupby site pour lags/rolling
    g = df.groupby('site_id', group_keys=False)
    
    # Lags et rolling de la température de l'eau
    df['T_fleuve_lag1'] = g['T_fleuve'].shift(1) # pour chaque jour, on obtient la température du fleuve du jour précédent.
    df['T_fleuve_lag2'] = g['T_fleuve'].shift(2) # pour chaque jour, on obtient la température du fleuve 2 jours avant.
    df['T_fleuve_rm7']  = g['T_fleuve'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    # et moyenne glissante sur 7 jours
    
    # Lags/rolling de T_air
    df['T_air_lag1'] = g['T_air'].shift(1)
    df['T_air_rm3']  = g['T_air'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Interaction amont-local et modulation par Q (éviter div par 0) où Q est le débit
    # Ces instructions créent des features dérivées combinant température et débit pour enrichir le modèle
    # df['delta_air'] = df['T_air'] - df['T_air_local']
    df['delta_air_over_Q'] = df['_air'T] / (df['Q'] + 1e-6)
    df['inv_Q'] = 1.0 / (df['Q'] + 1e-6)
    
    return df

df = add_features(df)

# ---------------------------
# 3) Sélection des datasets historic / futur
# ---------------------------
# Si 'scenario' == 'futur' présent : séparer directement
if 'futur' in df['scenario'].unique():
    hist_df = df[df['scenario'] == 'historique'].copy()
    futur_df = df[df['scenario'] == 'futur'].copy()
else:
    # Sinon : définir une date de split manuelle ; exemple : tout avant 2020 => historique, 2020+ => futur
    cut_date_future = pd.Timestamp('2020-01-01')  # adapte selon ton cas
    hist_df = df[df['date'] < cut_date_future].copy()
    futur_df = df[df['date'] >= cut_date_future].copy()

# Vérifier qu'on a des données historiques suffisantes
print("Période historique :", hist_df['date'].min(), "->", hist_df['date'].max())
print("Période future (scénarios) :", futur_df['date'].min(), "->", futur_df['date'].max())

# ---------------------------
# 4) Définir features + target
# ---------------------------
features = [
    'site_id',
    'T_air', 'T_air_lag1', 'T_air_rm3',
    'Q', 'inv_Q',
    'delta_air', 'delta_air_over_Q',
    'T_fleuve_lag1', 'T_fleuve_lag2', 'T_fleuve_rm7',
    'sin_doy', 'cos_doy'
]
target = 'T_fleuve'

# Vérifier colonnes
missing_feats = [c for c in features if c not in df.columns]
if missing_feats:
    raise RuntimeError(f"Features manquantes : {missing_feats}")

# Cast site_id en category
hist_df['site_id'] = hist_df['site_id'].astype('category')
futur_df['site_id'] = futur_df['site_id'].astype('category')

# Supprimer lignes d'entraînement avec NaN dans features/target
hist_model = hist_df.dropna(subset=features + [target]).copy()
print("Points historiques utilisables pour modèle :", len(hist_model))

# ---------------------------
# 5) Validation temporelle via TimeSeriesSplit sur les DATES
# ---------------------------
# Raisonnement :
# - On crée une TimeSeriesSplit sur la séquence des dates uniques (globales).
# - Pour chaque fold la partition est : train_dates / val_dates
# - On prend toutes les lignes dont 'date' est dans train_dates (resp. val_dates).
# Avantage : chaque fold est une séparation temporelle valide pour toutes les stations.

n_splits = 5
unique_dates = np.sort(hist_model['date'].unique())
tscv = TimeSeriesSplit(n_splits=n_splits)

# paramètres LightGBM de base
lgb_params = {
    'objective':'regression',
    'metric':'mae',
    'learning_rate':0.05,
    'num_leaves':31,
    'max_depth':6,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbose':-1,
    'seed': 42
}

cv_results = []
residuals_list = []  # on collecte résidus par fold pour estimer sigma

# entrainement et test sur les différents folds
fold_id = 0
for train_idx, val_idx in tscv.split(unique_dates):
    fold_id += 1
    train_dates = unique_dates[train_idx]
    val_dates = unique_dates[val_idx]
    
    # sélection des lignes
    train_rows = hist_model[hist_model['date'].isin(train_dates)].copy()
    val_rows = hist_model[hist_model['date'].isin(val_dates)].copy()
    
    # sécurité : si un fold a peu de données pour certains sites, on l'accepte mais l'utilisateur doit surveiller
    print(f"Fold {fold_id} — train_dates {train_dates.min()} -> {train_dates.max()}, val_dates {val_dates.min()} -> {val_dates.max()}")
    print("  lignes train:", len(train_rows), "lignes val:", len(val_rows))
    
    # Préparer datasets en les mettant au format requis pour LightGBM
    dtrain = lgb.Dataset(train_rows[features], label=train_rows[target], categorical_feature=['site_id'])
    dval = lgb.Dataset(val_rows[features], label=val_rows[target], reference=dtrain, categorical_feature=['site_id'])
    
    # Entraînement rapide par fold (early stopping)
    bst = lgb.train(lgb_params, dtrain, valid_sets=[dtrain, dval], num_boost_round=2000,
                    early_stopping_rounds=50, verbose_eval=False)
    
    # Prédiction sur val
    val_pred = bst.predict(val_rows[features])
    mae = mean_absolute_error(val_rows[target], val_pred)
    print(f"  Fold {fold_id} MAE : {mae:.4f}")
    cv_results.append({'fold': fold_id, 'mae': mae, 'train_n': len(train_rows), 'val_n': len(val_rows)})
    
    # Stocker résidus (pour estimation sigma)
    temp = val_rows[['site_id','date','dayofyear']].copy()
    temp['pred'] = val_pred
    temp['true'] = val_rows[target].values
    temp['resid'] = temp['true'] - temp['pred']
    residuals_list.append(temp)

# Agréger diagnostics CV (c'est à dire les métriques d'évaluation du modèle sur cette cross validation)
cv_df = pd.DataFrame(cv_results)
print("\nRésultats CV :")
print(cv_df)
print("MAE CV moyen :", cv_df['mae'].mean())

# Concatener tous les résidus collectés sur les folds
resid_all = pd.concat(residuals_list, axis=0).reset_index(drop=True)

# ---------------------------
# 6) Estimation sigma (erreur) par site × doy_window
# ---------------------------
# On utilise les résidus collectés pour estimer la variabilité selon la saison.
WINDOW_DAYS = 15
resid_all['doy_window'] = (resid_all['dayofyear'] // WINDOW_DAYS).astype(int)

sigma_df = (resid_all
            .groupby(['site_id','doy_window'])['resid']
            .std()
            .reset_index()
            .rename(columns={'resid':'sigma'}))

# fallback par site, puis global
sigma_site = (resid_all.groupby('site_id')['resid'].std().reset_index().rename(columns={'resid':'sigma_site'}))
global_sigma = resid_all['resid'].std()

print("sigma global (résidus CV):", global_sigma)

# Merge sigma sur futur
futur_df = futur_df.copy()
futur_df['doy_window'] = (futur_df['dayofyear'] // WINDOW_DAYS).astype(int)
futur_df = futur_df.merge(sigma_df, on=['site_id','doy_window'], how='left')
futur_df = futur_df.merge(sigma_site, on='site_id', how='left')
# Remplissage : sigma spécifique > sigma_site > global
futur_df['sigma'] = futur_df['sigma'].fillna(futur_df['sigma_site'])
futur_df['sigma'] = futur_df['sigma'].fillna(global_sigma)
futur_df.drop(columns=['sigma_site'], inplace=True)

# Attention : si futur_df contient dates hors des années vues en CV, les windows peuvent manquer -> fallback ok
print("Extrait sigma (quelques lignes) :")
print(futur_df[['site_id','date','doy_window','sigma']].head())

# ---------------------------
# 7) Entraînement final sur toute la période historique
# ---------------------------
# On ré-entraine un modèle sur tout l'historique (pour prédire le futur)
hist_full = hist_model.copy()
dtrain_full = lgb.Dataset(hist_full[features], label=hist_full[target], categorical_feature=['site_id'])

bst_full = lgb.train(lgb_params, dtrain_full, num_boost_round=bst.best_iteration or 100)  # utiliser best_iteration dernier modèle fold si dispo

# Prédictions sur historique (diagnostic) et futur
hist_full['pred_T'] = bst_full.predict(hist_full[features])
futur_df['pred_T'] = bst_full.predict(futur_df[features])

# ---------------------------
# 8) Fonctions : espérance + MC
# ---------------------------

# Fonctions adaptées pour intervalle d'années
# ---------------------------


def expected_days_interval(df_future, site, threshold, start_year, end_year):
    """
    Calcule l'espérance du nombre de jours par an > threshold sur un intervalle
    donné (start_year à end_year), renvoie un float : nombre moyen de jours/an.
    """
    sub = df_future[(df_future['site_id']==site) & 
                    (df_future['year']>=start_year) & 
                    (df_future['year']<=end_year)].copy()
    if sub.empty:
        return np.nan
    
    eps = 1e-8
    z = (threshold - sub['pred_T']) / (sub['sigma'] + eps)
    sub['p_exceed'] = 1.0 - norm.cdf(z)
    
    # somme journalière par année
    yearly_sum = sub.groupby('year')['p_exceed'].sum()
    
    # moyenne sur toutes les années de l'intervalle
    mean_days_per_year = yearly_sum.mean()
    return mean_days_per_year

def mc_counts_interval(df_future, site, threshold, start_year, end_year, n_mc=500, random_state=42):
    """
    Monte-Carlo : simule n_mc trajectoires T ~ N(pred_T, sigma^2) sur un intervalle
    d'années, retourne la distribution du nombre moyen de jours/an > threshold.
    """
    sub = df_future[(df_future['site_id']==site) & 
                    (df_future['year']>=start_year) & 
                    (df_future['year']<=end_year)].copy()
    if sub.empty:
        return np.array([])
    
    rng = np.random.default_rng(random_state)
    mus = sub['pred_T'].values
    sigs = sub['sigma'].values
    years = sub['year'].values
    unique_years = np.unique(years)
    
    # simulations
    sims = rng.normal(loc=mus.reshape(-1,1), scale=sigs.reshape(-1,1), size=(len(mus), n_mc))
    exceed = sims > threshold  # bool
    
    # somme par année puis moyenne sur l'intervalle
    mean_days = []
    for i in range(n_mc):
        counts_per_year = []
        for y in unique_years:
            mask = (years==y)
            counts_per_year.append(exceed[mask, i].sum())
        mean_days.append(np.mean(counts_per_year))
    return np.array(mean_days)  # distribution Monte-Carlo


# ---------------------------
# Exemple d'application : 2031–2050 (pour 2041)
# ---------------------------
# Exemple de dictionnaire de seuils par site (adapte-le)
sites = sorted(futur_df['site_id'].unique())
threshold = 25.0
# seuils = {s: threshold for s in sites}

start_year = 2031
end_year = 2050

results = []
mc_results = []

for s in sites:
    # espérance
    mean_days = expected_days_interval(futur_df, s, seuils[s], start_year, end_year)
    results.append({'site_id': s, 'mean_days_per_year': mean_days, 
                    'threshold': seuils[s]})
    
    # Monte-Carlo
    mc_arr = mc_counts_interval(futur_df, s, seuils[s], start_year, end_year, n_mc=500)
    if len(mc_arr) > 0:
        mc_results.append({'site_id': s,
                           'median': np.median(mc_arr),
                           'p2.5': np.percentile(mc_arr, 2.5),
                           'p97.5': np.percentile(mc_arr, 97.5),
                           'threshold': seuils[s]})

expected_interval_df = pd.DataFrame(results)
mc_interval_df = pd.DataFrame(mc_results)

print(f"Espérance nombre de jours/an supérieurs à {threshold} sur intervalle 2030-2050 pour chaque station :")
print(expected_interval_df)
print("\nMC (distribution) sur intervalle 2030-2050 :")
print(mc_interval_df)


# ---------------------------
# 10) Visualisation (exemple)
# ---------------------------

def plot_sites_mean_days(expected_df, mc_df=None):
    sites = expected_df['site_id'].values
    means = expected_df['mean_days_per_year'].values
    plt.figure(figsize=(10,5))
    if mc_df is not None and not mc_df.empty:
        lower, upper = [], []
        for s in sites:
            row = mc_df[mc_df['site_id']==s]
            if not row.empty:
                lower.append(row['p2.5'].values[0])
                upper.append(row['p97.5'].values[0])
            else:
                lower.append(np.nan)
                upper.append(np.nan)
        plt.errorbar(sites, means, yerr=[means - np.array(lower), np.array(upper) - means],
                     fmt='o', capsize=5, label='espérance ± IC 95% (MC)')
    else:
        plt.bar(sites, means, alpha=0.7, label='espérance')
    plt.ylabel("Nombre moyen de jours/an > seuil")
    plt.xlabel("Site")
    plt.title(f"Nombre moyen de jours/an > seuil sur intervalle {start_year}-{end_year}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_sites_mean_days(expected_interval_df, mc_interval_df)

# ---------------------------
# 8) Export résultats
# ---------------------------
expected_interval_df.to_csv("expected_days_per_year_by_site.csv", index=False)
mc_interval_df.to_csv("mc_summary_interval_by_site.csv", index=False)

print("Pipeline complet terminé — CSV et graphique générés.")
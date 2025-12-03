"""
Idée centrale :

Entraîner un modèle unique de régression journalière qui prédit la température de fleuve à chaque station (site, jour).
Pourquoi :

Un seul modèle → maintenance + entraînement rapides.

Tu peux ensuite tester n’importe quel seuil pour chaque centrale (calcul de jours dépassés par année).

Permet d’exploiter toutes les données (partage d’information entre sites) via une feature site_id.
"""

"""
Pipeline résumé

Construire dataset journalier multi-site : colonnes date, site_id, T_fleuve, T_air_local, T_air_amont, Q_fleuve, dayofyear, ...

Entraîner LightGBM (regression) pour prédire T_fleuve.

Pour chaque année future et pour chaque site + seuil :

prédire la température journalière 

convertir en nombre de jours dépassés (ou mieux, estimer probabilité p_d et sommer p_d pour obtenir l’espérance)

Estimer l'incertitude : méthode simple (MC avec résidus) ou quantile regression.
"""

"""
Format des features (rapide et efficace)

Par ligne (site × jour) :

site_id (catégorie) — LightGBM gère les catégoriques ; sinon one-hot

date / dayofyear (sin/cos)

T_air_local(t) et T_air_amont(t) — pour jours futurs, tu fournis les valeurs scénarisées

Q_fleuve mensuel upsamplé (repeat ou interpolation)

Lags (optionnels mais utiles) : T_fleuve_t-1, T_air_t-1, T_air_amont_t-1, rolling_mean_7

éventuelles features site spécifiques (distance amont, capacité centrale, etc.)
"""
"""
Format des features (rapide et efficace)

Par ligne (site × jour) :

site_id (catégorie) — LightGBM gère les catégoriques ; sinon one-hot

date / dayofyear (sin/cos)

T_air_local(t) et T_air_amont(t) — pour jours futurs, tu fournis les valeurs scénarisées

Q_fleuve mensuel upsamplé (repeat ou interpolation)

Lags (optionnels mais utiles) : T_fleuve_t-1, T_air_t-1, T_air_amont_t-1, rolling_mean_7

éventuelles features site spécifiques (distance amont, capacité centrale, etc.)
"""

"""
Validation (très important)

Time-series split (rolling) : ex. training jusqu’à fin 2016 → validate 2017 ; etc.

Évaluer à la fois : MAE quotidien & MAE sur counts annuels (|pred_count − true_count|).
"""

"""
Incertitude / probabilité d’excès (deux options)
Option 1 — rapide (déterministe)

Prédire T^d

Pour un seuil s, compter jours où T^d>s.

Rapide, mais ignore incertitude du modèle (risque de biais).
"""

# -*- coding: utf-8 -*-
"""
Notebook complet : prédiction T_eau par site (LightGBM) + TimeSeriesSplit (walk-forward)
+ estimation du nombre de jours dépassant un seuil (espérance + MC CI)

Usage :
- Charge ton DataFrame `df` avec au moins les colonnes :
  ['date','site_id','T_fleuve','T_air_local','T_air_amont','Q']
- Optionnel : colonne 'scenario' avec 'historique'/'futur' si tu veux séparer
- Exécute cellule par cellule.
"""

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

# ---------------------------
# 1) Chargement des données
# ---------------------------
# ATTENTION : adapte le chemin/lecture à tes données
# Ex : df = pd.read_csv("data_fleuve.csv", parse_dates=["date"])
if 'df' not in globals():
    raise RuntimeError("Définis d'abord le DataFrame `df` contenant les colonnes nécessaires.")

# Vérification colonnes obligatoires
required_cols = {'date','site_id','T_fleuve','T_air_local','T_air_amont','Q'}
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
    df['T_fleuve_lag1'] = g['T_fleuve'].shift(1)
    df['T_fleuve_lag2'] = g['T_fleuve'].shift(2)
    df['T_fleuve_rm7']  = g['T_fleuve'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Lags/rolling de T_air_local
    df['T_air_local_lag1'] = g['T_air_local'].shift(1)
    df['T_air_local_rm3']  = g['T_air_local'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Lags/rolling de T_air_amont
    df['T_air_amont_lag1'] = g['T_air_amont'].shift(1)
    df['T_air_amont_rm3']  = g['T_air_amont'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Interaction amont-local et modulation par Q (éviter div par 0)
    df['delta_air'] = df['T_air_amont'] - df['T_air_local']
    df['delta_air_over_Q'] = df['delta_air'] / (df['Q'] + 1e-6)
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
    'T_air_local', 'T_air_local_lag1', 'T_air_local_rm3',
    'T_air_amont', 'T_air_amont_lag1', 'T_air_amont_rm3',
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

fold_id = 0
for train_idx, val_idx in tscv.split(unique_dates):
    fold_id += 1
    train_dates = unique_dates[train_idx]
    val_dates = unique_dates[val_idx]
    
    # sélection des lignes
    train_rows = hist_model[hist_model['date'].isin(train_dates)].copy()
    val_rows   = hist_model[hist_model['date'].isin(val_dates)].copy()
    
    # sécurité : si un fold a peu de données pour certains sites, on l'accepte mais l'utilisateur doit surveiller
    print(f"Fold {fold_id} — train_dates {train_dates.min()} -> {train_dates.max()}, val_dates {val_dates.min()} -> {val_dates.max()}")
    print("  lignes train:", len(train_rows), "lignes val:", len(val_rows))
    
    # Préparer datasets LightGBM
    dtrain = lgb.Dataset(train_rows[features], label=train_rows[target], categorical_feature=['site_id'])
    dval   = lgb.Dataset(val_rows[features], label=val_rows[target], reference=dtrain, categorical_feature=['site_id'])
    
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

# Agréger diagnostics CV
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
def expected_days_for_threshold(df_future, site, threshold):
    """
    Calcule l'espérance du nombre de jours par année où T_eau > threshold pour un site.
    On suppose df_future contient colonnes : ['site_id','date','year','pred_T','sigma'].
    Retour : DataFrame (year, expected_days).
    """
    sub = df_future[df_future['site_id'] == site].copy()
    if sub.empty:
        return pd.DataFrame(columns=['year','expected_days'])
    eps = 1e-8
    z = (threshold - sub['pred_T']) / (sub['sigma'] + eps)
    sub['p_exceed'] = 1.0 - norm.cdf(z)
    res = sub.groupby('year')['p_exceed'].sum().reset_index().rename(columns={'p_exceed':'expected_days'})
    return res

def mc_counts(df_future, site, threshold, n_mc=500, random_state=42, batch_size=None):
    """
    Monte-Carlo : simule n_mc trajectoires (normales) T ~ N(pred_T, sigma^2).
    Retour : dict {year: array of counts length n_mc}
    Pour mémoire : si n_mc * n_days est grand, on peut simuler par batch.
    """
    sub = df_future[df_future['site_id'] == site].copy()
    if sub.empty:
        return {}
    rng = np.random.default_rng(random_state)
    mus = sub['pred_T'].values  # (n_days,)
    sigs = sub['sigma'].values
    years = sub['year'].values
    unique_years = np.unique(years)
    n_days = len(mus)
    
    # Gestion mémoire : si batch_size fourni -> faire par lots de simulations
    if batch_size is None:
        # simulation complète (n_days x n_mc)
        sims = rng.normal(loc=mus.reshape(-1,1), scale=sigs.reshape(-1,1), size=(n_days, n_mc))
        exceed = sims > threshold  # bool
        results = {}
        for y in unique_years:
            mask = (years == y)
            counts = exceed[mask, :].sum(axis=0)
            results[int(y)] = counts
        return results
    else:
        # simulation par batch de n_batch sims
        results_acc = {int(y): np.zeros(batch_size, dtype=int) for y in unique_years}  # initialisation dynamique
        results_full = {int(y): [] for y in unique_years}
        n_done = 0
        while n_done < n_mc:
            this_batch = min(batch_size, n_mc - n_done)
            sims = rng.normal(loc=mus.reshape(-1,1), scale=sigs.reshape(-1,1), size=(n_days, this_batch))
            exceed = sims > threshold
            for y in unique_years:
                mask = (years == y)
                counts = exceed[mask, :].sum(axis=0)
                results_full[int(y)].append(counts)
            n_done += this_batch
        # concat per year
        results = {y: np.hstack(results_full[y]) for y in unique_years}
        return results

# ---------------------------
# 9) Application : calculer espérance + MC pour tous les sites & seuils
# ---------------------------
# Exemple de dictionnaire de seuils par site (adapte-le)
sites = sorted(futur_df['site_id'].unique())
default_threshold = 28.0
seuils = {s: default_threshold for s in sites}

# Calcul des espérances
expected_list = []
for s in sites:
    res = expected_days_for_threshold(futur_df, s, seuils[s])
    if res.empty:
        continue
    res['site_id'] = s
    res['threshold'] = seuils[s]
    expected_list.append(res)
expected_df = pd.concat(expected_list, axis=0).reset_index(drop=True)
print("Extrait espérances :")
print(expected_df.head())

# Exemple MC pour un site
if len(sites) > 0:
    example_site = sites[0]
    mc_res = mc_counts(futur_df, example_site, seuils[example_site], n_mc=500)
    # résumé MC (median + 95% CI)
    mc_summary = []
    for year, arr in mc_res.items():
        mc_summary.append({'site_id': example_site,
                           'year': year,
                           'median': np.median(arr),
                           'p2.5': np.percentile(arr, 2.5),
                           'p97.5': np.percentile(arr, 97.5)})
    mc_summary = pd.DataFrame(mc_summary).sort_values(['site_id','year'])
    print(f"\nRésumé MC pour site {example_site} :")
    print(mc_summary.head())
else:
    mc_summary = pd.DataFrame()

# ---------------------------
# 10) Visualisation (exemple)
# ---------------------------
def plot_site_expected_and_ci(site, expected_df, mc_summary):
    exp = expected_df[expected_df['site_id'] == site].set_index('year').sort_index()
    mc = mc_summary[mc_summary['site_id'] == site].set_index('year').sort_index()
    plt.figure(figsize=(10,5))
    if not mc.empty:
        plt.fill_between(mc.index, mc['p2.5'], mc['p97.5'], alpha=0.25, label='IC 95% (MC)')
        plt.plot(mc.index, mc['median'], marker='o', linestyle='--', label='mediane (MC)')
    if not exp.empty:
        plt.plot(exp.index, exp['expected_days'], marker='o', label='espérance (somme p_j)')
    plt.title(f"Jours dépassant seuil - site {site}")
    plt.xlabel("Année")
    plt.ylabel("Nombre de jours")
    plt.legend()
    plt.show()

# Tracer pour site d'exemple si existant
if len(sites) > 0:
    plot_site_expected_and_ci(example_site, expected_df, mc_summary)

# ---------------------------
# 11) Export résultats
# ---------------------------
expected_df.to_csv("expected_days_per_year_by_site.csv", index=False)
if not mc_summary.empty:
    mc_summary.to_csv(f"mc_summary_{example_site}.csv", index=False)

print("Pipeline terminé — fichiers sauvegardés si souhaité.")

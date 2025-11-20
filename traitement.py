# ============================================
# ANALYSE ET PRÉDICTION DE PRIX DE PRODUITS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("=" * 80)
print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
print("=" * 80)

# Chargement
client = pd.read_csv('clients.csv', sep=';')
commande = pd.read_csv('commandes.csv', sep=';')
produit = pd.read_csv('produits.csv', sep=';')

print(f"\n✓ Clients : {client.shape[0]} lignes, {client.shape[1]} colonnes")
print(f"  Colonnes : {list(client.columns)}")
print(f"\n✓ Commandes : {commande.shape[0]} lignes, {commande.shape[1]} colonnes")
print(f"  Colonnes : {list(commande.columns)}")
print(f"\n✓ Produits : {produit.shape[0]} lignes, {produit.shape[1]} colonnes")
print(f"  Colonnes : {list(produit.columns)}")

# Aperçu des données
print("\n📊 Aperçu des premières lignes :")
print("\nClients :")
print(client.head(3))
print("\nCommandes :")
print(commande.head(3))
print("\nProduits :")
print(produit.head(3))

# ============================================
# 2. FUSION ET NETTOYAGE
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 2 : FUSION ET NETTOYAGE DES DONNÉES")
print("=" * 80)

# Fusion
df = commande.merge(client, on='client_id', how='left').merge(produit, on='produit_id', how='left')
print(f"\n Dataset fusionné : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Nettoyage
df['date_commande'] = pd.to_datetime(df['date_commande'])
df['email'] = df['email'].str.lower().str.strip()
df['ville'] = df['ville'].str.strip()
df['categorie'] = df['categorie'].str.strip()

# Vérification des valeurs manquantes
print("\n Valeurs manquantes :")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(pd.DataFrame({
        'Colonne': missing.index,
        'Manquantes': missing.values,
        'Pourcentage': (missing.values / len(df) * 100).round(2)
    })[missing.values > 0])
else:
    print(" Aucune valeur manquante !")

# Vérification des doublons
duplicates = df.duplicated().sum()
print(f"\n{'⚠️' if duplicates > 0 else '✓'} Doublons : {duplicates}")

# Statistiques de base
print("\n Statistiques des variables numériques :")
print(df[['quantite', 'montant', 'prix_unitaire']].describe())

# ============================================
# 3. ANALYSE EXPLORATOIRE DÉTAILLÉE
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 3 : ANALYSE EXPLORATOIRE")
print("=" * 80)

# Créer dossier pour graphiques
import os
os.makedirs('graphs', exist_ok=True)

# Analyse de la variable cible (montant)
print("\n Analyse du MONTANT :")
print(f"  • Moyenne : {df['montant'].mean():.2f}€")
print(f"  • Médiane : {df['montant'].median():.2f}€")
print(f"  • Min : {df['montant'].min():.2f}€")
print(f"  • Max : {df['montant'].max():.2f}€")
print(f"  • Écart-type : {df['montant'].std():.2f}€")

# Vérifier cohérence montant = quantité * prix_unitaire
df['montant_calcule'] = df['quantite'] * df['prix_unitaire']
df['diff_montant'] = abs(df['montant'] - df['montant_calcule'])
incoherent = (df['diff_montant'] > 0.01).sum()
print(f"\n{'⚠️' if incoherent > 0 else '✓'} Incohérences montant vs (quantité × prix) : {incoherent}")

# Top catégories
print("\n Top 10 Catégories de produits :")
top_categories = df['categorie'].value_counts().head(10)
print(top_categories)

# Top villes
print("\n Top 10 Villes :")
top_villes = df['ville'].value_counts().head(10)
print(top_villes)

# Visualisations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Distribution du montant
axes[0, 0].hist(df['montant'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Distribution du Montant', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Montant (€)')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].axvline(df['montant'].mean(), color='red', linestyle='--', label=f'Moyenne: {df["montant"].mean():.2f}€')
axes[0, 0].legend()

# 2. Distribution de la quantité
axes[0, 1].hist(df['quantite'], bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_title('Distribution de la Quantité', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Quantité')
axes[0, 1].set_ylabel('Fréquence')

# 3. Boxplot du montant
axes[0, 2].boxplot(df['montant'], vert=True)
axes[0, 2].set_title('Boxplot du Montant', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('Montant (€)')

# 4. Top 10 catégories
top_cat = df['categorie'].value_counts().head(10)
axes[1, 0].barh(range(len(top_cat)), top_cat.values, color='teal')
axes[1, 0].set_yticks(range(len(top_cat)))
axes[1, 0].set_yticklabels(top_cat.index)
axes[1, 0].set_title('Top 10 Catégories', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Nombre de commandes')

# 5. Montant moyen par catégorie (top 10)
montant_cat = df.groupby('categorie')['montant'].mean().sort_values(ascending=False).head(10)
axes[1, 1].barh(range(len(montant_cat)), montant_cat.values, color='purple')
axes[1, 1].set_yticks(range(len(montant_cat)))
axes[1, 1].set_yticklabels(montant_cat.index)
axes[1, 1].set_title('Montant Moyen par Catégorie (Top 10)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Montant moyen (€)')

# 6. Évolution du montant dans le temps
df_time = df.groupby(df['date_commande'].dt.to_period('M'))['montant'].mean()
axes[1, 2].plot(range(len(df_time)), df_time.values, marker='o', linewidth=2, color='green')
axes[1, 2].set_title('Évolution du Montant Moyen', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Période')
axes[1, 2].set_ylabel('Montant moyen (€)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/01_analyse_exploratoire.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegardé : graphs/01_analyse_exploratoire.png")
plt.close()

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 4 : FEATURE ENGINEERING")
print("=" * 80)

# Features temporelles
df['annee'] = df['date_commande'].dt.year
df['mois'] = df['date_commande'].dt.month
df['jour_semaine'] = df['date_commande'].dt.dayofweek
df['trimestre'] = df['date_commande'].dt.quarter
df['jour_mois'] = df['date_commande'].dt.day
df['semaine_annee'] = df['date_commande'].dt.isocalendar().week
df['est_weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
df['est_debut_mois'] = (df['jour_mois'] <= 10).astype(int)
df['est_fin_mois'] = (df['jour_mois'] >= 20).astype(int)

print("Features temporelles créées (9 features)")

# Features agrégées par CLIENT
print("\n Création des features CLIENT...")
client_agg = df.groupby('client_id').agg({
    'montant': ['count', 'mean', 'sum', 'std', 'min', 'max'],
    'quantite': ['sum', 'mean', 'std'],
    'commande_id': 'count'
}).reset_index()

client_agg.columns = ['client_id', 'nb_commandes_client', 'montant_moyen_client', 
                      'total_depense_client', 'montant_std_client', 'montant_min_client',
                      'montant_max_client', 'quantite_totale_client', 'quantite_moyenne_client',
                      'quantite_std_client', 'total_transactions']

# Calcul de la récence (jours depuis dernière commande)
derniere_commande = df.groupby('client_id')['date_commande'].max().reset_index()
derniere_commande.columns = ['client_id', 'derniere_commande']
derniere_commande['jours_depuis_derniere_commande'] = (
    pd.Timestamp.now() - derniere_commande['derniere_commande']
).dt.days

client_agg = client_agg.merge(derniere_commande[['client_id', 'jours_depuis_derniere_commande']], 
                              on='client_id', how='left')

# Nombre de catégories achetées par client
cat_par_client = df.groupby('client_id')['categorie'].nunique().reset_index()
cat_par_client.columns = ['client_id', 'nb_categories_client']
client_agg = client_agg.merge(cat_par_client, on='client_id', how='left')

df = df.merge(client_agg, on='client_id', how='left')
print(f" {len(client_agg.columns)-1} features CLIENT créées")

# Features agrégées par PRODUIT
print("\n Création des features PRODUIT...")
produit_agg = df.groupby('produit_id').agg({
    'montant': ['count', 'mean', 'std', 'sum'],
    'quantite': ['mean', 'sum', 'std']
}).reset_index()

produit_agg.columns = ['produit_id', 'popularite_produit', 'montant_moyen_produit',
                       'montant_std_produit', 'ca_total_produit', 'quantite_moyenne_produit',
                       'quantite_totale_produit', 'quantite_std_produit']

df = df.merge(produit_agg, on='produit_id', how='left')
print(f" {len(produit_agg.columns)-1} features PRODUIT créées")

# Features agrégées par CATÉGORIE
print("\n Création des features CATÉGORIE...")
categorie_agg = df.groupby('categorie').agg({
    'montant': ['count', 'mean', 'std'],
    'quantite': 'mean',
    'prix_unitaire': ['mean', 'std']
}).reset_index()

categorie_agg.columns = ['categorie', 'nb_ventes_categorie', 'montant_moyen_categorie',
                         'montant_std_categorie', 'quantite_moyenne_categorie',
                         'prix_moyen_categorie', 'prix_std_categorie']

df = df.merge(categorie_agg, on='categorie', how='left')
print(f"{len(categorie_agg.columns)-1} features CATÉGORIE créées")

# Features agrégées par VILLE
print("\n Création des features VILLE...")
ville_agg = df.groupby('ville').agg({
    'montant': ['count', 'mean', 'sum'],
    'client_id': 'nunique'
}).reset_index()

ville_agg.columns = ['ville', 'nb_commandes_ville', 'montant_moyen_ville',
                     'ca_total_ville', 'nb_clients_ville']

df = df.merge(ville_agg, on='ville', how='left')
print(f"✓ {len(ville_agg.columns)-1} features VILLE créées")

# Features de ratio
df['ratio_montant_vs_prix_unitaire'] = df['montant'] / (df['prix_unitaire'] + 1)
df['ratio_client_vs_categorie'] = df['montant_moyen_client'] / (df['montant_moyen_categorie'] + 1)
df['ratio_produit_vs_categorie'] = df['montant_moyen_produit'] / (df['montant_moyen_categorie'] + 1)

print("\n Features de ratio créées (3 features)")

print(f"\n🎯 TOTAL : {len(df.columns)} colonnes dans le dataset enrichi")

# ============================================
# 5. PRÉPARATION POUR LE MACHINE LEARNING
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 5 : PRÉPARATION POUR LE ML")
print("=" * 80)

# Variable cible
target = 'montant'

# Sélection des features numériques
numeric_features = [
    # Temporelles
    'annee', 'mois', 'jour_semaine', 'trimestre', 'jour_mois', 'semaine_annee',
    'est_weekend', 'est_debut_mois', 'est_fin_mois',
    # Commande
    'quantite', 'prix_unitaire',
    # Client
    'nb_commandes_client', 'montant_moyen_client', 'total_depense_client',
    'montant_std_client', 'quantite_totale_client', 'quantite_moyenne_client',
    'jours_depuis_derniere_commande', 'nb_categories_client',
    # Produit
    'popularite_produit', 'montant_moyen_produit', 'montant_std_produit',
    'quantite_moyenne_produit',
    # Catégorie
    'nb_ventes_categorie', 'montant_moyen_categorie', 'quantite_moyenne_categorie',
    'prix_moyen_categorie',
    # Ville
    'nb_commandes_ville', 'montant_moyen_ville', 'nb_clients_ville',
    # Ratios
    'ratio_montant_vs_prix_unitaire', 'ratio_client_vs_categorie', 
    'ratio_produit_vs_categorie'
]

# Encodage des variables catégorielles
categorical_features = ['categorie', 'ville']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    numeric_features.append(f'{col}_encoded')

print(f"\n Variables catégorielles encodées : {categorical_features}")

# Sélection finale des features
features_to_use = [f for f in numeric_features if f in df.columns]
print(f"\n Features sélectionnées : {len(features_to_use)}")
print(f"\nListe des features :")
for i, f in enumerate(features_to_use, 1):
    print(f"  {i:2d}. {f}")

# Suppression des lignes avec valeurs manquantes
df_clean = df[features_to_use + [target]].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

print(f"\n Dataset nettoyé : {len(df_clean)} lignes (suppression de {len(df) - len(df_clean)} lignes)")

# Préparation X et y
X = df_clean[features_to_use]
y = df_clean[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n Données d'entraînement : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f" Données de test : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Normalisation appliquée (StandardScaler)")

# ============================================
# 6. ENTRAÎNEMENT DES MODÈLES
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 6 : ENTRAÎNEMENT DES MODÈLES")
print("=" * 80)

models = {
    'Régression Linéaire': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, 
        max_depth=20,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n🤖 Entraînement : {name}...")
    
    # Entraînement
    if name in ['Régression Linéaire', 'Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Métriques
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_pred_test
    }
    
    print(f"  Train - RMSE: {train_rmse:.2f}€ | MAE: {train_mae:.2f}€ | R²: {train_r2:.4f}")
    print(f"  Test  - RMSE: {test_rmse:.2f}€ | MAE: {test_mae:.2f}€ | R²: {test_r2:.4f}")
    print(f"  {'✓ Pas de surapprentissage' if abs(train_r2 - test_r2) < 0.1 else '⚠️ Surapprentissage détecté'}")

# ============================================
# 7. COMPARAISON DES MODÈLES
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 7 : COMPARAISON DES MODÈLES")
print("=" * 80)

# Tableau comparatif
comparison_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'RMSE (Test)': [results[m]['test_rmse'] for m in results],
    'MAE (Test)': [results[m]['test_mae'] for m in results],
    'R² (Test)': [results[m]['test_r2'] for m in results],
    'R² (Train)': [results[m]['train_r2'] for m in results]
})

print("\n Résultats comparatifs :")
print(comparison_df.to_string(index=False))

# Meilleur modèle basé sur R² test
best_model_name = comparison_df.loc[comparison_df['R² (Test)'].idxmax(), 'Modèle']
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['test_r2']

print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   R² (Test) = {best_r2:.4f}")
print(f"   RMSE (Test) = {results[best_model_name]['test_rmse']:.2f}€")
print(f"   MAE (Test) = {results[best_model_name]['test_mae']:.2f}€")

# Graphique de comparaison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (name, result) in enumerate(results.items()):
    if idx < 5:
        axes[idx].scatter(y_test, result['predictions'], alpha=0.5, s=20)
        axes[idx].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Prédiction parfaite')
        axes[idx].set_xlabel('Montant Réel (€)', fontsize=11)
        axes[idx].set_ylabel('Montant Prédit (€)', fontsize=11)
        axes[idx].set_title(f'{name}\nR² = {result["test_r2"]:.4f} | RMSE = {result["test_rmse"]:.2f}€',
                           fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

# Graphique de comparaison des métriques
ax = axes[5]
x_pos = np.arange(len(results))
width = 0.25

r2_scores = [results[m]['test_r2'] for m in results]
ax.bar(x_pos, r2_scores, width, label='R²', alpha=0.8)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_xlabel('Modèle', fontsize=11)
ax.set_title('Comparaison des Scores R²', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(list(results.keys()), rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graphs/02_comparaison_modeles.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : graphs/02_comparaison_modeles.png")
plt.close()

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n" + "=" * 80)
    print("ÉTAPE 8 : IMPORTANCE DES FEATURES")
    print("=" * 80)
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features_to_use,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top 20 features les plus importantes :")
    print(feature_importance_df.head(20).to_string(index=False))
    
    # Graphique
    plt.figure(figsize=(12, 8))
    top_20 = feature_importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['Importance'], color='steelblue')
    plt.yticks(range(len(top_20)), top_20['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 20 Features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/03_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé : graphs/03_feature_importance.png")
    plt.close()

# ============================================
# 9. ANALYSE DES ERREURS
# ============================================
print("\n" + "=" * 80)
print("ÉTAPE 9 : ANALYSE DES ERREURS")
print("=" * 80)

# Erreurs du meilleur modèle
if best_model_name in ['Régression Linéaire', 'Ridge', 'Lasso']:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

errors = y_test - y_pred_best
errors_pct = (errors / y_test) * 100

print(f"\n📊 Statistiques des erreurs :")
print(f"  • Erreur moyenne : {errors.mean():.2f}€")
print(f"  • Erreur médiane : {errors.median():.2f}€")
print(f"  • Erreur absolue moyenne : {abs(errors).mean():.2f}€")
print(f"  • Erreur max : {errors.max():.2f}€")
print(f"  • Erreur min : {errors.min():.2f}€")
print(f"  • % d'erreur moyenne : {abs(errors_pct).mean():.1f}%")

# Distribution des erreurs
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Erreur = 0')
axes[0].set_xlabel('Erreur (€)')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Distribution des Erreurs de Prédiction', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, errors, alpha=0.5, s=20)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Montant Réel (€)')
axes[1].set_ylabel('Erreur (€)')
axes[1].set_title('Erreurs vs Montant Réel', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/04_analyse_erreurs.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : graphs/04_analyse_erreurs.png")
plt.close()

# ============================================
# 10. SAUVEGARDE DU MODÈLE
# ============================================

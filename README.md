# 💰 Système de Prédiction de Prix de Produits

Projet complet d'analyse de données et de machine learning pour prédire les montants des commandes.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)

---

## 🎯 Vue d'ensemble

Ce projet permet de :
- ✅ Analyser l'historique des ventes
- ✅ Créer des features avancées (temporelles, comportementales, géographiques)
- ✅ Entraîner et comparer plusieurs modèles ML
- ✅ Prédire le montant d'une commande avec précision
- ✅ Utiliser une application web interactive

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étape 1 : Cloner le projet

```bash
cd C:\Users\Michael\Documents\projet_daina\application_prediction_produit
```

### Étape 2 : Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📊 Utilisation

### Option 1 : Analyse complète et entraînement du modèle

Exécutez le script d'analyse principal :

```bash
python analyse_complete.py
```

**Ce script va :**
1. Charger et fusionner vos 3 fichiers CSV
2. Effectuer une analyse exploratoire complète
3. Créer 40+ features automatiquement
4. Entraîner 5 modèles différents
5. Comparer les performances
6. Sauvegarder le meilleur modèle
7. Générer 4 graphiques d'analyse
8. Créer un rapport final

**Durée estimée :** 2-5 minutes

**Fichiers générés :**
```
models/
├── best_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── metadata.json
└── dataset_stats.json

graphs/
├── 01_analyse_exploratoire.png
├── 02_comparaison_modeles.png
├── 03_feature_importance.png
└── 04_analyse_erreurs.png

rapport_final.txt
```

### Option 2 : Lancer l'application web

Une fois le modèle entraîné, lancez l'application Streamlit :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

## 🏗️ Structure du Projet

```
application_prediction_produit/
│
├── clients.csv              # Données clients
├── commandes.csv            # Données commandes
├── produits.csv             # Données produits
│
├── analyse_complete.py      # Script principal d'analyse
├── app.py                   # Application web Streamlit
├── requirements.txt         # Dépendances Python
├── README.md               # Ce fichier
│
├── models/                 # Modèles entraînés
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── metadata.json
│   └── dataset_stats.json
│
└── graphs/                 # Graphiques générés
    ├── 01_analyse_exploratoire.png
    ├── 02_comparaison_modeles.png
    ├── 03_feature_importance.png
    └── 04_analyse_erreurs.png
```

## 🔬 Méthodologie

### 1. Chargement et Fusion des Données

- **clients.csv** : client_id, nom, email, ville
- **commandes.csv** : commande_id, client_id, produit_id, date_commande, quantite, montant
- **produits.csv** : produit_id, nom_produit, categorie, prix_unitaire

### 2. Feature Engineering (40+ features créées)

#### Features Temporelles (9)
- Année, mois, jour de la semaine, trimestre
- Semaine de l'année, jour du mois
- Est weekend, est début de mois, est fin de mois

#### Features Client (12)
- Nombre de commandes
- Montant moyen, total, min, max, écart-type
- Quantité totale, moyenne, écart-type
- Jours depuis dernière commande
- Nombre de catégories achetées

#### Features Produit (7)
- Popularité (nombre de ventes)
- Montant moyen, écart-type, CA total
- Quantité moyenne, totale, écart-type

#### Features Catégorie (6)
- Nombre de ventes
- Montant moyen, écart-type
- Quantité moyenne
- Prix moyen, écart-type

#### Features Ville (4)
- Nombre de commandes
- Montant moyen, CA total
- Nombre de clients

#### Features Dérivées (3)
- Ratio montant vs prix unitaire
- Ratio client vs catégorie
- Ratio produit vs catégorie

### 3. Modèles Entraînés

| Modèle | Description | Utilisation |
|--------|-------------|-------------|
| **Régression Linéaire** | Baseline simple | Référence |
| **Ridge** | Régression avec régularisation L2 | Évite le surapprentissage |
| **Lasso** | Régression avec régularisation L1 | Sélection de features |
| **Random Forest** | Ensemble de 100 arbres | Performance robuste |
| **Gradient Boosting** | Boosting séquentiel | Haute précision |

### 4. Métriques d'Évaluation

- **R² Score** : Variance expliquée (0 à 1, plus proche de 1 = meilleur)
- **RMSE** : Erreur quadratique moyenne (en €)
- **MAE** : Erreur absolue moyenne (en €)

## 📈 Résultats Attendus

Avec des données typiques de e-commerce :

- **R² > 0.85** : Le modèle explique >85% de la variance
- **RMSE < 20€** : Erreur moyenne de prédiction
- **MAE < 15€** : Erreur absolue moyenne

## 🎨 Fonctionnalités de l'Application Web

### Page 1 : Prédiction 🎯
- Sélection interactive client/produit
- Paramètres de commande (quantité, date)
- Prédiction en temps réel
- Intervalle de confiance
- Comparaison avec estimation simple

### Page 2 : Tableau de Bord 📊
- KPIs (CA, panier moyen, nombre de clients)
- Top 10 catégories par CA
- Top 10 villes par CA
- Évolution temporelle du CA

### Page 3 : Analyses 📈
- **Onglet Clients** : Top clients, historique
- **Onglet Produits** : Produits les plus vendus
- **Onglet Tendances** : Ventes par jour de la semaine

### Page 4 : À propos ℹ️
- Informations sur le modèle
- Métriques de performance
- Structure des données

## 🔧 Utilisation de la Fonction de Prédiction

### En Python

```python
import pickle
import pandas as pd

# Charger le modèle
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Faire une prédiction
from analyse_complete import predict_price

result = predict_price(
    client_id=123,
    produit_id=456,
    quantite=2,
    date_commande='2024-06-15'
)

print(f"Montant prédit : {result['montant_predit']:.2f}€")
print(f"Intervalle : [{result['intervalle_min']:.2f}€, {result['intervalle_max']:.2f}€]")
```

## 📝 Exemples d'Utilisation

### Exemple 1 : Prédire pour un nouveau client

```python
# Un nouveau client commande 3 unités d'un produit populaire
prediction = predict_price(
    client_id=999,  # Nouveau client
    produit_id=10,  # Produit populaire
    quantite=3
)
```

### Exemple 2 : Analyser l'impact de la quantité

```python
for qty in [1, 5, 10, 20]:
    result = predict_price(
        client_id=123,
        produit_id=456,
        quantite=qty
    )
    print(f"Quantité {qty}: {result['montant_predit']:.2f}€")
```

### Exemple 3 : Comparer plusieurs produits

```python
produits_ids = [10, 20, 30, 40]
for pid in produits_ids:
    result = predict_price(
        client_id=123,
        produit_id=pid,
        quantite=1
    )
    print(f"Produit {pid}: {result['montant_predit']:.2f}€")
```

## 🐛 Résolution de Problèmes

### Erreur : "FileNotFoundError"


**Problème :** Les fichiers CSV ne sont pas trouvés.

**Solution :**
```bash
# Vérifier que vous êtes dans le bon répertoire
cd C:\Users\Michael\Documents\projet_daina\application_prediction_produit

# Vérifier la présence des fichiers
dir clients.csv commandes.csv produits.csv
```

### Erreur : "Module not found"

**Problème :** Dépendances manquantes.

**Solution :**
```bash
pip install -r requirements.txt
```

### Erreur : "Model file not found"

**Problème :** Le modèle n'a pas été entraîné.

**Solution :**
```bash
# D'abord entraîner le modèle
python analyse_complete.py

# Puis lancer l'application
streamlit run app.py
```

## 📊 Interprétation des Résultats

### R² Score
- **0.9-1.0** : Excellent (90-100% de variance expliquée)
- **0.8-0.9** : Très bon
- **0.7-0.8** : Bon
- **<0.7** : À améliorer

### RMSE / MAE
- Plus c'est bas, mieux c'est
- Comparez avec le montant moyen de vos commandes
- Exemple : si montant moyen = 100€ et RMSE = 15€, l'erreur est de 15%

## 🎓 Pour Aller Plus Loin

### Améliorer le Modèle

1. **Ajouter plus de données** : Plus d'historique = meilleures prédictions
2. **Feature engineering avancé** : 
   - Saisonnalité
   - Promotions
   - Météo
   - Événements
3. **Hyperparameter tuning** : Optimiser les paramètres du modèle
4. **Deep Learning** : Réseaux de neurones pour relations complexes

### Déploiement en Production

1. **API REST** : Créer une API avec FastAPI ou Flask
2. **Docker** : Conteneuriser l'application
3. **Cloud** : Déployer sur AWS, Azure ou GCP
4. **Monitoring** : Suivre les performances en temps réel

## 📞 Support

Pour toute question ou problème :

1. Consultez ce README
2. Vérifiez les fichiers de log générés
3. Examinez le fichier `rapport_final.txt` après l'entraînement

## 📄 Licence

Ce projet est fourni à des fins éducatives et professionnelles.

---

**Créé avec ❤️ | Machine Learning & Data Science**

*Dernière mise à jour : Novembre 2024*
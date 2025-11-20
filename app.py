import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Prix - Produits",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Chargement des données et modèles
@st.cache_data
def load_data():
    """Charger les données CSV"""
    try:
        client = pd.read_csv('fusion_clients_commandes_produits\clients.csv', sep=';')
        commande = pd.read_csv('fusion_clients_commandes_produits\commandes.csv', sep=';')
        produit = pd.read_csv('fusion_clients_commandes_produits\produits.csv', sep=';')
        
        # Fusion
        df = commande.merge(client, on='client_id', how='left').merge(produit, on='produit_id', how='left')
        df['date_commande'] = pd.to_datetime(df['date_commande'])
        
        return client, commande, produit, df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, None, None, None

@st.cache_resource
def load_models():
    """Charger le modèle et les composants"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, encoders, metadata
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None, None, None

# Charger les données
client, commande, produit, df = load_data()
model, scaler, encoders, metadata = load_models()

if df is None or model is None:
    st.error("⚠️ Impossible de charger les données ou le modèle. Assurez-vous que les fichiers sont présents.")
    st.stop()

# Fonction de prédiction
def predict_price(client_id, produit_id, quantite, date_commande=None):
    """Prédire le montant d'une commande"""
    
    if date_commande is None:
        date_commande = pd.Timestamp.now()
    else:
        date_commande = pd.to_datetime(date_commande)
    
    # Récupérer les infos
    client_info = client[client['client_id'] == client_id]
    if len(client_info) == 0:
        return {"error": "Client non trouvé"}
    
    produit_info = produit[produit['produit_id'] == produit_id]
    if len(produit_info) == 0:
        return {"error": "Produit non trouvé"}
    
    ville = client_info.iloc[0]['ville']
    categorie = produit_info.iloc[0]['categorie']
    prix_unitaire = produit_info.iloc[0]['prix_unitaire']
    nom_produit = produit_info.iloc[0]['nom_produit']
    nom_client = client_info.iloc[0]['nom']
    
    # Calculer les features agrégées
    client_commandes = df[df['client_id'] == client_id]
    produit_commandes = df[df['produit_id'] == produit_id]
    categorie_commandes = df[df['categorie'] == categorie]
    ville_commandes = df[df['ville'] == ville]
    
    # Features temporelles
    features = {
        'annee': date_commande.year,
        'mois': date_commande.month,
        'jour_semaine': date_commande.dayofweek,
        'trimestre': date_commande.quarter,
        'jour_mois': date_commande.day,
        'semaine_annee': date_commande.isocalendar()[1],
        'est_weekend': 1 if date_commande.dayofweek in [5, 6] else 0,
        'est_debut_mois': 1 if date_commande.day <= 10 else 0,
        'est_fin_mois': 1 if date_commande.day >= 20 else 0,
        'quantite': quantite,
        'prix_unitaire': prix_unitaire,
    }
    
    # Features client
    if len(client_commandes) > 0:
        features.update({
            'nb_commandes_client': len(client_commandes),
            'montant_moyen_client': client_commandes['montant'].mean(),
            'total_depense_client': client_commandes['montant'].sum(),
            'montant_std_client': client_commandes['montant'].std() if len(client_commandes) > 1 else 0,
            'montant_min_client': client_commandes['montant'].min(),
            'montant_max_client': client_commandes['montant'].max(),
            'quantite_totale_client': client_commandes['quantite'].sum(),
            'quantite_moyenne_client': client_commandes['quantite'].mean(),
            'quantite_std_client': client_commandes['quantite'].std() if len(client_commandes) > 1 else 0,
            'total_transactions': len(client_commandes),
            'jours_depuis_derniere_commande': (pd.Timestamp.now() - client_commandes['date_commande'].max()).days,
            'nb_categories_client': client_commandes['categorie'].nunique(),
        })
    else:
        features.update({
            'nb_commandes_client': 0, 'montant_moyen_client': df['montant'].mean(),
            'total_depense_client': 0, 'montant_std_client': 0, 'montant_min_client': 0,
            'montant_max_client': 0, 'quantite_totale_client': 0, 'quantite_moyenne_client': 0,
            'quantite_std_client': 0, 'total_transactions': 0,
            'jours_depuis_derniere_commande': 365, 'nb_categories_client': 0,
        })
    
    # Features produit
    if len(produit_commandes) > 0:
        features.update({
            'popularite_produit': len(produit_commandes),
            'montant_moyen_produit': produit_commandes['montant'].mean(),
            'montant_std_produit': produit_commandes['montant'].std() if len(produit_commandes) > 1 else 0,
            'ca_total_produit': produit_commandes['montant'].sum(),
            'quantite_moyenne_produit': produit_commandes['quantite'].mean(),
            'quantite_totale_produit': produit_commandes['quantite'].sum(),
            'quantite_std_produit': produit_commandes['quantite'].std() if len(produit_commandes) > 1 else 0,
        })
    else:
        features.update({
            'popularite_produit': 0, 'montant_moyen_produit': prix_unitaire,
            'montant_std_produit': 0, 'ca_total_produit': 0,
            'quantite_moyenne_produit': 1, 'quantite_totale_produit': 0,
            'quantite_std_produit': 0,
        })
    
    # Features catégorie
    features.update({
        'nb_ventes_categorie': len(categorie_commandes),
        'montant_moyen_categorie': categorie_commandes['montant'].mean(),
        'montant_std_categorie': categorie_commandes['montant'].std(),
        'quantite_moyenne_categorie': categorie_commandes['quantite'].mean(),
        'prix_moyen_categorie': categorie_commandes['prix_unitaire'].mean(),
        'prix_std_categorie': categorie_commandes['prix_unitaire'].std(),
    })
    
    # Features ville
    features.update({
        'nb_commandes_ville': len(ville_commandes),
        'montant_moyen_ville': ville_commandes['montant'].mean(),
        'ca_total_ville': ville_commandes['montant'].sum(),
        'nb_clients_ville': ville_commandes['client_id'].nunique(),
    })
    
    # Ratios
    features.update({
        'ratio_montant_vs_prix_unitaire': quantite,
        'ratio_client_vs_categorie': features['montant_moyen_client'] / (features['montant_moyen_categorie'] + 1),
        'ratio_produit_vs_categorie': features['montant_moyen_produit'] / (features['montant_moyen_categorie'] + 1),
    })
    
    # Encodage
    features['categorie_encoded'] = encoders['categorie'].transform([categorie])[0] if categorie in encoders['categorie'].classes_ else -1
    features['ville_encoded'] = encoders['ville'].transform([ville])[0] if ville in encoders['ville'].classes_ else -1
    
    # Créer DataFrame
    X_new = pd.DataFrame([features])[metadata['features']]
    X_new = X_new.fillna(0)
    
    # Prédiction
    if metadata['requires_scaling']:
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)[0]
    else:
        prediction = model.predict(X_new)[0]
    
    rmse = metadata['test_rmse']
    
    return {
        'montant_predit': max(0, prediction),
        'intervalle_min': max(0, prediction - rmse),
        'intervalle_max': prediction + rmse,
        'categorie': categorie,
        'ville': ville,
        'prix_unitaire': prix_unitaire,
        'quantite': quantite,
        'nom_produit': nom_produit,
        'nom_client': nom_client,
        'estimation_simple': prix_unitaire * quantite
    }

# ============================================
# INTERFACE PRINCIPALE
# ============================================

# En-tête
st.markdown('<h1 class="main-header">💰 Prédiction de Prix de Produits</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
    st.title("Navigation")
    page = st.radio("", ["🎯 Prédiction", "📊 Tableau de bord", "📈 Analyses", "ℹ️ À propos"])
    
    st.markdown("---")
    st.markdown(f"""
    **Modèle utilisé:**  
    {metadata['best_model_name']}
    
    **Performance:**  
    R² = {metadata['test_r2']:.3f}  
    RMSE = {metadata['test_rmse']:.2f}€
    """)

# ============================================
# PAGE 1 : PRÉDICTION
# ============================================
if page == "🎯 Prédiction":
    st.header("🎯 Faire une Prédiction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Paramètres de la commande")
        
        # Sélection du client
        client_list = client['client_id'].tolist()
        client_names = client.set_index('client_id')['nom'].to_dict()
        client_options = [f"{cid} - {client_names[cid]}" for cid in client_list]
        
        selected_client = st.selectbox(
            "🧑 Sélectionner un client",
            options=client_options,
            index=0
        )
        client_id = int(selected_client.split(" - ")[0])
        
        # Afficher infos client
        client_info = client[client['client_id'] == client_id].iloc[0]
        st.info(f"📧 Email: {client_info['email']} | 🏙️ Ville: {client_info['ville']}")
        
        # Sélection du produit
        produit_list = produit['produit_id'].tolist()
        produit_names = produit.set_index('produit_id')['nom_produit'].to_dict()
        produit_options = [f"{pid} - {produit_names[pid]}" for pid in produit_list]
        
        selected_produit = st.selectbox(
            "📦 Sélectionner un produit",
            options=produit_options,
            index=0
        )
        produit_id = int(selected_produit.split(" - ")[0])
        
        # Afficher infos produit
        produit_info = produit[produit['produit_id'] == produit_id].iloc[0]
        st.info(f"🏷️ Catégorie: {produit_info['categorie']} | 💵 Prix unitaire: {produit_info['prix_unitaire']:.2f}€")
        
        # Quantité
        quantite = st.number_input(
            "📊 Quantité",
            min_value=1,
            max_value=1000,
            value=1,
            step=1
        )
        
        # Date
        date_commande = st.date_input(
            "📅 Date de la commande",
            value=datetime.now(),
            min_value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() + timedelta(days=365)
        )
        
        # Bouton de prédiction
        if st.button("🚀 Prédire le montant", type="primary", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                result = predict_price(client_id, produit_id, quantite, date_commande)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.session_state['last_prediction'] = result
    
    with col2:
        st.subheader("📊 Statistiques")
        
        # Stats client
        client_orders = df[df['client_id'] == client_id]
        if len(client_orders) > 0:
            st.metric("Commandes", len(client_orders))
            st.metric("Dépense totale", f"{client_orders['montant'].sum():.2f}€")
            st.metric("Panier moyen", f"{client_orders['montant'].mean():.2f}€")
        else:
            st.info("Nouveau client - Aucun historique")
    
    # Affichage du résultat
    if 'last_prediction' in st.session_state:
        result = st.session_state['last_prediction']
        
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            💰 Montant prédit: {result['montant_predit']:.2f}€
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intervalle Min", f"{result['intervalle_min']:.2f}€")
        with col2:
            st.metric("Estimation Simple", f"{result['estimation_simple']:.2f}€")
        with col3:
            st.metric("Intervalle Max", f"{result['intervalle_max']:.2f}€")
        
        # Graphique de comparaison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Estimation Simple', 'Prédiction IA'],
            y=[result['estimation_simple'], result['montant_predit']],
            marker_color=['lightblue', 'darkblue'],
            text=[f"{result['estimation_simple']:.2f}€", f"{result['montant_predit']:.2f}€"],
            textposition='auto',
        ))
        fig.update_layout(
            title="Comparaison: Estimation Simple vs Prédiction IA",
            yaxis_title="Montant (€)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2 : TABLEAU DE BORD
# ============================================
elif page == "📊 Tableau de bord":
    st.header("📊 Tableau de Bord")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Commandes", f"{len(df):,}")
    with col2:
        st.metric("CA Total", f"{df['montant'].sum():,.2f}€")
    with col3:
        st.metric("Panier Moyen", f"{df['montant'].mean():.2f}€")
    with col4:
        st.metric("Clients Actifs", f"{df['client_id'].nunique():,}")
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Top catégories
        top_cat = df.groupby('categorie')['montant'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_cat.values,
            y=top_cat.index,
            orientation='h',
            title="Top 10 Catégories par CA",
            labels={'x': 'CA (€)', 'y': 'Catégorie'},
            color=top_cat.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top villes
        top_villes = df.groupby('ville')['montant'].sum().sort_values(ascending=False).head(10)
        fig = px.pie(
            values=top_villes.values,
            names=top_villes.index,
            title="Top 10 Villes par CA"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Évolution temporelle
    df_time = df.groupby(df['date_commande'].dt.to_period('M'))['montant'].agg(['sum', 'mean', 'count']).reset_index()
    df_time['date_commande'] = df_time['date_commande'].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_time['date_commande'],
        y=df_time['sum'],
        mode='lines+markers',
        name='CA Total',
        line=dict(color='blue', width=3)
    ))
    fig.update_layout(
        title="Évolution du Chiffre d'Affaires",
        xaxis_title="Mois",
        yaxis_title="CA (€)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 3 : ANALYSES
# ============================================
elif page == "📈 Analyses":
    st.header("📈 Analyses Approfondies")
    
    tab1, tab2, tab3 = st.tabs(["Clients", "Produits", "Tendances"])
    
    with tab1:
        st.subheader("Analyse des Clients")
        
        # Top clients
        top_clients = df.groupby('client_id').agg({
            'montant': ['sum', 'count', 'mean']
        }).reset_index()
        top_clients.columns = ['client_id', 'total', 'nb_commandes', 'panier_moyen']
        top_clients = top_clients.sort_values('total', ascending=False).head(20)
        top_clients = top_clients.merge(client[['client_id', 'nom', 'ville']], on='client_id')
        
        st.dataframe(
            top_clients[['nom', 'ville', 'nb_commandes', 'total', 'panier_moyen']],
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Analyse des Produits")
        
        # Top produits
        top_produits = df.groupby('produit_id').agg({
            'montant': ['sum', 'count']
        }).reset_index()
        top_produits.columns = ['produit_id', 'ca', 'nb_ventes']
        top_produits = top_produits.sort_values('ca', ascending=False).head(20)
        top_produits = top_produits.merge(produit[['produit_id', 'nom_produit', 'categorie']], on='produit_id')
        
        st.dataframe(
            top_produits[['nom_produit', 'categorie', 'nb_ventes', 'ca']],
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Tendances")
        
        # Analyse par jour de la semaine
        df['jour_semaine'] = df['date_commande'].dt.dayofweek
        jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        ventes_jour = df.groupby('jour_semaine')['montant'].agg(['sum', 'count']).reset_index()
        ventes_jour['jour'] = ventes_jour['jour_semaine'].map(lambda x: jours[x])
        
        fig = px.bar(
            ventes_jour,
            x='jour',
            y='sum',
            title="CA par Jour de la Semaine",
            labels={'sum': 'CA (€)', 'jour': 'Jour'},
            color='sum',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 4 : À PROPOS
# ============================================
else:
    st.header("ℹ️ À propos du Modèle")
    
    st.markdown(f"""
    ### 🤖 Informations sur le Modèle
    
    **Algorithme:** {metadata['best_model_name']}  
    **Date d'entraînement:** {metadata['date_training']}  
    **Nombre de features:** {len(metadata['features'])}
    
    ### 📊 Performance
    
    - **R² Score:** {metadata['test_r2']:.4f} ({metadata['test_r2']*100:.2f}% de variance expliquée)
    - **RMSE:** {metadata['test_rmse']:.2f}€
    - **MAE:** {metadata['test_mae']:.2f}€
    
    ### 🎯 Utilisation
    
    Ce modèle prédit le montant d'une commande en fonction de:
    - Historique du client
    - Caractéristiques du produit
    - Contexte temporel (jour, mois, saison)
    - Localisation géographique
    - Tendances de la catégorie
    
    ### 📁 Structure des Données
    
    **Clients:** {client.shape[0]} lignes  
    **Produits:** {produit.shape[0]} lignes  
    **Commandes:** {commande.shape[0]} lignes  
    **Période:** {df['date_commande'].min().date()} au {df['date_commande'].max().date()}
    """)
    
    st.info("💡 **Conseil:** Le modèle est plus précis pour les clients et produits avec un historique important.")

# Footer
 #using Streamlit
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Made by Daina KAMTA | Modèle de Machine Learning | © 2025
</div>
""", unsafe_allow_html=True)
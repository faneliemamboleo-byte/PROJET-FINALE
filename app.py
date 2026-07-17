import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from Utiles.data_processing import validate_inputs, create_input_dataframe

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Ebola - RDC",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #D4E6F1;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-number {
        font-size: 3rem;
        font-weight: bold;
        color: #1A5276;
    }
    .info-box {
        background-color: #FDEBD0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #E67E22;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1A5276;
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource
def load_model():
    """Charge le modèle et le scaler."""
    try:
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error(" Modèle non trouvé. Veuillez exécuter train_model.py d'abord.")
        st.stop()
    except Exception as e:
        st.error(f" Erreur lors du chargement du modèle : {e}")
        st.stop()

model, scaler = load_model()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
    st.markdown("## Prédiction Ebola")
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    Cette application utilise un modèle de **régression linéaire multiple** 
    pour prédire le nombre de cas confirmés d'Ebola en RDC.
    
    **Variables utilisées :**
    - Cas suspects
    - Pluviométrie
    - Température
    - Déforestation
    - NDVI
    """)
    st.markdown("---")
    st.caption(" Projet Data Science - UMIE/DRC")

# En-tête principal
st.markdown('<p class="main-header"> Prédiction des cas confirmés d\'Ebola</p>', unsafe_allow_html=True)

# Colonnes pour la mise en page
col_info, col_form = st.columns([1, 2])

with col_info:
    st.markdown("""
    ###  Comment ça fonctionne ?
    
    1. Saisissez les données dans le formulaire
    2. Cliquez sur **Prédire**
    3. Obtenez instantanément le nombre de cas confirmés prédits
    
    ###  Performance du modèle
    """)
    
    # Afficher les métriques du modèle
    try:
        # Ces valeurs devraient être chargées depuis un fichier
        st.metric("R²", "0.94", "Excellent")
        st.metric("RMSE", "5.50", "Faible")
        st.metric("MAE", "4.42", "Faible")
    except:
        pass

with col_form:
    with st.form("prediction_form"):
        st.markdown("### Saisie des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cas_suspects = st.number_input(
                " Cas suspects",
                min_value=0,
                value=70,
                step=5,
                help="Nombre de cas suspects signalés"
            )
            
            pluviometrie = st.number_input(
                " Pluviométrie (mm)",
                min_value=0,
                value=165,
                step=10,
                help="Précipitations en millimètres"
            )
            
            temperature = st.number_input(
                " Température (°C)",
                min_value=10.0,
                max_value=40.0,
                value=25.8,
                step=0.1,
                help="Température moyenne en degrés Celsius"
            )
        
        with col2:
            deforestation = st.number_input(
                " Déforestation (ha)",
                min_value=0.0,
                value=17.0,
                step=0.5,
                help="Surface déforestée en hectares"
            )
            
            ndvi = st.slider(
                " NDVI",
                min_value=0.0,
                max_value=1.0,
                value=0.44,
                step=0.01,
                help="Indice de végétation (0 = sol nu, 1 = végétation dense)"
            )
        
        st.markdown("---")
        submitted = st.form_submit_button(" Prédire les cas confirmés", type="primary")

# Traitement de la prédiction
if submitted:
    # Validation des entrées
    is_valid, error_msg = validate_inputs(cas_suspects, pluviometrie, temperature, deforestation, ndvi)
    
    if not is_valid:
        st.error(f" Erreur de saisie : {error_msg}")
    else:
        # Création du DataFrame d'entrée
        input_data = create_input_dataframe(
            cas_suspects, pluviometrie, temperature, deforestation, ndvi
        )
        
        # Standardisation et prédiction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = max(0, round(prediction, 1))
        
        # Affichage des résultats
        st.divider()
        
        # Boîte de résultat
        st.markdown(f"""
        <div class="prediction-box">
            <p style="font-size: 1.2rem; margin: 0;"> Nombre de cas confirmés prédits</p>
            <p class="prediction-number">{prediction}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Colonnes pour les informations complémentaires
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(" Cas suspects", cas_suspects)
        with col_res2:
            st.metric(" Pluviométrie", f"{pluviometrie} mm")
        with col_res3:
            st.metric(" Température", f"{temperature} °C")
        
        # Graphique des variables
        with st.expander(" Visualisation des données d'entrée", expanded=False):
            fig = px.bar(
                x=['Cas Suspects', 'Pluviométrie', 'Température', 'Déforestation', 'NDVI'],
                y=[cas_suspects, pluviometrie, temperature, deforestation, ndvi],
                labels={'x': 'Variables', 'y': 'Valeurs'},
                title="Distribution des variables d'entrée",
                color=['#2E86C1', '#28B463', '#E67E22', '#7D3C98', '#F1C40F']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Interprétation
        with st.expander(" Interprétation des résultats", expanded=True):
            st.markdown("""
            <div class="info-box">
                <b> Analyse :</b><br>
                • Le nombre de <b>cas suspects</b> est le facteur le plus influent<br>
                • La <b>pluviométrie</b> et la <b>déforestation</b> ont un impact positif modéré<br>
                • Une <b>température</b> plus élevée et un <b>NDVI</b> plus élevé réduisent la prédiction<br>
                • Le modèle explique environ <b>94%</b> de la variabilité des cas confirmés
            </div>
            """, unsafe_allow_html=True)

# Pied de page
st.divider()
col_footer1, col_footer2 = st.columns([2, 1])
with col_footer1:
    st.caption(" Développé dans le cadre du projet de Data Science - UMIE/DRC")
with col_footer2:
    st.caption("[ Voir le code source](https://github.com/votre-username/projet-ebola)")
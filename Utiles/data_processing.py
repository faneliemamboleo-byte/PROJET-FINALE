import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filepath):
    """
    Chargement et préparation des données pour l'entraînement.
    
    Args:
        filepath (str): Chemin vers le fichier CSV
        
    Returns:
        tuple: (X, y, feature_names)
    """
    df = pd.read_csv(filepath)
    
    # Sélection des features et de la cible
    feature_columns = ['Cas_Suspects', 'Pluviometrie_mm', 'Temperature_C', 
                      'Deforestation_ha', 'NDVI']
    target_column = 'Cas_Confirmes'
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y, feature_columns

def create_input_dataframe(cas_suspects, pluviometrie, temperature, deforestation, ndvi):
    """
    Création du DataFrame à partir des entrées utilisateur.
    
    Args:
        cas_suspects (int): Nombre de cas suspects
        pluviometrie (float): Pluviométrie en mm
        temperature (float): Température en °C
        deforestation (float): Déforestation en ha
        ndvi (float): Indice NDVI
        
    Returns:
        pd.DataFrame: DataFrame formaté pour la prédiction
    """
    return pd.DataFrame({
        'Cas_Suspects': [cas_suspects],
        'Pluviometrie_mm': [pluviometrie],
        'Temperature_C': [temperature],
        'Deforestation_ha': [deforestation],
        'NDVI': [ndvi]
    })

def validate_inputs(cas_suspects, pluviometrie, temperature, deforestation, ndvi):
    """
    Valide les entrées utilisateur.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if cas_suspects < 0:
        return False, "Les cas suspects ne peuvent pas être négatifs"
    if pluviometrie < 0:
        return False, "La pluviométrie ne peut pas être négative"
    if temperature < -20 or temperature > 50:
        return False, "La température doit être entre -20°C et 50°C"
    if deforestation < 0:
        return False, "La déforestation ne peut pas être négative"
    if ndvi < 0 or ndvi > 1:
        return False, "Le NDVI doit être entre 0 et 1"
    return True, "OK"
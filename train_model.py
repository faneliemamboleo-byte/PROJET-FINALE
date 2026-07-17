import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from Utiles.data_processing import load_and_prepare_data

def train_model():
    """
    Entraîne le modèle de régression linéaire multiple.
    """
    print(" Début de l'entraînement du modèle...")
    
    # 1. Chargement des données
    X, y, feature_names = load_and_prepare_data('data/donnees_epidemiologiques_RDC.csv')
    
    print(f" Données chargées : {X.shape[0]} observations, {X.shape[1]} variables")
    
    # 2. Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f" Train set: {X_train.shape[0]} observations")
    print(f" Test set: {X_test.shape[0]} observations")
    
    # 3. Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Entraînement
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 5. Évaluation
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n Performance du modèle :")
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")
    
    # 6. Affichage des coefficients
    print("\n Coefficients du modèle :")
    for name, coef in zip(feature_names, model.coef_):
        print(f"   {name}: {coef:.4f}")
    print(f"   Intercept: {model.intercept_:.4f}")
    
    # 7. Sauvegarde
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n Modèle sauvegardé dans 'models/model.pkl'")
    print(" Scaler sauvegardé dans 'models/scaler.pkl'")
    
    return model, scaler

if __name__ == "__main__":
    train_model()
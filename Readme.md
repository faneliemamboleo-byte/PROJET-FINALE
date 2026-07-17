## Installation

### Prérequis
- Python 3.8 ou supérieur
- Git

### Étapes d'installation
```bash
# 1. Cloner le dépôt
git clone https://github.com/Onesime243/projet-Epi_INHOHA.git 
cd Epi_regression

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Entraîner le modèle
python train_model.py

# 5. Lancer l'application
streamlit run app.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Agora definimos os caminhos baseados na Raiz, não importa onde você esteja no terminal
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

def train():
    print(f"--- Iniciando Pipeline no Projeto: {PROJECT_ROOT.name} ---")
    
    # 1. Carregar dados
    file_path = PROCESSED_DATA_PATH / "train.csv"
    print(f"[1/4] Carregando dados de: {file_path}")
    
    if not file_path.exists():
        print(f"ERRO CRÍTICO: Arquivo não encontrado em: {file_path}")
        print("Verifique se você rodou o notebook '02_feature_engineering.ipynb' e se a pasta data/processed contém os arquivos.")
        return

    train_df = pd.read_csv(file_path)

    X = train_df.drop('Class', axis=1)
    y = train_df['Class']

    # 2. Aplicar SMOTE
    print(f"[2/4] Aplicando SMOTE no dataset de treino...")
    print(f"   - Antes: {y.value_counts().to_dict()}")
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    print(f"   - Depois: {y_res.value_counts().to_dict()}")

    # 3. Treinar Modelo
    print("[3/4] Treinando Random Forest (pode demorar)...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_res, y_res)

    # 4. Salvar Modelo
    MODELS_PATH.mkdir(parents=True, exist_ok=True) # Cria a pasta models se não existir
    model_path = MODELS_PATH / "random_forest_v1.pkl"
    joblib.dump(model, model_path)
    
    print(f"[4/4] Sucesso! Modelo salvo em: {model_path}")
    print("--- Fim do Pipeline ---")

if __name__ == "__main__":
    train()
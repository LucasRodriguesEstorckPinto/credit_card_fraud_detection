import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os

# Configurações de Caminhos (Paths absolutos ou relativos ao script)

PROCESSED_DATA_PATH = "data/processed"
MODELS_PATH = "models"

def train():
    print("--- Iniciando Pipeline de Treinamento ---")
    
    # 1. Carregar dados
    print(f"[1/4] Carregando dados de: {PROCESSED_DATA_PATH}")
    try:
        train_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/train.csv")
    except FileNotFoundError:
        print("ERRO: Arquivo train.csv não encontrado. Rode o notebook de engenharia antes.")
        return

    X = train_df.drop('Class', axis=1)
    y = train_df['Class']

    # 2. Aplicar SMOTE
    print(f"[2/4] Aplicando SMOTE no dataset de treino...")
    print(f"   - Antes: {y.value_counts().to_dict()}")
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    print(f"   - Depois: {y_res.value_counts().to_dict()}")

    # 3. Treinar Modelo
    print("[3/4] Treinando Random Forest (pode demorar um pouco)...")
    # Usando os parâmetros que deram certo no seu teste
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_res, y_res)

    # 4. Salvar Modelo
    os.makedirs(MODELS_PATH, exist_ok=True)
    model_path = f"{MODELS_PATH}/random_forest_v1.pkl"
    joblib.dump(model, model_path)
    
    print(f"[4/4] Sucesso! Modelo salvo em: {model_path}")
    print("--- Fim do Pipeline ---")

if __name__ == "__main__":
    train()
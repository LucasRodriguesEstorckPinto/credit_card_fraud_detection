
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_v1.pkl"

# 1. Inicializar a API
app = FastAPI(
    title="API de Detecção de Fraude (HFT - Lucas)",
    description="Microserviço para analisar transações financeiras em tempo real.",
    version="1.0.0"
)

# 2. Carregar o Modelo na Memória
print(f"Carregando modelo de: {MODEL_PATH}")
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso!")
else:
    print("ERRO CRÍTICO: Modelo não encontrado.")
    model = None

# 3. Schema de Entrada
class Transaction(BaseModel):
    scaled_amount: float
    scaled_time: float
    features_v: list[float] 

# 4. Rota de Predição
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não está carregado no servidor.")
    
    # Validação simples
    if len(transaction.features_v) != 28:
        raise HTTPException(status_code=400, detail="A lista features_v deve conter exatos 28 valores anonimizados.")

    # Montar vetor de entrada (exatamente como no treino)
    input_data = [transaction.scaled_amount, transaction.scaled_time] + transaction.features_v
    
    # Scikit-learn espera um array 2D [[...]]
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1] # Probabilidade da classe 1 (Fraude)

    return {
        "prediction": int(prediction),
        "status": "FRAUDE" if prediction == 1 else "APROVADO",
        "fraud_probability": float(round(probability, 4)),
        "risk_level": "CRÍTICO" if probability > 0.8 else ("ALTO" if probability > 0.5 else "BAIXO")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
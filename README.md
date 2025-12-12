# Credit Card Fraud Detection System 🛡️💳

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange)
![License](https://img.shields.io/badge/License-MIT-grey)

## 📌 Sobre o Projeto

Este projeto é um sistema *end-to-end* de Detecção de Fraudes em Cartões de Crédito. O objetivo foi resolver o problema clássico de **dados desbalanceados** (apenas 0.17% de fraudes) em um cenário financeiro real.

Diferente de notebooks acadêmicos comuns, este projeto foi estruturado com foco em **Engenharia de Machine Learning**, incluindo pipelines de treinamento reprodutíveis e deploy via API REST.

## 🚀 Arquitetura e Engenharia

O projeto segue a estrutura padrão de Data Science (Cookiecutter) e consiste em 4 etapas principais:

1.  **Exploratory Data Analysis (EDA):** Identificação de *outliers* e padrões temporais.
2.  **Feature Engineering:**
    * Aplicação de `RobustScaler` para mitigar outliers extremos em valores monetários.
    * Split estratégico de dados para evitar *Data Leakage*.
3.  **Modelagem e Balanceamento:**
    * Uso de **SMOTE** (Synthetic Minority Over-sampling Technique) apenas no conjunto de treino.
    * Evolução de *Logistic Regression* para **Random Forest**, reduzindo Falsos Positivos em 99%.
4.  **Deploy (Produção):**
    * API desenvolvida em **FastAPI** para inferência em tempo real.
    * Validação de dados com **Pydantic**.

## 📊 Resultados Técnicos

O modelo final (Random Forest) alcançou performance superior para o negócio, priorizando a redução de bloqueios indevidos (Falsos Positivos) sem perder a capacidade de detectar fraudes.

| Métrica | Performance (Test Set) |
| :--- | :--- |
| **Recall (Fraude)** | **0.82** (Detecta 82% das fraudes) |
| **Precision** | **0.85** (Alta confiabilidade nos alertas) |
| **AUPRC** | **0.87** (Área sob a curva Precision-Recall) |
| **Latência API** | ~50ms por requisição |

## 🛠️ Instalação e Uso

### 1. Clone o repositório
```bash
git clone [https://github.com/SEU_USUARIO/fraud-detection-finance.git](https://github.com/SEU_USUARIO/fraud-detection-finance.git)
cd fraud-detection-finance
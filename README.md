# Sistema de Previsão de Churn de Clientes

Este projeto é uma solução de machine learning de ponta a ponta para prever o churn (rotatividade) de clientes em uma empresa de telecomunicações. Ele contém um pipeline completo que inclui exploração de dados, pré-processamento e treinamento de modelos.

Melhorias essenciais são: rastreamento de experimentos com MLflow, disponibilização do modelo via uma API de inferência FastAPI e uma interface gráfica via Streamlit.

---

## Estrutura do Projeto

```bash
MLOps-training/
│
├── data/                   # Arquivos de dados brutos e processados
│   ├── processed/
│   └── raw/
├── models/                 # Artefatos salvos: best_model.pkl, scaler.pkl, columns.pkl
├── notebooks/              # Notebooks Jupyter para EDA
├── outputs/                # Plots gerados (ROC, matrizes de confusão)
├── src/                    # Módulos Python principais
│   ├── __init__.py
│   ├── config.py           # Caminhos + configuração do MLflow
│   ├── data_preprocessing.py
│   ├── model.py            # Treino com tracking MLflow + seleção do melhor modelo
│   ├── visualization.py
│   ├── api.py              # API de inferência FastAPI
│   └── app.py              # Interface Streamlit
├── tests/                  # Testes pytest (preprocess, api, model)
├── main.py                 # Pipeline end-to-end de treino
├── Dockerfile.api          # Imagem da API
├── Dockerfile.ui           # Imagem da UI
├── docker-compose.yml      # Stack completa (mlflow + api + ui)
├── Makefile                # Atalhos: train, serve, ui, test, up
└── README.md
```

---

## Componentes & Visão Geral do Pipeline

1. **Exploração de Dados (notebooks/)**
   - Entender distribuições de features, correlações e valores ausentes.
   - Visualizações comparando clientes que churnaram e os que não churnaram.

2. **Pré-processamento (src/data_preprocessing.py)**
   - Tratamento de valores ausentes
   - Codificação de variáveis categóricas
   - Escalonamento/normalização de features

3. **Treinamento de Modelos (src/model.py + notebook)**
   - Modelos treinados:
     - Regressão Logística
     - Random Forest
     - XGBoost
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Multi-layer Perceptron (MLP)

4. **Avaliação (src/visualization.py + notebook)**
   - Acurácia, F1 score, ROC AUC
   - Plots: Matriz de Confusão, curvas ROC, etc.

---

## Ferramentas & Bibliotecas Utilizadas

- **Python 3.12**
- **Pandas**, **NumPy**, **scikit-learn**: processamento de dados e modelagem
- **XGBoost**: modelo de boosting
- **MLflow**: rastreamento de experimentos e registro de modelos
- **FastAPI**: API de inferência
- **Streamlit**: frontend
- **Docker**: conteinerização para desenvolvimento e deploy
- **matplotlib**, **seaborn**: visualizações

---

## Como Rodar

> Os comandos usam [`uv`](https://docs.astral.sh/uv/). Há atalhos equivalentes no `Makefile` (ex.: `make train`).

### 1. Configurar o Ambiente

```bash
uv sync                    # instala dependências (runtime + dev) a partir do uv.lock
```

### 2. Treinar os Modelos (MLflow)

```bash
uv run python main.py                       # treina 6 modelos, loga no MLflow, salva o melhor
uv run python main.py --no-plot             # sem gerar o plot de ROC
uv run python main.py --experiment my-exp   # nome de experimento customizado
```

O melhor modelo (por **ROC AUC**, métrica escolhida por o dataset ser desbalanceado)
é salvo em `models/best_model.pkl`, junto de `scaler.pkl` e `columns.pkl`.

### 3. Visualizar Experimentos no MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# abre em http://localhost:5000
```

### 4. Servir a API de Inferência (FastAPI)

```bash
uv run uvicorn src.api:app --reload --port 8000
# docs interativas: http://localhost:8000/docs
```

Exemplo de requisição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,
       "PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL",
       "OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":29.85,"TotalCharges":29.85}'
# -> {"prediction":1,"probability":0.61,"label":"Churn"}
```

Endpoints: `POST /predict`, `GET /health`, `GET /` (redireciona para `/docs`).

### 5. Interface Gráfica (Streamlit)

```bash
uv run streamlit run src/app.py
# http://localhost:8501  (configure a API_URL na sidebar se necessário)
```

### 6. Stack Completa com Docker

```bash
docker compose up --build
# mlflow  -> http://localhost:5000
# api     -> http://localhost:8000/docs
# ui      -> http://localhost:8501
```

### 7. Testes

```bash
uv run pytest        # suíte completa
uv run ruff check .  # lint
```

---

## Notas Técnicas

- **Data leakage (conhecido):** `preprocess_data` faz `scaler.fit_transform` sobre o
  dataset inteiro antes do split treino/teste. É aceitável para esta entrega
  educacional, mas em produção o scaler deveria ser ajustado **apenas no treino**
  (idealmente dentro de um `Pipeline` do scikit-learn). _TODO._
- **Alinhamento de features:** a API usa `reindex(columns=columns.pkl, fill_value=0)`
  para garantir que o one-hot encoding da requisição bata exatamente com as colunas
  vistas no treino.
- **Tracking do MLflow:** por padrão usa SQLite (`mlflow.db`); o file store foi
  descontinuado no MLflow 3.x. Em Docker, sobrescrito por `MLFLOW_TRACKING_URI`.

### Trabalhos Futuros

- Adicionar **testes** unitários e de integração
- Automatizar todo o pipeline com **CI/CD**
- Adicionar **monitoramento** e logging em tempo real
- Incorporar **re-treinamento automático** de modelos

## Licença

Este projeto é para **fins educacionais**.

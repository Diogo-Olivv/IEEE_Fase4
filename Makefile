.PHONY: help install train mlflow serve ui demo up down test lint

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências (uv sync)
	uv sync

train:  ## Treina os 6 modelos + MLflow + salva o melhor
	uv run python main.py

mlflow:  ## Sobe só a MLflow UI (http://localhost:5000)
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:  ## Sobe só a API FastAPI (http://localhost:8000/docs)
	uv run uvicorn src.api:app --reload --port 8000

ui:  ## Sobe só a interface Streamlit (http://localhost:8501)
	uv run streamlit run src/app.py --server.port 8501

demo:  ## Sobe MLflow + API + UI juntos (script run_all.sh)
	./run_all.sh

up:  ## Sobe a stack completa via Docker Compose
	docker compose up --build

down:  ## Derruba a stack Docker
	docker compose down

test:  ## Roda a suíte de testes
	uv run pytest -q

lint:  ## Lint com ruff
	uv run ruff check .

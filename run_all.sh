#!/usr/bin/env bash
#
# Sobe os tres servicos visualizaveis do projeto de uma vez:
#   MLflow UI      http://localhost:5000
#   API (FastAPI)  http://localhost:8000/docs
#   UI (Streamlit) http://localhost:8501
#
# Uso:
#   ./run_all.sh                sobe os 3 servicos (treina antes se faltar modelo)
#   ./run_all.sh --train        forca o re-treino antes de subir
#   ./run_all.sh --no-browser   nao abre o navegador automaticamente
#
# Ctrl+C derruba os tres servicos.

set -euo pipefail

export PATH="/home/diogo/snap/code/241/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERRO: 'uv' nao encontrado no PATH." >&2
  exit 1
fi

cd "$(dirname "$0")"

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

FORCE_TRAIN=0
OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --train)      FORCE_TRAIN=1 ;;
    --no-browser) OPEN_BROWSER=0 ;;
    -h|--help)    sed -n '3,14p' "$0"; exit 0 ;;
    *) echo "Argumento desconhecido: $arg" >&2; exit 1 ;;
  esac
done

echo "Sincronizando dependencias (uv sync)..."
uv sync --quiet

if [[ "$FORCE_TRAIN" -eq 1 || ! -f models/best_model.pkl ]]; then
  echo "Treinando modelos..."
  uv run python main.py
fi

PIDS=()

cleanup() {
  echo ""
  echo "Encerrando servicos..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

start() {
  local name="$1"; local log="$2"; shift 2
  echo "Iniciando $name (log: $log)"
  "$@" >"$log" 2>&1 &
  PIDS+=("$!")
}

start "MLflow UI" "$LOG_DIR/mlflow.log" \
  uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5000

start "API (FastAPI)" "$LOG_DIR/api.log" \
  uv run uvicorn src.api:app --port 8000

start "UI (Streamlit)" "$LOG_DIR/streamlit.log" \
  uv run streamlit run src/app.py --server.port 8501 --server.headless true

wait_port() {
  local port="$1"; local name="$2"
  for _ in $(seq 1 60); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  echo "AVISO: $name nao respondeu na porta $port (veja o log)." >&2
}

echo ""
echo "Aguardando servicos ficarem prontos..."
wait_port 5000 "MLflow UI"
wait_port 8000 "API"
wait_port 8501 "Streamlit"

echo ""
echo "Tudo no ar:"
echo "  MLflow (experimentos)   http://localhost:5000"
echo "  API + Swagger docs      http://localhost:8000/docs"
echo "  Interface (Streamlit)   http://localhost:8501"
echo "  Plot de ROC             outputs/roc_curves.png"
echo ""
echo "Pressione Ctrl+C para encerrar todos os servicos."
echo ""

if [[ "$OPEN_BROWSER" -eq 1 ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:5000"      >/dev/null 2>&1 || true
  xdg-open "http://localhost:8000/docs" >/dev/null 2>&1 || true
  xdg-open "http://localhost:8501"      >/dev/null 2>&1 || true
fi

wait

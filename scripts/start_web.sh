# Start the FastAPI backend + Vite dev server (for local web UI use)
# Usage: bash scripts/start_web.sh [conda-env]

ENV="${1:-final}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Project: $PROJECT_DIR"

# Activate conda environment 
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV" 2>/dev/null || true

cd "$PROJECT_DIR"

# Kill any previous instances
echo "Cleaning up old processes..."
pkill -f "uvicorn server:app" 2>/dev/null || true
pkill -f "vite"               2>/dev/null || true
sleep 2

# ── Start FastAPI backend on port 8080
echo "Starting FastAPI backend on http://localhost:8080 ..."
uvicorn server:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait until backend is ready before starting Vite
echo "Waiting for backend to be ready..."
until curl -s http://localhost:8080/docs > /dev/null 2>&1; do
  sleep 1
done
echo "Backend is ready!"

# ── Start Vite dev server on port 5173
echo "Starting Vite frontend on http://localhost:5173 ..."
cd "$PROJECT_DIR/client"
npm run dev -- --host 0.0.0.0 --port 5173
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "==================================================="
echo "  Backend  → http://localhost:8080"
echo "  Frontend → http://localhost:5173  (open this!)"
echo "==================================================="
echo "Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait

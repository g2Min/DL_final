#!/usr/bin/env bash
set -euo pipefail

ENV="${1:-final}"
GPU_LIST="${2:-0,1,2,3}"

shift 2 2>/dev/null || true
OTHER_CMD="$*"

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
GPU_UTIL=0.4
MAX_LEN=4096

PROJECT_DIR="/workspace/DL_final"
VLLM_LOG="$PROJECT_DIR/vllm_server.log"
OTHER_LOG="$PROJECT_DIR/other_model.log"

# GPU 문자열 정리
GPU_LIST="${GPU_LIST// /}"
IFS=',' read -ra GPU_ARR <<< "$GPU_LIST"

if [ "${#GPU_ARR[@]}" -lt 1 ] || [ -z "${GPU_ARR[0]}" ]; then
    echo "GPU_LIST가 비어 있습니다."
    exit 1
fi

VLLM_GPU="${GPU_ARR[0]}"

OTHERS_JOINED=""
if [ "${#GPU_ARR[@]}" -gt 1 ]; then
    OTHERS_JOINED=$(IFS=,; echo "${GPU_ARR[*]:1}")
fi

echo "Activating conda environment: $ENV"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

cd "$PROJECT_DIR"

# 기존 8000 포트 서버가 있으면 중복 실행 방지
if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM server is already running on port $PORT"
    VLLM_PID=""
else
    echo "Starting vLLM serve ($MODEL)"
    echo "GPU: $VLLM_GPU"
    echo "Log: $VLLM_LOG"

    CUDA_VISIBLE_DEVICES="$VLLM_GPU" nohup vllm serve "$MODEL" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_LEN" \
        > "$VLLM_LOG" 2>&1 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    READY=0

    echo "Waiting for vLLM server to be ready..."

    # 최초 모델 다운로드까지 고려해 최대 10분 대기
    for i in $(seq 1 120); do
        if curl -fsS \
            "http://127.0.0.1:$PORT/v1/models" \
            >/dev/null 2>&1; then

            READY=1
            echo "vLLM server is ready."
            break
        fi

        if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
            echo "vLLM process exited unexpectedly."
            echo "===== vLLM log ====="
            tail -n 200 "$VLLM_LOG"
            exit 1
        fi

        if (( i % 10 == 0 )); then
            echo "Still waiting... (${i}/120)"
            tail -n 5 "$VLLM_LOG" || true
        fi

        sleep 5
    done

    if [ "$READY" -ne 1 ]; then
        echo "vLLM server did not become ready."
        echo "===== vLLM log ====="
        tail -n 200 "$VLLM_LOG"

        if [ -n "$VLLM_PID" ]; then
            kill "$VLLM_PID" 2>/dev/null || true
        fi

        exit 1
    fi
fi

if [ -n "$OTHER_CMD" ]; then
    if [ -z "$OTHERS_JOINED" ]; then
        echo "다른 명령에 할당할 GPU가 없습니다."
    else
        echo "Starting other command on GPUs: $OTHERS_JOINED"

        CUDA_VISIBLE_DEVICES="$OTHERS_JOINED" \
            nohup bash -lc "$OTHER_CMD" \
            > "$OTHER_LOG" 2>&1 &

        OTHER_PID=$!
        echo "Other command PID: $OTHER_PID"
    fi
else
    echo "OTHER_CMD가 비어 있습니다. 다른 모델은 실행하지 않습니다."
fi

# 파이프라인 실행 직전 한 번 더 점검
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null

echo "vLLM log: $VLLM_LOG"
echo "파이프라인은 웹 클라이언트 요청 시 server.py가 실행합니다."
echo "  → bash scripts/start_web.sh 로 웹 서버를 시작하세요."
#!/bin/bash
set -a
source ./.env
set +a

unset SSL_CERT_FILE  # 이 줄 추가
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ashraq/fashion-product-images-small",
    repo_type="dataset",
    local_dir="./datasets",
    token=os.environ["HF_TOKEN"],
    max_workers=1,
)
PY
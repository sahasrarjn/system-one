#!/usr/bin/env bash
# Runs ON the GPU instance. Everything from raw corpora to the reliability
# diagram. Ordered so the cheap things that can fail, fail early.
set -euo pipefail
PY=/opt/so/bin/python
export PYTHONPATH=.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

echo "=============== 0. environment"
$PY - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "no GPU visible"
EOF

echo "=============== 1. corpora"
mkdir -p artifacts/data
if [ ! -f artifacts/data/complaints.csv ]; then
  echo "-- CFPB complaint database (~350MB zipped)"
  curl -sSL --retry 3 -o /tmp/ccdb.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip
  unzip -o -q /tmp/ccdb.zip -d artifacts/data/
  rm -f /tmp/ccdb.zip
  ls -la artifacts/data/complaints.csv
fi

echo "=============== 2. build dataset"
$PY scripts/build_data.py \
  --cfpb-csv artifacts/data/complaints.csv \
  --per-product 3000 --civil-n 40000 \
  --out-dir artifacts/data

echo "=============== 3. smoke (30 steps, throws away weights)"
$PY scripts/train.py --data-dir artifacts/data --out artifacts/runs/smoke \
  --limit-steps 30 --eval-every 0 --batch-size 8 --grad-accum 2 --no-save

echo "=============== 4. full train"
$PY scripts/train.py --data-dir artifacts/data --out artifacts/runs/run1 \
  --epochs 2 --batch-size 8 --grad-accum 2 --lr 2e-5 \
  --max-state-tokens 512 --eval-every 400 --eval-batches 150 \
  2>&1 | tee train.log

echo "=============== 5. evaluate"
$PY scripts/evaluate.py --preds artifacts/runs/run1/val_preds.npz | tee -a train.log

echo "=============== done"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

#!/usr/bin/env bash
# All three queued experiments in one instance session.
#
# Ordered to bank cheap results before the expensive one. Neither pipeline has
# ever run against real weights, so every stage is preceded by a smoke test
# that fails in minutes rather than hours.
set -uo pipefail          # NOT -e: one failing experiment must not kill the rest
PY=/opt/so/bin/python
export PYTHONPATH=.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p artifacts/runs

stage() { echo; echo "================= $* ================="; date -u +"  %H:%M:%SZ"; }
ok=(); bad=()
run() { name="$1"; shift; if "$@"; then ok+=("$name"); else bad+=("$name"); echo "!! FAILED: $name"; fi; }

stage 0 environment
$PY - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
assert torch.cuda.is_available(), "no GPU visible"
print("gpu:", torch.cuda.get_device_name(0),
      "| bf16:", torch.cuda.is_bf16_supported(),
      "| vram: %.0f GiB" % (torch.cuda.get_device_properties(0).total_memory/2**30))
EOF
[ $? -ne 0 ] && { echo "!! no GPU, aborting"; exit 1; }

# ---------------------------------------------------------------- vision first
# Cheapest, and the code has literally never executed against real weights.
stage 1 "vision smoke (8 examples)"
run vision-smoke $PY scripts/vision_run.py cache --split trainval --limit 8

stage 2 "vision: encode-once vs re-encode latency"
run vision-bench $PY scripts/vision_bench.py 40

stage 3 "vision: full feature cache"
run vision-cache-train $PY scripts/vision_run.py cache --split trainval --limit 2500
run vision-cache-test  $PY scripts/vision_run.py cache --split test     --limit 1500

stage 4 "vision: train probe + compare against bi-encoder"
run vision-train $PY scripts/vision_run.py train

# ------------------------------------------------------------------ then text
stage 5 "text: corpora"
if [ ! -f artifacts/data/complaints.csv ]; then
  curl -sSL --retry 3 -o /tmp/ccdb.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip \
    && unzip -o -q /tmp/ccdb.zip -d artifacts/data/ && rm -f /tmp/ccdb.zip
fi
ls -la artifacts/data/complaints.csv 2>/dev/null || echo "!! no CFPB csv"

stage 6 "text: build dataset"
run text-data $PY scripts/build_data.py --cfpb-csv artifacts/data/complaints.csv \
  --per-product 3000 --civil-n 40000 --out-dir artifacts/data

stage 7 "text: smoke (30 steps)"
run text-smoke $PY scripts/train.py --data-dir artifacts/data --out artifacts/runs/smoke \
  --limit-steps 30 --eval-every 0 --batch-size 8 --grad-accum 2 --no-save

stage 8 "text: full train"
run text-train $PY scripts/train.py --data-dir artifacts/data --out artifacts/runs/run1 \
  --epochs 2 --batch-size 8 --grad-accum 2 --lr 2e-5 --max-state-tokens 512 \
  --eval-every 400 --eval-batches 150

stage 9 "text: reliability"
run text-eval $PY scripts/evaluate.py --preds artifacts/runs/run1/val_preds.npz

stage 10 summary
echo "  succeeded: ${ok[*]:-none}"
echo "  failed:    ${bad[*]:-none}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

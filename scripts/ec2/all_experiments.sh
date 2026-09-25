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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p artifacts/runs

stage() { echo; echo "================= $* ================="; date -u +"  %H:%M:%SZ"; }
ok=(); bad=()
run() { name="$1"; shift; if "$@"; then ok+=("$name"); else bad+=("$name"); echo "!! FAILED: $name"; fi; }

stage 0 environment
$PY - <<'EOF'
# Import everything the run needs up front. A missing package should fail here
# in two seconds, not three stages and one model download later.
import torch, transformers, torchvision, numpy, pandas, datasets, PIL
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| torchvision", torchvision.__version__)
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
stage 5 "text: build dataset (CLINC150 + Civil Comments)"
# CFPB is gone: as of Sep 2026 it publishes complaint metadata but no longer
# the narratives, and the narrative was the document. CLINC150 replaces it and
# is a better fit anyway: 150 intents rather than ten skewed products, and a
# native out-of-scope class that gives the text side an abstention test.
# Run 4 was 72% Civil Comments, a 2-option task with a skewed prior, which
# meant a single accuracy figure could be carried almost entirely by the easy
# half. Roughly equal now. Civil is still the calibration anchor, since its
# targets are fractions of human annotators rather than one-hot labels.
run text-data $PY scripts/build_data.py --clinc-n 20000 --civil-n 16000 \
  --out-dir artifacts/data

stage 6 "text: smoke (30 steps)"
run text-smoke $PY scripts/train.py --data-dir artifacts/data --out artifacts/runs/smoke \
  --limit-steps 30 --eval-every 0 --max-tokens 2560 --grad-accum 4 \
  --grad-checkpointing --no-save

stage 7 "text: full train"
# Batches are sized by TOKENS, not by example count. Lengths span 49 tokens at
# the median to 645 at the max, and a fixed batch size tuned on the average
# dies on the tail: batch 8 trained twenty steps and then met a batch holding
# several 151-option questions. A 2560-token budget peaks at roughly 47% of
# the memory that failed, and lets short questions travel 40 at a time.
run text-train $PY scripts/train.py --data-dir artifacts/data --out artifacts/runs/run1 \
  --epochs 2 --max-tokens 2560 --grad-accum 4 --lr 3e-5 --max-state-tokens 256 \
  --grad-checkpointing --eval-every 150 --eval-batches 60

stage 8 "text: reliability"
run text-eval $PY scripts/evaluate.py --preds artifacts/runs/run1/val_preds.npz

stage 9 summary
echo "  succeeded: ${ok[*]:-none}"
echo "  failed:    ${bad[*]:-none}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

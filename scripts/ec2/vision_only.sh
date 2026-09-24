#!/usr/bin/env bash
# The vision arm alone. Run 4 got the feature cache built and then died on a
# NameError in the comparison, so everything expensive already works; this is
# the cheap half repeated with the fix in place.
set -uo pipefail
PY=/opt/so/bin/python
export PYTHONPATH=.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p artifacts/vision

stage() { echo; echo "================= $* ================="; date -u +"  %H:%M:%SZ"; }
ok=(); bad=()
run() { name="$1"; shift; if "$@"; then ok+=("$name"); else bad+=("$name"); echo "!! FAILED: $name"; fi; }

stage 0 environment
$PY - <<'EOF'
import torch, transformers, torchvision, numpy, PIL
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
assert torch.cuda.is_available(), "no GPU visible"
print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
EOF
[ $? -ne 0 ] && { echo "!! no GPU, aborting"; exit 1; }

stage 1 "latency: encode-once vs re-encode"
# measured 1.9x and 1.8x on two prior instances; a third reading settles it
run bench $PY scripts/vision_bench.py 40

stage 2 "feature cache"
run cache-train $PY scripts/vision_run.py cache --split trainval --limit 2500
run cache-test  $PY scripts/vision_run.py cache --split test     --limit 1500

stage 3 "cross-encoder probe vs SigLIP bi-encoder"
run compare $PY scripts/vision_run.py train
# Stage 4 rebuilds the cache in place and re-runs train, so preserve this
# report before it is clobbered.
cp artifacts/vision/report.json artifacts/vision/report_block.json 2>/dev/null || true
cp artifacts/vision/logits.pt  artifacts/vision/logits_block.pt   2>/dev/null || true

stage 4 "same comparison under the causal mask, as a control"
# If the block mask is doing nothing, this should match stage 3. The text
# ablation found exactly that on synthetic data, so it is worth checking here
# rather than assuming images behave differently.
run cache-causal-train $PY scripts/vision_run.py cache --split trainval --limit 2500 --mask causal
run cache-causal-test  $PY scripts/vision_run.py cache --split test     --limit 1500 --mask causal
run compare-causal $PY scripts/vision_run.py train
cp artifacts/vision/report.json artifacts/vision/report_causal.json 2>/dev/null || true

stage 4b "block vs causal, side by side"
$PY - <<'EOF'
import json, os
try:
    b = json.load(open("artifacts/vision/report_block.json"))
    c = json.load(open("artifacts/vision/report_causal.json"))
except Exception as e:
    print("  could not compare:", e); raise SystemExit
print(f"  {'arm':<24}{'acc block':>11}{'acc causal':>12}{'ECE block':>11}{'ECE causal':>12}")
print("  " + "-" * 70)
for rb in b:
    rc = next((x for x in c if x["name"] == rb["name"]), None)
    if not rc: continue
    print(f"  {rb['name']:<24}{rb['accuracy']:>11.3f}{rc['accuracy']:>12.3f}"
          f"{rb['ece']:>11.3f}{rc['ece']:>12.3f}")
EOF

stage 5 summary
echo "  succeeded: ${ok[*]:-none}"
echo "  failed:    ${bad[*]:-none}"

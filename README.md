# systemone

A calibrated, type-safe decision model on an open backbone — the Jev
architecture rebuilt from Qwen3-0.6B.

No decoding loop. The state is encoded once, every question is scored against
it in one parallel pass, and each answer comes back as a distribution over an
option set that was declared before the call. A value outside that set is not
unlikely; it is unrepresentable.

## Quickstart

```bash
python -m venv .venv && ./.venv/bin/pip install -e . -r requirements.txt
./.venv/bin/python -m pytest                       # 9 tests, ~5s, no GPU
```

Build data (no GPU, no teacher, no annotation budget):

```bash
curl -O https://files.consumerfinance.gov/ccdb/complaints.csv.zip && unzip complaints.csv.zip
./.venv/bin/python scripts/build_data.py --cfpb-csv complaints.csv --civil-n 40000
```

Train, then produce the deliverable:

```bash
./.venv/bin/python scripts/train.py --data-dir artifacts/data --out artifacts/runs/run1
./.venv/bin/python scripts/evaluate.py --preds artifacts/runs/run1/val_preds.npz --plot reliability.png
```

## How it works

**No `lm_head`.** We load `AutoModel`, not `AutoModelForCausalLM`. For
Qwen3-0.6B that drops a 1024 × 151936 = 156M-parameter output projection —
about a quarter of the model — that we would otherwise run once per generated
token. `transformers` confirms it on load: `lm_head.weight | UNEXPECTED`.

**A shared `d → 1` probe**, not `d → k`. Parameters are independent of the
option count, which is what lets the option set change per request. The entire
head is 1,025 parameters.

**A block attention mask** (`systemone/model/mask.py`) — the part that is easy
to get wrong:

```
[ state ................. ][ question | options+slots ]
  bidirectional, and              bidirectional,
  attends nowhere forward         attends back to state
```

Three quadrants attend; `state → suffix` does not. That single masked quadrant
makes the state's keys and values independent of the question, which is what
lets one encode serve N questions. A globally bidirectional mask would destroy
it. `tests/test_mask.py::test_state_is_independent_of_suffix` asserts the state
hidden states are **bit-identical** (`atol=0`) under a changed suffix.

**Three answer shapes, one code path.** `noul` is a two-option `choice`
(softmax over two is sigmoid of the difference), and `score` is L ordered
levels read out as a probability-weighted expectation — monotone and smooth,
never a sampled digit.

**Confidence is `1 - H(p)/log k`.** Max-probability is not comparable across
questions with different arity: 0.5 is decisive in a binary and near-uniform
across ten options. Normalised entropy is k-invariant, which matters the moment
one threshold routes every question type.

## Data

| Corpus | Labels | Role |
|---|---|---|
| Civil Comments (1.8M) | **human annotator fraction** | calibration anchor |
| CFPB complaints | product · issue · response | real labelled choice tasks |

Civil Comments stores toxicity as the fraction of up to ten annotators who said
yes — a human-calibrated probability on every row. Do not threshold it; the
float is the target. CFPB is 80.5% credit reporting, so `cfpb.load()` caps per
product: sample it raw and the model learns that confidently guessing the
majority class is an excellent strategy, which is the degenerate solution the
objective exists to prevent.

## Objective

```
loss = cross_entropy(soft_target, p) + 0.3 * brier(p, y)
```

Brier is a strictly proper scoring rule, minimised exactly when stated
probabilities match true ones, and it decomposes into
`reliability - resolution + uncertainty`. Not label smoothing — it flatters ECE
while destroying the resolution threshold routing depends on.

You likely do not need RL here. RLHF needs it because preferences give no
gradient; RLVR needs it because correctness is a non-differentiable verdict.
Calibration has proper scoring rules, which are differentiable. Plain SGD.

## Evaluation

`scripts/evaluate.py` reports equal-mass ECE (fixed-width is bin-sensitive),
the Brier decomposition, and the selective-prediction curve — accuracy against
coverage as the confidence threshold sweeps. That last one is the operational
metric: at what coverage do you hit your error bar? Resolution is the number to
watch; a base-rate guesser is perfectly calibrated and perfectly useless.

## Status

Tests pass and the pipeline runs end to end on synthetic data. It has **not**
been trained on real data yet — awaiting AWS G-family quota. Expect ~1–1.5h on
a `g5.xlarge` for the week-one configuration.

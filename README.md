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

**No `lm_head`.** We load `AutoModel`, not `AutoModelForCausalLM`, so the
vocabulary projection is never built and never run. For Qwen3-0.6B that skips a
1024 × 151936 matmul at every position, which a decoder pays once per generated
token. `transformers` confirms the drop on load: `lm_head.weight | UNEXPECTED`.

This is a compute saving, not a memory one. `tie_word_embeddings` is true for
Qwen3-0.6B, so that matrix is the input embedding table reused transposed. We
still need it to embed tokens, so dropping the output projection frees no
parameters. (It is a real 622M-parameter saving on models that do not tie, such
as Qwen3-VL-8B.)

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

## The vision arm (`vision/`)

The same technique on a vision-language model. Nothing about the surgery
changes: a VLM turns pixels into tokens before the language model sees
anything, and the block mask, the slot head, the loss and the metrics are all
imported unchanged from the text pipeline above. Only the packing differs.

    python scripts/vision_run.py cache --split trainval --limit 2000
    python scripts/vision_run.py cache --split test     --limit 1500
    python scripts/vision_run.py train

**The experiment.** Oxford-IIIT Pets, 37 breeds, reshaped into questions whose
option list changes per call. Three question modes, each testing something
different:

| mode | option set | what it tests |
|---|---|---|
| `random` | true breed + uniform distractors | the easy case |
| `confusable` | true breed + distractors of the same species | whether options seeing each other helps |
| `abstain` | true breed REMOVED, `none of these` correct | a claim about the option set as a whole |

**The comparison.** Both arms train a small head on frozen features, so the
difference is wiring rather than budget:

* **cross-encoder** — option tokens go through the language model alongside the
  image and attend into it at every layer. A shared `d → 1` probe reads each
  option position. 2,049 parameters.
* **bi-encoder** — SigLIP-2 embeds the image and each label separately and they
  meet once at a dot product, which is how open-vocabulary classification
  already works. Gets a trained temperature, a trained diagonal reweighting,
  and a learned abstention rule so it is not a straw man.

Qwen3-VL's own vision tower *is* SigLIP-2, so both arms see the pixels through
the same kind of encoder.

**Why caching the backbone is legitimate here.** With the vision tower and the
language model both frozen, the hidden state at an option position is a fixed
function of the input, so it is computed once and the probe trains on the saved
vectors in seconds. The cost is that nothing below the probe can learn;
unfreezing the merger is the next step up and needs a real GPU.

`tests/test_vision_mask.py` checks the part that would invalidate everything
else if it were wrong: that the 4D block mask is actually applied, that the
state block's hidden states do not move when the question changes, and that a
globally-bidirectional control *does* move them. It runs on a randomly
initialised tiny model in seconds and needs no weights.

    state drift when the question changes     0.000e+00
    same test, globally bidirectional mask    5.480e-01

Bit-identical, and the control moves, so the mask is what is doing it. The
tiny model sets `deepstack_visual_indexes`, so DeepStack injection runs during
that test: injecting multi-level ViT features into the first three LLM layers
does **not** break the state's question-independence, because what is injected
depends only on the image.

**Status: nothing here has been trained.** The mechanism is verified and the
experiment is specified; the feature-extraction pass wants a GPU and the quota
request is still open. No numbers in this section, because there are none
yet.

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

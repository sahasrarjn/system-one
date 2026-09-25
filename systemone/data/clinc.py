"""CLINC150 -> records. The variable-option-set corpus.

Replaces CFPB, which as of September 2026 no longer publishes complaint
narratives in either its bulk download or its API: the structured metadata
survives, the consumer's text does not, and the text was the document.

CLINC150 is a better fit than CFPB was, for two reasons that matter to this
design rather than to convenience:

  * 150 intents instead of ten heavily skewed products, so the option set is
    large enough that presenting a DIFFERENT subset per request is a real
    test of the shared probe rather than a formality;
  * a native out-of-scope class, which gives the text side the same abstention
    question the vision side has. "None of these" is a claim about the option
    set as a whole, and it is the one thing a fixed-arity head cannot express.

Two caveats to keep in view.

Utterances are short, about 36 characters, so the state block is tiny and the
option suffix is the bulk of the sequence. That is the exact inverse of the
image case. Nothing about calibration changes; the KV-reuse argument simply
does not apply at this shape.

The published splits carry deliberately mismatched out-of-scope rates: 1.6% in
train, 3.2% in validation, 18.2% in test. That gap is a feature of the dataset,
built to stress out-of-scope detection, and it is a trap for a calibration
study. Fitting at a 1.6% abstention prior and scoring at 18.2% produces a model
that reads as badly miscalibrated for reasons that have nothing to do with the
method. So we build from the TRAIN split alone and let the pipeline carve its
own validation slice out of it, which keeps the prior matched. The official
test split is a separate, harder question -- calibration under prior shift --
and deserves to be reported as such rather than folded into the headline.
"""
import random
from typing import List

from .schema import Record, Question, onehot

DATASET = "clinc/clinc_oos"
CONFIG = "plus"
OOS = "oos"
NONE_OPTION = "none of these"
QUESTION = "What is the user asking for?"


def pretty(name: str) -> str:
    return name.replace("_", " ")


# Arity ladder rather than a narrow random range. Run 4 sampled 4-13 options
# out of 151 and produced a task the model found trivial: 97% accuracy, and a
# Brier resolution of 0.006 against an uncertainty of 0.028, meaning the stated
# confidence barely varied and carried almost no information. Both selective
# prediction figures came out at 1.0, which is the metric saying nothing at all.
#
# Spanning 4 to the full 150 does two things. It restores a real difficulty
# gradient, which is what resolution measures. And it turns the arity itself
# into a variable we can report against, which tests the claim the whole
# architecture rests on: that one shared probe stays calibrated as the option
# count changes.
ARITY_LADDER = (4, 8, 16, 32, 64, 150)

# Run 7 showed the arity ladder alone does not create difficulty: accuracy was
# flat from 2 options to 151, and Brier resolution stayed at 0.006, barely
# above the trivial task it replaced. Adding 147 obviously-wrong options does
# not confuse a model any more than adding three. Difficulty lives in
# CONFUSABILITY, not count.
#
# Rather than hardcode the published 10-domain grouping from memory, derive
# near-neighbours from the utterances themselves: a tf-idf centroid per intent
# and cosine between centroids. That measures what we actually want (which
# intents look alike in practice) rather than a proxy for it, and it is
# checkable. It recovers the domain structure and then some -- pto_request's
# nearest neighbours come out as pto_request_status, pto_balance, pto_used.
def confusability(ds, names, oos_id):
    """[n_intents, n_intents] cosine between tf-idf centroids. Diagonal and
    the out-of-scope row/column are set to -1 so they never rank as similar."""
    import collections, re
    import numpy as np

    docs = collections.defaultdict(list)
    for text, y in zip(ds["text"], ds["intent"]):
        if y != oos_id:
            docs[y].extend(re.findall(r"[a-z']+", text.lower()))
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs.values() for w in d}))}
    M = np.zeros((len(names), len(vocab)), dtype=np.float32)
    for y, words in docs.items():
        for w in words:
            M[y, vocab[w]] += 1.0
    idf = np.log(len(docs) / np.maximum((M > 0).sum(0), 1))
    M *= idf
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    S = M @ M.T
    np.fill_diagonal(S, -1.0)
    S[oos_id, :] = -1.0
    S[:, oos_id] = -1.0
    return S


def load(n: int = 20_000, split: str = "train", *, seed: int = 0,
         ladder=ARITY_LADDER, p_none: float = 0.5, p_hard: float = 0.7,
         hard_pool: int = 40):
    """One record per utterance, with an option subset that varies per record.

    For an in-scope utterance the true intent is always present, and a "none of
    these" escape hatch is added half the time so the model cannot learn that
    the escape hatch is never correct. For an out-of-scope utterance the true
    intent does not exist, every sampled option is wrong, and "none of these"
    is the answer.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG, split=split)
    names = ds.features["intent"].names
    oos_id = names.index(OOS)
    in_scope = [i for i in range(len(names)) if i != oos_id]
    rng = random.Random(seed)
    import numpy as np
    S = confusability(ds, names, oos_id)
    nearest = {y: [int(j) for j in np.argsort(-S[y])[:hard_pool]]
               for y in in_scope}

    # The published file is ordered by intent. Taking the first n rows yields
    # a handful of intents and, because the out-of-scope rows sit together, no
    # abstention cases at all. Shuffle the indices before truncating.
    order = list(range(len(ds)))
    rng.shuffle(order)
    order = order[:n]

    for i in order:
        row = ds[i]
        text = " ".join(str(row["text"]).split())
        if len(text) < 3:
            continue
        y = row["intent"]
        k = min(rng.choice(ladder), len(in_scope))

        mode = "abstain"
        if y == oos_id:
            opts = [pretty(names[j]) for j in rng.sample(in_scope, k)]
            opts.append(NONE_OPTION)
            rng.shuffle(opts)
            target = onehot(len(opts), opts.index(NONE_OPTION))
        else:
            # hard: distractors drawn from this intent's nearest neighbours,
            # which is where the model actually has to discriminate. easy:
            # uniform over everything else, as before. Mixing the two is what
            # produces a spread of difficulty, and a spread is what Brier
            # resolution measures.
            hard = rng.random() < p_hard and k - 1 <= hard_pool
            pool = nearest[y] if hard else [j for j in in_scope if j != y]
            if k - 1 >= len(pool):
                pool = [j for j in in_scope if j != y]
                hard = False
            mode = "hard" if hard else "easy"
            opts = [pretty(names[j]) for j in rng.sample(pool, k - 1)]
            opts.append(pretty(names[y]))
            if rng.random() < p_none:
                opts.append(NONE_OPTION)
            rng.shuffle(opts)
            target = onehot(len(opts), opts.index(pretty(names[y])))

        yield Record(
            state_id=f"clinc:{split}:{i}", state=text, source="clinc150",
            questions=[Question(id="intent", type="choice", options=opts,
                                target=target,
                                label_source=f"native/{mode}")],
        ).validate()

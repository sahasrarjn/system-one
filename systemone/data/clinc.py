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


def load(n: int = 20_000, split: str = "train", *, seed: int = 0,
         k_min: int = 4, k_max: int = 12, p_none: float = 0.5):
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
        k = rng.randint(k_min, k_max)

        if y == oos_id:
            opts = [pretty(names[j]) for j in rng.sample(in_scope, k)]
            opts.append(NONE_OPTION)
            rng.shuffle(opts)
            target = onehot(len(opts), opts.index(NONE_OPTION))
        else:
            pool = [j for j in in_scope if j != y]
            opts = [pretty(names[j]) for j in rng.sample(pool, k - 1)]
            opts.append(pretty(names[y]))
            if rng.random() < p_none:
                opts.append(NONE_OPTION)
            rng.shuffle(opts)
            target = onehot(len(opts), opts.index(pretty(names[y])))

        yield Record(
            state_id=f"clinc:{split}:{i}", state=text, source="clinc150",
            questions=[Question(id="intent", type="choice", options=opts,
                                target=target, label_source="native")],
        ).validate()

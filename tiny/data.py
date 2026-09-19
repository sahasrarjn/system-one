"""A synthetic decision task whose TRUE posterior we can compute exactly.

Why synthetic: to learn calibration you need to know the right answer's
*probability*, not just its label. Here we choose the generative process, so
Bayes' rule gives the exact P(category | document) in closed form. That gives
us two things real data cannot:

  1. a calibration target that is truth, not an estimate
  2. a known accuracy CEILING, so we can tell "the model is bad" apart from
     "the task is genuinely ambiguous"

Generative process for one document:
    c ~ Uniform(categories)
    L ~ Uniform(12, 24)
    each of the L words is drawn from a MIXTURE:
        with prob `signal`  -> uniform over c's signature words
        otherwise           -> uniform over the whole vocabulary

The mixture is what keeps every word possible under every category (full
support), so the posterior is graded rather than degenerate. Turn `signal`
down and the task gets more ambiguous; the true posteriors move away from
1.0 and calibration starts to matter.
"""
import math, random
from typing import List, Tuple

CATEGORIES = ["billing", "technical", "sales", "spam"]

SIGNATURE = {
    "billing":   ["invoice", "charge", "refund", "payment", "overcharged", "bill"],
    "technical": ["error", "crash", "timeout", "login", "bug", "restart"],
    "sales":     ["pricing", "demo", "quote", "upgrade", "plan", "enterprise"],
    "spam":      ["winner", "free", "click", "prize", "urgent", "congratulations"],
}

FILLER = ["the", "a", "my", "your", "please", "help", "account", "issue",
          "today", "again", "cannot", "need", "this", "with", "about", "now"]

VOCAB = FILLER + [w for c in CATEGORIES for w in SIGNATURE[c]]
STOI = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
PAD = V  # one extra id for padding


def word_logprobs(signal: float):
    """log P(word | category) for the mixture above. Shape [C][V]."""
    out = []
    for c in CATEGORIES:
        sig = set(SIGNATURE[c])
        row = []
        for w in VOCAB:
            p = (1.0 - signal) / V                      # background
            if w in sig:
                p += signal / len(sig)                  # category signal
            row.append(math.log(p))
        out.append(row)
    return out


def true_posterior(token_ids: List[int], logp) -> List[float]:
    """EXACT Bayes posterior P(category | document).

    log P(c|d) = log P(c) + sum_w log P(w|c) + const.  Uniform prior, so the
    prior term is constant and drops out of the softmax.
    """
    scores = [sum(logp[ci][t] for t in token_ids) for ci in range(len(CATEGORIES))]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps)
    return [e / Z for e in exps]


def sample_doc(rng: random.Random, signal: float = 0.20) -> Tuple[List[int], int]:
    ci = rng.randrange(len(CATEGORIES))
    sig = SIGNATURE[CATEGORIES[ci]]
    L = rng.randint(6, 14)
    ids = []
    for _ in range(L):
        if rng.random() < signal:
            ids.append(STOI[rng.choice(sig)])
        else:
            ids.append(rng.randrange(V))
    return ids, ci


def make_dataset(n: int, signal: float = 0.20, seed: int = 0):
    """Returns (ids, true_label, exact_posterior) triples."""
    rng = random.Random(seed)
    logp = word_logprobs(signal)
    out = []
    for _ in range(n):
        ids, ci = sample_doc(rng, signal)
        out.append((ids, ci, true_posterior(ids, logp)))
    return out


def bayes_ceiling(data) -> float:
    """Accuracy of the optimal decision rule. No model can beat this; the gap
    between it and 100% is irreducible ambiguity, not a modelling failure."""
    return sum(1.0 for _, y, p in data if max(range(len(p)), key=p.__getitem__) == y) / len(data)


def decode(ids: List[int]) -> str:
    return " ".join(VOCAB[i] for i in ids)


# --------------------------------------------------------------- packing
# Token layout:  [0 .. V-1] words | [V .. V+C-1] one token per option | PAD
OPT_BASE = V
PAD_ID = V + len(CATEGORIES)
VOCAB_SIZE = PAD_ID + 1
MAX_DOC = 14


def pack(doc_ids, option_cats, max_doc: int = MAX_DOC):
    """[doc padded to max_doc][one token per declared option].

    Every document is padded to the SAME length so n_state is a scalar and
    the mask stays a single [T,T] matrix. The production repo masks padding
    as keys; here we let the model learn to ignore PAD, to keep the code
    readable. That is the only shortcut in this file.
    """
    d = list(doc_ids[:max_doc]) + [PAD_ID] * max(0, max_doc - len(doc_ids))
    ids = d + [OPT_BASE + c for c in option_cats]
    slots = [max_doc + j for j in range(len(option_cats))]
    return ids, max_doc, slots

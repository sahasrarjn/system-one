"""Oxford-IIIT Pets, reshaped into the question format.

Why this dataset: 37 breeds, and the breeds come in two species. That gives a
free and honest notion of a confusable distractor (another breed of the same
species) versus an easy one (the other species entirely), without inventing a
similarity metric or leaking the model's own embeddings into the task design.

Two option-set modes exist because they test different claims:

  confusable  the answer turns on telling two near-neighbours apart, which is
              where a cross-encoder should beat independent dot products
  abstain     the true breed is REMOVED and "none of these" is correct, which
              is a statement about the option set as a whole and therefore
              something a per-label dot product cannot represent
"""
import random
from dataclasses import dataclass
from typing import List

NONE_OPTION = "none of these"
QUESTION = "What breed is the animal in the photograph?"

# Oxford-IIIT Pets: the first 12 classes alphabetically are cats, the rest
# dogs. Stored explicitly rather than inferred, so it is checkable.
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}


@dataclass
class Example:
    image_index: int
    question: str
    options: List[str]
    answer: int          # index into options
    mode: str            # random | confusable | abstain
    k: int


def pretty(name: str) -> str:
    return name.replace("_", " ").strip()


def build_examples(labels: List[int], classes: List[str], *, seed: int = 0,
                   k_min: int = 2, k_max: int = 6,
                   p_confusable: float = 0.5, p_abstain: float = 0.2
                   ) -> List[Example]:
    """One question per image, with an option set that varies per question.

    The option count varies deliberately: a fixed-arity head cannot do this at
    all, so it is the part of the setup that needs the shared probe.
    """
    rng = random.Random(seed)
    names = [pretty(c) for c in classes]
    cats = [i for i, n in enumerate(names) if n in CAT_BREEDS]
    dogs = [i for i in range(len(names)) if i not in set(cats)]
    assert len(cats) == 12, f"expected 12 cat breeds, found {len(cats)}"

    out = []
    for idx, y in enumerate(labels):
        k = rng.randint(k_min, k_max)
        same = cats if y in cats else dogs

        if rng.random() < p_confusable:
            pool, mode = [i for i in same if i != y], "confusable"
        else:
            pool, mode = [i for i in range(len(names)) if i != y], "random"

        abstain = rng.random() < p_abstain
        distractors = rng.sample(pool, min(k - 1, len(pool)))

        if abstain:
            # true answer withheld; "none of these" is correct
            opts = [names[i] for i in distractors] + [NONE_OPTION]
            rng.shuffle(opts)
            ans = opts.index(NONE_OPTION)
            mode = "abstain"
        else:
            opts = [names[i] for i in distractors] + [names[y]]
            rng.shuffle(opts)
            ans = opts.index(names[y])
            # every non-abstain question also carries the escape hatch, so the
            # model cannot learn that "none of these" is always right
            if rng.random() < 0.5:
                opts.append(NONE_OPTION)

        out.append(Example(idx, QUESTION, opts, ans, mode, len(opts)))
    return out


def load_pets(root: str = "artifacts/data/pets", split: str = "trainval"):
    """Returns (dataset, classes). Downloads ~800MB on first call."""
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(root=root, split=split, target_types="category",
                       download=True)
    return ds, list(ds.classes)

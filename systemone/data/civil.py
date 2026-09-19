"""Civil Comments -> records. The calibration anchor.

1,804,874 comments, each shown to up to ten annotators (some to more than a
hundred), with toxicity stored as THE FRACTION OF ANNOTATORS WHO SAID YES.

That fraction is a human-calibrated probability on every row. It is exactly
the target this design is chasing, with no LLM in the loop and no annotation
budget. Do NOT threshold it to a boolean - the float is the point.
"""
from .schema import Record, Question


def load(n: int = 40_000, split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset("google/civil_comments", split=split, streaming=True)

    for i, row in enumerate(ds):
        if i >= n:
            break
        text = " ".join(str(row["text"]).split())
        if len(text) < 20:
            continue
        p = float(row["toxicity"])
        yield Record(
            state_id=f"civil:{i}", state=text, source="civil_comments",
            questions=[Question(
                id="toxic", type="noul", options=["no", "yes"],
                target=[1.0 - p, p],            # <- the human distribution
                label_source="human/civil_comments",
            )],
        ).validate()

"""Civil Comments -> records. The calibration anchor.

1,804,874 comments, each shown to up to ten annotators (some to more than a
hundred), with toxicity stored as THE FRACTION OF ANNOTATORS WHO SAID YES.

That fraction is a human-calibrated probability on every row. It is exactly
the target this design is chasing, with no LLM in the loop and no annotation
budget. Do NOT threshold it to a boolean - the float is the point.
"""
from .schema import Record, Question


def load(n: int = 40_000, split: str = "train", *, seed: int = 0):
    """Not streaming, for two reasons. Abandoning a streaming generator part
    way leaves a background reader that aborts the interpreter at exit
    ("PyGILState_Release: auto-releasing thread-state"), which fails the stage
    after its work is already done. And a materialised split can be shuffled,
    so `n` is a sample rather than whatever happens to sit at the top of the
    file."""
    from datasets import load_dataset

    # shuffle().select() keeps access sequential over the arrow file; indexing
    # a shuffled list row by row would be random I/O over ~1.8M rows.
    ds = load_dataset("google/civil_comments", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    for i, row in enumerate(ds):
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

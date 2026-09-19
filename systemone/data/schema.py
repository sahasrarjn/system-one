"""The one record shape every corpus normalises into.

`label_source` is load-bearing: it is what later lets you evaluate against
human ground truth separately from teacher-distilled labels. Never drop it.
"""
from dataclasses import dataclass, asdict
from typing import List, Literal
import json

QType = Literal["choice", "noul", "score"]


@dataclass
class Question:
    id: str
    type: QType
    options: List[str]          # noul -> ["no","yes"]; score -> level labels
    target: List[float]         # a DISTRIBUTION, sums to 1. one-hot if native.
    label_source: str           # "native" | "teacher/<models>" | "human/<n>"

    def validate(self):
        assert len(self.options) == len(self.target), f"{self.id}: arity mismatch"
        assert len(self.options) >= 2, f"{self.id}: need >=2 options"
        assert abs(sum(self.target) - 1.0) < 1e-4, f"{self.id}: target must sum to 1"
        if self.type == "noul":
            assert len(self.options) == 2, "noul is a 2-option choice"
        if self.type == "score":
            assert 2 <= len(self.options) <= 10, "score takes 2-10 ordered levels"


@dataclass
class Record:
    state_id: str
    state: str
    source: str
    questions: List[Question]

    def validate(self):
        assert self.state.strip(), f"{self.state_id}: empty state"
        for q in self.questions:
            q.validate()
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Record":
        d = json.loads(line)
        d["questions"] = [Question(**q) for q in d["questions"]]
        return Record(**d)


def onehot(n: int, i: int) -> List[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v

"""Records -> packed training examples. One example per QUESTION."""
import json, random
import torch
from torch.utils.data import Dataset

from .schema import Record
from ..model.packing import pack_one, collate

QUESTION_TEXT = {
    "product": "Which financial product is this complaint about?",
    "company_response": "How did the company close this complaint?",
    "toxic": "Would a reader consider this comment toxic?",
}


class DecisionDataset(Dataset):
    def __init__(self, path, tokenizer, cfg, shuffle_options=True):
        self.items = []
        self.tok, self.cfg = tokenizer, cfg
        self.shuffle_options = shuffle_options
        with open(path) as f:
            for line in f:
                r = Record.from_json(line)
                for q in r.questions:
                    self.items.append((r.state, q))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        state, q = self.items[i]
        options, target = list(q.options), list(q.target)

        # Shuffle option order. Under a bidirectional suffix the scores should
        # already be order-invariant; shuffling makes sure we never learn a
        # positional shortcut, and turns order-sensitivity into a measurable bug.
        if self.shuffle_options and q.type != "score":
            perm = list(range(len(options)))
            random.shuffle(perm)
            options = [options[j] for j in perm]
            target = [target[j] for j in perm]

        text = QUESTION_TEXT.get(q.id, q.id.replace("_", " ") + "?")
        ids, ns, slots = pack_one(self.tok, state, text, options,
                                  self.cfg.max_state_tokens,
                                  self.cfg.max_option_tokens)
        return ids, ns, slots, target


def make_collate(pad_id):
    return lambda batch: collate(batch, pad_id)

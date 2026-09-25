"""Records -> packed training examples. One example per QUESTION."""
import json, random
import torch
from torch.utils.data import Dataset

from .schema import Record
from ..model.packing import pack_one, collate

QUESTION_TEXT = {
    "intent": "What is the user asking for?",
    "toxic": "Would a reader consider this comment toxic?",
}


class DecisionDataset(Dataset):
    def __init__(self, path, tokenizer, cfg, shuffle_options=True):
        self.items = []
        # Parallel list, same order. A blended corpus makes a single accuracy
        # number ambiguous: a binary task with a skewed prior can carry the
        # headline while a 13-option task learns nothing. Evaluation needs to
        # be able to split them.
        self.sources = []
        self.modes = []
        self.tok, self.cfg = tokenizer, cfg
        self.shuffle_options = shuffle_options
        with open(path) as f:
            for line in f:
                r = Record.from_json(line)
                for q in r.questions:
                    self.items.append((r.state, q))
                    self.sources.append(r.source)
                    # label_source carries "native/hard" etc; the suffix says
                    # how the distractors were drawn, which is the variable
                    # that actually moves difficulty.
                    self.modes.append(q.label_source.split("/")[-1])

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


    def approx_lengths(self):
        """Cheap per-example sequence-length estimate, for batching.

        Tokenising 28k examples up front just to size batches is not worth the
        minute it costs; word count times 1.3 plus four tokens per option is
        close enough to bucket by, and the collate still pads exactly.
        """
        out = []
        for state, q in self.items:
            n_state = min(int(len(state.split()) * 1.3) + 2,
                          self.cfg.max_state_tokens)
            out.append(n_state + 8 + 4 * len(q.options))
        return out


class TokenBudgetSampler(torch.utils.data.Sampler):
    """Batches sized by total tokens, not by example count.

    A fixed batch size is wrong when sequence length varies twentyfold. It gets
    tuned against the average and then dies on the tail: run 5 trained twenty
    steps at batch 8 and then hit a batch holding several 151-option questions.
    Halving the batch only moves that cliff. Budgeting tokens bounds the thing
    that actually drives memory, so wide questions simply travel in smaller
    groups.

    Examples are sorted by length inside large shuffled chunks, which also
    removes most padding waste, and the batch order is shuffled again so the
    model does not see all the short questions first.
    """

    def __init__(self, lengths, max_tokens=4096, max_batch=32, seed=0):
        self.lengths, self.max_tokens = lengths, max_tokens
        self.max_batch, self.seed, self.epoch = max_batch, seed, 0

    def _build(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        chunk = self.max_batch * 64
        batches, cur, cur_max = [], [], 0
        for c0 in range(0, len(idx), chunk):
            for i in sorted(idx[c0:c0 + chunk], key=lambda j: self.lengths[j]):
                m = max(cur_max, self.lengths[i])
                if cur and ((len(cur) + 1) * m > self.max_tokens
                            or len(cur) >= self.max_batch):
                    batches.append(cur); cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i); cur_max = m
            if cur:
                batches.append(cur); cur, cur_max = [], 0
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._build()
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self._build())


def make_collate(pad_id):
    return lambda batch: collate(batch, pad_id)

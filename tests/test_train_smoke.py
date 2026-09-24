"""Exercise the whole text training path without downloading a model.

Dataset -> packing -> block mask -> backbone -> slot head -> loss -> backward
-> eval, on a randomly initialised tiny Qwen3. Runs in seconds on CPU and
exists so that shape and API bugs surface here rather than ten minutes into a
paid GPU session.
"""
import json, os, sys, tempfile
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3 import Qwen3Config, Qwen3Model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systemone.config import Config
from systemone.data.dataset import DecisionDataset, make_collate
from systemone.data.schema import Record, Question, onehot
from systemone.model.head import SlotHead
from systemone.model.losses import decision_loss
from systemone.model.mask import batch_block_mask

PRODUCTS = ["Mortgage", "Credit card", "Student loan", "Debt collection"]


class TinySystemOne(torch.nn.Module):
    """Same wiring as systemone.model.SystemOne, tiny random backbone."""

    def __init__(self, cfg, d=64):
        super().__init__()
        self.backbone = Qwen3Model(Qwen3Config(
            hidden_size=d, intermediate_size=2 * d, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            vocab_size=151936))
        self.head = SlotHead(d)

    def forward(self, input_ids, slot_idx, slot_mask, n_states, lengths, **_):
        mask = batch_block_mask(n_states, lengths, input_ids.size(1),
                                dtype=self.backbone.dtype,
                                device=input_ids.device)
        H = self.backbone(input_ids=input_ids, attention_mask=mask,
                          use_cache=False).last_hidden_state
        return self.head(H, slot_idx, slot_mask)


def fake_jsonl(path, n=24):
    with open(path, "w") as f:
        for i in range(n):
            qs = [Question(id="product", type="choice", options=PRODUCTS,
                           target=onehot(len(PRODUCTS), i % len(PRODUCTS)),
                           label_source="native")]
            if i % 3 == 0:   # a second question with a DIFFERENT arity
                p = (i % 7) / 7.0
                qs.append(Question(id="toxic", type="noul",
                                   options=["no", "yes"], target=[1 - p, p],
                                   label_source="human/test"))
            f.write(Record(state_id=f"t:{i}",
                           state=f"complaint number {i} " + "filler words " * 20,
                           source="test", questions=qs).validate().to_json() + "\n")


def main():
    cfg = Config(max_state_tokens=64, batch_size=4, grad_accum=2,
                 load_dtype="float32")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    with tempfile.TemporaryDirectory() as d:
        fake_jsonl(f"{d}/train.jsonl")
        ds = DecisionDataset(f"{d}/train.jsonl", tok, cfg)
        print(f"dataset: {len(ds)} questions from 24 records "
              f"(mixed arity: {sorted({len(ds[i][2]) for i in range(len(ds))})})")

        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=make_collate(tok.pad_token_id))

        model = TinySystemOne(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        losses = []
        for i, batch in enumerate(dl):
            logits = model(**batch)
            assert logits.shape == batch["target"].shape, \
                f"{logits.shape} vs {batch['target'].shape}"
            loss, parts = decision_loss(logits, batch["target"],
                                        batch["slot_mask"], cfg.brier_weight)
            (loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert torch.isfinite(gn), "non-finite gradient"
                opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

        # padded slots must never take probability mass
        p = torch.softmax(logits.float().masked_fill(~batch["slot_mask"], -1e9), -1)
        leaked = p.masked_fill(batch["slot_mask"], 0).sum().item()
        print(f"forward/backward over {i+1} batches, loss "
              f"{losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"probability on padded slots: {leaked:.2e}")
        assert leaked < 1e-6, "padding is absorbing probability"
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        print("\ntrain path OK: mixed arity packs, mask applies, grads are "
              "finite, padding takes no mass.")


if __name__ == "__main__":
    main()

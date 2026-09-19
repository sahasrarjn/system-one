"""The objective. Cross-entropy against a soft target, plus Brier.

Note on KL vs cross-entropy: KL(target || p) = H(target, p) - H(target), and
H(target) is constant with respect to the parameters, so optimising
cross-entropy against the soft target is equivalent and cheaper.

Brier is a STRICTLY PROPER scoring rule, minimised exactly when the stated
probabilities match the true ones, and it decomposes into
reliability - resolution + uncertainty. That is why it is here and why label
smoothing is not: smoothing flatters ECE while destroying the resolution that
threshold routing depends on.
"""
import torch
import torch.nn.functional as F


def decision_loss(logits, target, slot_mask, brier_weight: float = 0.3):
    logits = logits.float()
    logp = torch.log_softmax(logits.masked_fill(~slot_mask, -1e9), dim=-1)
    p = logp.exp()

    ce = -(target * logp.masked_fill(~slot_mask, 0.0)).sum(-1).mean()

    hard = target.argmax(-1)
    onehot = F.one_hot(hard, target.size(-1)).float()
    brier = (((p - onehot) ** 2) * slot_mask).sum(-1).mean()

    return ce + brier_weight * brier, {"ce": ce.item(), "brier": brier.item()}

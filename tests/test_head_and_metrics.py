import numpy as np, torch
from systemone.model.head import SlotHead
from systemone.model.systemone import confidence
from systemone.model.losses import decision_loss
from systemone.evaluation.metrics import ece, brier_decomposition, selective


def test_head_is_arity_independent():
    """The probe is d->1, so parameter count must not depend on k."""
    h = SlotHead(64)
    n = sum(p.numel() for p in h.parameters())
    assert n == 64 + 1
    H = torch.randn(2, 10, 64)
    for k in (2, 5, 9):
        idx = torch.arange(k).expand(2, k)
        m = torch.ones(2, k, dtype=torch.bool)
        assert h(H, idx, m).shape == (2, k)


def test_padded_slots_get_no_mass():
    h = SlotHead(32)
    H = torch.randn(1, 12, 32)
    idx = torch.tensor([[0, 1, 2, 0]])
    m = torch.tensor([[True, True, True, False]])
    p = torch.softmax(h(H, idx, m).float(), -1)
    assert p[0, 3] < 1e-6


def test_confidence_is_k_invariant():
    """max-prob would call 0.5-in-a-binary and 0.5-of-ten equally confident."""
    binary = confidence(torch.tensor([0.5, 0.5]))
    ten = confidence(torch.tensor([0.5] + [0.5 / 9] * 9))
    assert binary < 1e-6                      # a coin flip is zero confidence
    assert ten > 0.2                          # 0.5 of ten is informative


def test_loss_rewards_matching_the_soft_target():
    torch.manual_seed(0)
    m = torch.ones(1, 3, dtype=torch.bool)
    tgt = torch.tensor([[0.6, 0.3, 0.1]])
    matched = torch.log(tgt)
    overconf = torch.tensor([[8.0, 0.0, -8.0]])
    a, _ = decision_loss(matched, tgt, m)
    b, _ = decision_loss(overconf, tgt, m)
    assert a < b, "an over-confident model must be penalised"


def test_metrics_on_synthetic():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 4000)
    correct = (rng.uniform(size=4000) < p).astype(float)   # perfectly calibrated
    assert ece(p, correct) < 0.03

    bad = np.clip(p + 0.25, 0, 1)                          # over-confident
    assert ece(bad, correct) > ece(p, correct)

    d = brier_decomposition(p, correct)
    assert d["resolution"] > 0
    cov, acc = selective(p, correct)
    assert acc[:50].mean() > acc[-50:].mean()              # conf must rank

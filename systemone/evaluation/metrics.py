"""The harness TypeSafe never published.

Four numbers matter, and accuracy is the least interesting of them.
"""
import numpy as np


def reliability(p_top: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    """EQUAL-MASS bins. Fixed-width ECE is notoriously bin-sensitive and can be
    flattered by accident; equal-mass puts the same count in every bin."""
    order = np.argsort(p_top)
    bins = np.array_split(order, n_bins)
    return [(float(p_top[b].mean()), float(correct[b].mean()), int(len(b)))
            for b in bins if len(b)]


def ece(p_top, correct, n_bins: int = 15) -> float:
    pts = reliability(p_top, correct, n_bins)
    n = sum(c for _, _, c in pts)
    return float(sum(c / n * abs(conf - acc) for conf, acc, c in pts))


def brier_decomposition(p_top, correct, n_bins: int = 15):
    """Murphy's decomposition: brier = reliability - resolution + uncertainty.

    Resolution is the one to watch. A model that always answers the base rate
    is perfectly calibrated and perfectly useless; resolution is what separates
    honest from informative.
    """
    base = float(correct.mean())
    pts = reliability(p_top, correct, n_bins)
    n = len(p_top)
    rel = sum(c / n * (conf - acc) ** 2 for conf, acc, c in pts)
    res = sum(c / n * (acc - base) ** 2 for conf, acc, c in pts)
    unc = base * (1 - base)
    return {"brier": float(np.mean((p_top - correct) ** 2)),
            "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc)}


def selective(conf: np.ndarray, correct: np.ndarray):
    """THE operational curve: sweep the confidence threshold and report
    accuracy against coverage. This is what you show whoever is deciding how
    much traffic to automate."""
    o = np.argsort(-conf)
    cov = np.arange(1, len(o) + 1) / len(o)
    acc = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
    return cov, acc


def coverage_at(conf, correct, target_acc: float):
    """What fraction of traffic clears the bar at a given error rate?"""
    cov, acc = selective(conf, correct)
    ok = np.where(acc >= target_acc)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0

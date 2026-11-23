"""Threshold rules. All are fitted on scores of NORMAL calibration data only (no fraud labels).

The first five are the rules tried during the hackathon (kept for comparison);
`quantile` is the one adopted (ADR-005): pick the alert rate you can afford on normal traffic.
"""
import numpy as np
import pandas as pd


def fixed(scores, value=0.05):
    return float(value)


def batch_percentile(scores, q=96):
    return float(np.percentile(scores, q))


def tukey(scores, k=1.5):
    q1, q3 = np.percentile(scores, [25, 75])
    return float(q3 + k * (q3 - q1))


def mad(scores, k=3.0):
    med = np.median(scores)
    return float(med + k * np.median(np.abs(scores - med)))


def skew_percentile(scores):
    q = min(99, 90 + 5 * abs(pd.Series(scores).skew()))
    return float(np.percentile(scores, q))


def quantile(scores, target_fpr=0.01):
    """Threshold such that `target_fpr` of normal calibration users get flagged."""
    return float(np.quantile(scores, 1 - target_fpr))


RULES = {
    "fixed_0.05 (third.ipynb)": fixed,
    "p96 (api.py)": batch_percentile,
    "tukey_1.5IQR (third(1).ipynb)": tukey,
    "median+3MAD": mad,
    "skew_adjusted_percentile": skew_percentile,
    "quantile_fpr1% (adopted)": quantile,
}

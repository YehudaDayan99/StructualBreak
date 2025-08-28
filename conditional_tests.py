"""
Conditional statistical tests for ADIA structural breakpoint detection.
"""

import numpy as np
from typing import Tuple, Callable
from .core_statistics import jackknife_mu, nw_variance
from .config import OVERLAP_THRESHOLD, EPSILON


def T_mu_statistic(x_grid: np.ndarray, X: np.ndarray, Y: np.ndarray, idx_split: int, h: float) -> Tuple[float, float]:
    X1, Y1 = X[:idx_split+1], Y[:idx_split+1]
    X2, Y2 = X[idx_split+1:], Y[idx_split+1:]
    mu1, _ = jackknife_mu(x_grid, X1, Y1, h)
    mu2, _ = jackknife_mu(x_grid, X2, Y2, h)
    mu_all, f_all = jackknife_mu(x_grid, X, Y, h)
    sig2_all = nw_variance(x_grid, X, Y, mu_all, h)
    scale = np.sqrt(np.maximum(f_all, EPSILON) / np.maximum(sig2_all, EPSILON))
    diff = np.abs(mu1 - mu2)
    T = float(np.max(scale * diff))
    overlap = float(np.mean(f_all > (OVERLAP_THRESHOLD / (len(X) * h))))
    return T, overlap


def T_sigma_statistic(x_grid: np.ndarray, X: np.ndarray, Y: np.ndarray, idx_split: int, h: float) -> Tuple[float, float]:
    X1, Y1 = X[:idx_split+1], Y[:idx_split+1]
    X2, Y2 = X[idx_split+1:], Y[idx_split+1:]
    mu1, _ = jackknife_mu(x_grid, X1, Y1, h)
    mu2, _ = jackknife_mu(x_grid, X2, Y2, h)
    sig1 = np.sqrt(np.maximum(nw_variance(x_grid, X1, Y1, mu1, h), EPSILON))
    sig2 = np.sqrt(np.maximum(nw_variance(x_grid, X2, Y2, mu2, h), EPSILON))
    _, f_all = jackknife_mu(x_grid, X, Y, h)
    logdiff = np.abs(np.log(sig1) - np.log(sig2))
    T = float(np.max(np.sqrt(np.maximum(f_all, EPSILON)) * logdiff))
    overlap = float(np.mean(f_all > (OVERLAP_THRESHOLD / (len(X) * h))))
    return T, overlap


def circular_shift_bootstrap(stat_fn: Callable, X: np.ndarray, Y: np.ndarray, idx_split: int, h: float, x_grid: np.ndarray, B: int = 80, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(X)
    T_obs, _ = stat_fn(x_grid, X, Y, idx_split, h)
    Tb = np.empty(B)
    for b in range(B):
        s = int(rng.integers(0, n))
        Xb = np.roll(X, s)
        Yb = np.roll(Y, s)
        Tb[b], _ = stat_fn(x_grid, Xb, Yb, idx_split, h)
    p = (1.0 + np.sum(Tb >= T_obs)) / (B + 1.0)
    return float(T_obs), float(p)


def conditional_test_summary(x_grid: np.ndarray, X: np.ndarray, Y: np.ndarray, idx_split: int, h: float, B: int = 80, seed: int = 123) -> dict:
    T_mu, overlap_mu = T_mu_statistic(x_grid, X, Y, idx_split, h)
    _, p_mu = circular_shift_bootstrap(T_mu_statistic, X, Y, idx_split, h, x_grid, B, seed)
    T_sig, overlap_sig = T_sigma_statistic(x_grid, X, Y, idx_split, h)
    _, p_sig = circular_shift_bootstrap(T_sigma_statistic, X, Y, idx_split, h, x_grid, B, seed)
    return {
        'T_mu': T_mu,
        'p_mu': p_mu,
        'overlap_mu': overlap_mu,
        'T_sigma': T_sig,
        'p_sigma': p_sig,
        'overlap_sigma': overlap_sig,
        'idx_split': idx_split,
        'n_total': len(X),
        'n_before': idx_split + 1,
        'n_after': len(X) - idx_split - 1
    }

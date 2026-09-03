"""
GPU acceleration backend for temporal SETI.

WHY: The Monte Carlo permutation tests are the main computational bottleneck.
Each test generates hundreds of null distributions and computes a metric on
each one. On CPU with 10K+ photons and 100+ permutations, a single detector
can take minutes. GPU parallelization reduces this to seconds.

This module provides a transparent array backend: if cupy (GPU) is available,
operations run on the GPU; otherwise they fall back to numpy (CPU) with no
code changes needed in the detectors.

On Jetson (CUDA + ARM), cupy or torch.cuda may be available. On systems
without GPU, everything falls back to numpy transparently.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Try to import GPU backends in order of preference
_GPU_BACKEND: Optional[str] = None
_cupy = None

try:
    import cupy as cp
    _cupy = cp
    _GPU_BACKEND = "cupy"
except ImportError:
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_BACKEND = "torch"
        else:
            _GPU_BACKEND = None
    except ImportError:
        _GPU_BACKEND = None


def gpu_available() -> bool:
    """Check if any GPU backend is available."""
    return _GPU_BACKEND is not None


def get_backend_name() -> str:
    """Return the name of the active backend ('cupy', 'torch', or 'cpu')."""
    if _GPU_BACKEND == "cupy":
        return "cupy"
    elif _GPU_BACKEND == "torch":
        return "torch"
    else:
        return "cpu"


def asarray(arr, use_gpu: bool = True):
    """Convert a numpy array to the active backend's array type.

    If GPU is available and use_gpu=True, returns a cupy array.
    Otherwise returns the numpy array unchanged.
    """
    if use_gpu and _GPU_BACKEND == "cupy":
        return cp.asarray(arr)
    return np.asarray(arr)


def asnumpy(arr):
    """Convert any backend array back to numpy."""
    if _GPU_BACKEND == "cupy" and hasattr(arr, 'get'):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def generate_poisson_nulls(n_photons: int, exposure: float, n_perm: int,
                            seed: int = 42) -> np.ndarray:
    """Generate a batch of Poisson null distributions efficiently.

    WHY: This is the hot loop of the permutation test. Instead of generating
    one null at a time in Python, we generate all n_perm nulls at once as a
    2D array (n_perm × n_photons) and sort each row. On GPU this is a single
    kernel launch; on CPU it's still vectorized via numpy.

    Args:
        n_photons: number of photons in each null realization
        exposure: observation duration (uniform over [0, exposure])
        n_perm: number of null realizations to generate
        seed: random seed

    Returns:
        np.ndarray of shape (n_perm, n_photons), each row sorted ascending
    """
    if _GPU_BACKEND == "cupy" and n_perm * n_photons > 50000:
        # GPU path: generate and sort on device
        rng = cp.random.RandomState(seed)
        nulls = rng.uniform(0, exposure, size=(n_perm, n_photons))
        nulls = cp.sort(nulls, axis=1)
        return cp.asnumpy(nulls)
    else:
        # CPU path: vectorized numpy
        rng = np.random.default_rng(seed)
        nulls = rng.uniform(0, exposure, size=(n_perm, n_photons))
        nulls.sort(axis=1)
        return nulls


def batch_metric(metric_fn, observed_times: np.ndarray,
                  nulls: np.ndarray) -> tuple[float, float]:
    """Compute a metric on observed data and all null realizations.

    WHY: This vectorizes the metric computation across all permutations.
    The metric_fn is called once on the observed data and once per null
    row. For GPU backends, the metric_fn should use cupy operations to
    stay on device.

    Args:
        metric_fn: function(times) -> float
        observed_times: the real photon arrival times
        nulls: (n_perm, n_photons) array of null arrival times

    Returns:
        (observed_metric, p_value) where p_value is the fraction of
        null metrics >= observed metric
    """
    observed = metric_fn(observed_times)
    n_perm = len(nulls)
    count_exceed = 0

    for i in range(n_perm):
        null_metric = metric_fn(nulls[i])
        if null_metric >= observed:
            count_exceed += 1

    pvalue = count_exceed / n_perm if n_perm > 0 else 0.5
    return observed, pvalue
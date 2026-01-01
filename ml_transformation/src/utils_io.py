# utils_io.py
import os
import numpy as np

def _infer_txt_layout(arr: np.ndarray, n_traces_hint=None, n_samples_hint=None):
    """Return data with shape (nsamp, ntr)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if n_traces_hint is None or n_samples_hint is None:
            raise ValueError("1D TXT requires --ntraces and --nsamp to reshape.")
        if arr.size != n_traces_hint * n_samples_hint:
            raise ValueError("TXT size does not match ntraces*nsamp.")
        return arr.reshape((n_samples_hint, n_traces_hint))

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D.")

    r, c = arr.shape

    if n_traces_hint is not None and n_samples_hint is not None:
        if (r, c) == (n_samples_hint, n_traces_hint):
            return arr
        if (r, c) == (n_traces_hint, n_samples_hint):
            return arr.T
        if r == n_samples_hint:
            return arr
        if c == n_samples_hint:
            return arr.T

    # Heuristic: choose orientation that yields more traces in columns.
    return arr if r < c else arr.T

def read_txt_matrix(path: str, delimiter=None, n_traces_hint=None, n_samples_hint=None):
    arr = np.loadtxt(path, delimiter=delimiter)
    return _infer_txt_layout(arr, n_traces_hint, n_samples_hint)

def write_txt_matrix(path: str, data: np.ndarray, fmt="%.6f"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, data, fmt=fmt)

def read_segy_matrix(path: str):
    """Reads SEG-Y into (nsamp, ntr) float32 matrix. Requires segyio."""
    try:
        import segyio
    except Exception as e:
        raise ImportError("segyio is required for SEG-Y input. Install with: pip install segyio") from e

    with segyio.open(path, "r", ignore_geometry=True) as f:
        f.mmap()
        ntr = f.tracecount
        ns = f.samples.size
        data = np.empty((ns, ntr), dtype=np.float32)
        for i in range(ntr):
            data[:, i] = f.trace[i]
        dt_us = None
        try:
            dt_us = int(f.bin[segyio.BinField.Interval])
        except Exception:
            dt_us = None
    return data, dt_us

def normalize_dt_tmax(dt_ms, tmax_ms, nsamp):
    if dt_ms is None and tmax_ms is None:
        raise ValueError("Provide at least --dt-ms or --tmax-ms.")
    if dt_ms is None:
        dt_ms = tmax_ms / (nsamp - 1)
    if tmax_ms is None:
        tmax_ms = (nsamp - 1) * dt_ms
    return float(dt_ms), float(tmax_ms)

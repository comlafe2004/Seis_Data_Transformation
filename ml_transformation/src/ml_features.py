# ml_features.py
import numpy as np

def rolling_mean_std(x: np.ndarray, win: int):
    if win <= 1:
        return x.copy(), np.zeros_like(x)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    csum = np.cumsum(xp, dtype=np.float64)
    csum2 = np.cumsum(xp*xp, dtype=np.float64)
    m = (csum[win:] - csum[:-win]) / win
    v = (csum2[win:] - csum2[:-win]) / win - m*m
    v = np.maximum(v, 0.0)
    s = np.sqrt(v)
    return m.astype(np.float32), s.astype(np.float32)

def run_length_posrun(x: np.ndarray):
    n = x.size
    pos = (x > 0.0).astype(np.int8)
    run_len = np.zeros(n, dtype=np.int16)
    i = 0
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pos[j] == 1:
            j += 1
        L = j - i
        run_len[i:j] = L
        i = j
    return run_len.astype(np.int16), pos.astype(np.int8)

def build_feature_matrix(trace: np.ndarray, win: int, rmsd_max: float):
    x = trace.astype(np.float32)
    n = x.size

    d1 = np.zeros(n, dtype=np.float32); d1[1:] = x[1:] - x[:-1]
    d2 = np.zeros(n, dtype=np.float32); d2[2:] = x[2:] - 2*x[1:-1] + x[:-2]

    rmean, rstd = rolling_mean_std(x, win)
    run_len, pos_run = run_length_posrun(x)
    amp_le = (x <= float(rmsd_max)).astype(np.int8)

    X = np.column_stack([
        x, np.abs(x), d1, d2, rmean, rstd,
        run_len.astype(np.float32), pos_run.astype(np.float32),
        amp_le.astype(np.float32),
    ]).astype(np.float32)
    return X

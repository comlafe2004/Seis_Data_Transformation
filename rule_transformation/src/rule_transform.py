# rule_transform.py
import numpy as np

def compute_rmsd_positive_global_threshold(section: np.ndarray):
    """Compute RMSD_max across traces using positive samples only."""
    ns, ntr = section.shape
    rmsds = np.zeros(ntr, dtype=np.float64)
    for k in range(ntr):
        x = section[:, k]
        pos = x[x > 0.0]
        if pos.size == 0:
            rmsds[k] = 0.0
            continue
        mu = pos.mean()
        rmsds[k] = np.sqrt(np.mean((pos - mu) ** 2))
    return float(rmsds.max()), rmsds

def find_positive_peaks(trace: np.ndarray, min_peak_amp=1e-12, min_peak_distance=1):
    x = trace
    n = x.size
    peaks = []
    last = -10**9
    for i in range(1, n - 1):
        if x[i] > 0 and x[i] >= x[i-1] and x[i] >= x[i+1] and x[i] >= min_peak_amp:
            if i - last >= min_peak_distance:
                peaks.append(i)
                last = i
    return np.array(peaks, dtype=int)

def adjacent_peak_rule_transform(section: np.ndarray, rmsd_max: float,
                                min_peak_amp=1e-12, min_peak_distance=1):
    """Flip only positive samples <= RMSD_max that lie strictly between adjacent positive peaks."""
    ns, ntr = section.shape
    out = section.copy()
    flipped_mask = np.zeros_like(section, dtype=bool)

    for k in range(ntr):
        tr = out[:, k]
        peaks = find_positive_peaks(tr, min_peak_amp=min_peak_amp, min_peak_distance=min_peak_distance)
        if peaks.size < 2:
            continue
        for i in range(peaks.size - 1):
            a, b = int(peaks[i]), int(peaks[i+1])
            if b <= a + 1:
                continue
            seg = tr[a+1:b]
            if np.all(seg > 0.0):
                idx_local = np.where(seg <= rmsd_max)[0]
                if idx_local.size:
                    idx_global = a + 1 + idx_local
                    tr[idx_global] *= -1.0
                    flipped_mask[idx_global, k] = True
    return out, flipped_mask

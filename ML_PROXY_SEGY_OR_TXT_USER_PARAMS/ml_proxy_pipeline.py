#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML (Proxy) pipeline:
- Supports SEGY or TXT
- Implements corrected adjacent-positive-peak rule for labels
- Trains RF to emulate rule labels
- Applies RF to generate ML(proxy) transformed section
- Produces density, wiggle, and difference plots
"""

import argparse
from pathlib import Path
import numpy as np

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False


# ----------------------------
# IO
# ----------------------------
def load_txt_matrix(path: str, orientation: str = "cols") -> np.ndarray:
    """
    Load numeric matrix from txt. Default assumes traces in columns:
      shape = (n_samples, n_traces)
    If orientation='rows', returns transpose so output is always (samples, traces).
    """
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError(f"TXT must be 2D matrix, got shape {data.shape}")
    if orientation.lower().startswith("row"):
        data = data.T
    return data.astype(np.float32, copy=False)


def load_segy_matrix(path: str) -> np.ndarray:
    """
    Load SEGY into (samples, traces) float32 matrix using segyio.
    """
    try:
        import segyio
    except Exception as e:
        raise RuntimeError("segyio is required for SEG-Y input. Install with: pip install segyio") from e

    with segyio.open(path, "r", ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        n_samples = len(f.samples)
        arr = np.zeros((n_samples, n_traces), dtype=np.float32)
        for i in range(n_traces):
            arr[:, i] = f.trace[i].astype(np.float32, copy=False)
    return arr


# ----------------------------
# RMSD definitions
# ----------------------------
def rmsd_of_positive_samples(trace: np.ndarray) -> float:
    """RMSD computed on positive samples x_i > 0, relative to their (biased) mean."""
    pos = trace[trace > 0]
    if pos.size == 0:
        return 0.0
    mu_b = float(np.mean(pos))
    return float(np.sqrt(np.mean((pos - mu_b) ** 2)))


def compute_global_rmsdmax(section: np.ndarray):
    """
    section: (samples, traces)
    returns (RMSDmax, rmsd_per_trace)
    """
    n_traces = section.shape[1]
    rmsds = np.zeros(n_traces, dtype=np.float32)
    for k in range(n_traces):
        rmsds[k] = rmsd_of_positive_samples(section[:, k])
    return float(np.max(rmsds)), rmsds


# ----------------------------
# Adjacent-positive-peak rule (labels + transformation)
# ----------------------------
def find_positive_peaks(trace: np.ndarray, prom=None, dist=None) -> np.ndarray:
    """
    Positive peaks = local maxima with amplitude > 0.
    """
    kwargs = {}
    if prom is not None:
        kwargs["prominence"] = prom
    if dist is not None and dist > 0:
        kwargs["distance"] = dist

    peaks, _ = find_peaks(trace, **kwargs)
    peaks = peaks[trace[peaks] > 0]
    return peaks.astype(int, copy=False)


def rule_labels_adjacent_peaks(trace: np.ndarray, rmsdmax: float, peak_prom=None, peak_dist=None) -> np.ndarray:
    """
    Boolean mask of samples to flip:
      - consider intervals between adjacent positive peaks
      - if ALL samples in interval are > 0 (positive run)
        and there exists sample(s) with 0 < amp <= rmsdmax,
        then flip ONLY those samples with 0 < amp <= rmsdmax inside that interval.
    """
    n = trace.size
    flip = np.zeros(n, dtype=bool)
    peaks = find_positive_peaks(trace, prom=peak_prom, dist=peak_dist)
    if peaks.size < 2:
        return flip

    for i in range(peaks.size - 1):
        a = int(peaks[i])
        b = int(peaks[i + 1])
        if b <= a + 1:
            continue
        seg = trace[a + 1:b]
        if seg.size == 0:
            continue
        if np.all(seg > 0):
            cond = (seg > 0) & (seg <= rmsdmax)
            if np.any(cond):
                flip[a + 1:b][cond] = True
    return flip


def rule_transform_section(section: np.ndarray, rmsdmax: float, peak_prom=None, peak_dist=None):
    n_samples, n_traces = section.shape
    rule_sec = section.copy()
    flips = np.zeros((n_samples, n_traces), dtype=bool)
    for k in range(n_traces):
        m = rule_labels_adjacent_peaks(section[:, k], rmsdmax, peak_prom, peak_dist)
        flips[:, k] = m
        rule_sec[m, k] *= -1.0
    return rule_sec, flips


# ----------------------------
# Feature engineering (sample-level)
# ----------------------------
def run_length_features(trace: np.ndarray):
    """
    For each sample:
      - run_len: length of the current positive run if in positive run, else 0
      - pos_run: 1 if sample is in positive run, else 0
    """
    n = trace.size
    pos = trace > 0
    run_len = np.zeros(n, dtype=np.int32)
    pos_run = pos.astype(np.int8)

    i = 0
    while i < n:
        if not pos[i]:
            i += 1
            continue
        j = i
        while j < n and pos[j]:
            j += 1
        L = j - i
        run_len[i:j] = L
        i = j
    return run_len, pos_run


def rolling_stats(trace: np.ndarray, win: int):
    if win <= 1:
        return trace.astype(np.float32), np.zeros_like(trace, dtype=np.float32)

    pad = win // 2
    x = np.pad(trace.astype(np.float32), (pad, pad), mode="reflect")
    c1 = np.cumsum(x, dtype=np.float64)
    c2 = np.cumsum(x * x, dtype=np.float64)
    s1 = c1[win:] - c1[:-win]
    s2 = c2[win:] - c2[:-win]
    mean = (s1 / win).astype(np.float32)
    var = (s2 / win - (s1 / win) ** 2)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def build_features(section: np.ndarray, rmsdmax: float, roll_win: int = 11) -> np.ndarray:
    """
    X shape: (n_samples*n_traces, n_features)
    Features:
      amp, abs_amp, d1, d2, run_len, pos_run, roll_mean, roll_std, (amp>0 & amp<=rmsdmax)
    """
    n_samples, n_traces = section.shape
    X_list = []
    for k in range(n_traces):
        tr = section[:, k].astype(np.float32, copy=False)
        amp = tr
        abs_amp = np.abs(tr)
        d1 = np.gradient(tr).astype(np.float32)
        d2 = np.gradient(d1).astype(np.float32)
        run_len, pos_run = run_length_features(tr)
        rmean, rstd = rolling_stats(tr, roll_win)
        ind = ((tr > 0) & (tr <= rmsdmax)).astype(np.float32)

        Xk = np.column_stack([
            amp, abs_amp, d1, d2,
            run_len.astype(np.float32),
            pos_run.astype(np.float32),
            rmean, rstd,
            ind,
        ])
        X_list.append(Xk)
    return np.vstack(X_list)


def flatten_labels(flipmask_section: np.ndarray) -> np.ndarray:
    return flipmask_section.T.reshape(-1).astype(np.int8)


# ----------------------------
# ML proxy: train + apply
# ----------------------------
def train_ml_proxy(X: np.ndarray, y: np.ndarray, n_trees: int = 200, max_depth=None, random_state: int = 7):
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    clf.fit(X, y)
    return clf


def apply_ml_proxy(section: np.ndarray, clf, rmsdmax: float, roll_win: int = 11, prob_thr: float = 0.70,
                   enforce_positive_only: bool = True):
    n_samples, n_traces = section.shape
    X = build_features(section, rmsdmax, roll_win=roll_win)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= prob_thr)

    flipmask = pred.reshape((n_traces, n_samples)).T
    if enforce_positive_only:
        flipmask = flipmask & (section > 0)

    out = section.copy()
    out[flipmask] *= -1.0
    return out, flipmask


# ----------------------------
# Plotting
# ----------------------------
def robust_clip(section: np.ndarray, p: float = 99.0) -> float:
    a = np.abs(section).ravel()
    if a.size == 0:
        return 1.0
    return float(np.percentile(a, p))


def plot_density(section: np.ndarray, dt_ms: float, tmax_ms: float, out_png: str, title: str,
                 clip=None, cmap: str = "seismic"):
    n_samples, n_traces = section.shape
    if tmax_ms is None:
        tmax_ms = dt_ms * (n_samples - 1)
    if clip is None or clip <= 0:
        clip = robust_clip(section, 99.0)

    plt.figure(figsize=(12, 6), dpi=200)
    im = plt.imshow(
        section,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=-clip, vmax=clip,
        extent=[0, n_traces - 1, tmax_ms, 0],
    )
    plt.title(title)
    plt.xlabel("Trace index")
    plt.ylabel("Time (ms)")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("Amplitude (a.u.)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_wiggle_alltraces(section: np.ndarray, dt_ms: float, tmax_ms: float, out_png: str,
                          title: str, scale: float = 1.0, trace_step: int = 1):
    n_samples, n_traces = section.shape
    if tmax_ms is None:
        tmax_ms = dt_ms * (n_samples - 1)

    t = np.linspace(0, tmax_ms, n_samples)
    amax = np.max(np.abs(section))
    if amax <= 0:
        amax = 1.0
    norm = scale / amax

    plt.figure(figsize=(14, 7), dpi=200)
    for k in range(0, n_traces, trace_step):
        tr = section[:, k] * norm
        x = k + tr
        plt.plot(x, t, color="black", linewidth=0.6)
        plt.fill_betweenx(t, k, x, where=(x > k), color="black", alpha=0.25)

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Trace index")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to SEGY or TXT input")
    ap.add_argument("--input-type", choices=["segy", "txt"], required=True)
    ap.add_argument("--txt-orientation", choices=["cols", "rows"], default="cols")
    ap.add_argument("--dt-ms", type=float, default=4.0)
    ap.add_argument("--tmax-ms", type=float, default=None)
    ap.add_argument("--outdir", default="out_mlproxy")
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--cmap", default="seismic")
    ap.add_argument("--peak-prom", type=float, default=None)
    ap.add_argument("--peak-dist", type=int, default=None)
    ap.add_argument("--roll-win", type=int, default=11)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-threshold", type=float, default=0.70)
    ap.add_argument("--trace-step-wiggle", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.input_type == "txt":
        section = load_txt_matrix(args.input, orientation=args.txt_orientation)
    else:
        section = load_segy_matrix(args.input)

    n_samples, n_traces = section.shape
    tmax_ms = args.tmax_ms if args.tmax_ms is not None else args.dt_ms * (n_samples - 1)

    rmsdmax, rmsd_per_trace = compute_global_rmsdmax(section)
    np.savetxt(outdir / "rmsd_per_trace.txt", rmsd_per_trace, fmt="%.6f")
    with open(outdir / "rmsd_summary.csv", "w", encoding="utf-8") as f:
        f.write("n_samples,n_traces,dt_ms,tmax_ms,rmsdmax\n")
        f.write(f"{n_samples},{n_traces},{args.dt_ms},{tmax_ms},{rmsdmax}\n")

    rule_sec, rule_flipmask = rule_transform_section(section, rmsdmax, args.peak_prom, args.peak_dist)

    y = flatten_labels(rule_flipmask)
    X = build_features(section, rmsdmax, roll_win=args.roll_win)
    clf = train_ml_proxy(X, y, n_trees=args.rf_trees, max_depth=args.rf_max_depth)

    if HAVE_JOBLIB:
        joblib.dump(clf, outdir / "model.joblib")

    mlproxy_sec, _ = apply_ml_proxy(section, clf, rmsdmax, roll_win=args.roll_win, prob_thr=args.rf_threshold)

    diff_proxy_minus_orig = mlproxy_sec - section
    diff_proxy_minus_rule = mlproxy_sec - rule_sec

    plot_density(section, args.dt_ms, tmax_ms, str(outdir / "original_density.png"), "Original",
                 clip=args.clip, cmap=args.cmap)
    plot_density(rule_sec, args.dt_ms, tmax_ms, str(outdir / "rule_density.png"), "Rule (Adjacent-Peak) Transformed",
                 clip=args.clip, cmap=args.cmap)
    plot_density(mlproxy_sec, args.dt_ms, tmax_ms, str(outdir / "mlproxy_density.png"), "ML (Proxy) Transformed",
                 clip=args.clip, cmap=args.cmap)

    plot_density(diff_proxy_minus_orig, args.dt_ms, tmax_ms, str(outdir / "diff_mlproxy_minus_original.png"),
                 "Difference: ML (Proxy) - Original", clip=args.clip, cmap=args.cmap)
    plot_density(diff_proxy_minus_rule, args.dt_ms, tmax_ms, str(outdir / "diff_mlproxy_minus_rule.png"),
                 "Difference: ML (Proxy) - Rule", clip=args.clip, cmap=args.cmap)

    plot_wiggle_alltraces(section, args.dt_ms, tmax_ms, str(outdir / "original_wiggle_alltraces.png"),
                          "Original Wiggle (All Traces)", trace_step=max(1, args.trace_step_wiggle))
    plot_wiggle_alltraces(rule_sec, args.dt_ms, tmax_ms, str(outdir / "rule_wiggle_alltraces.png"),
                          "Rule Wiggle (All Traces)", trace_step=max(1, args.trace_step_wiggle))
    plot_wiggle_alltraces(mlproxy_sec, args.dt_ms, tmax_ms, str(outdir / "mlproxy_wiggle_alltraces.png"),
                          "ML (Proxy) Wiggle (All Traces)", trace_step=max(1, args.trace_step_wiggle))

    np.savetxt(outdir / "original.txt", section, fmt="%.6f")
    np.savetxt(outdir / "rule_transformed.txt", rule_sec, fmt="%.6f")
    np.savetxt(outdir / "mlproxy_transformed.txt", mlproxy_sec, fmt="%.6f")

    print("[OK] Done.")
    print(f"[OK] Section shape: samples={n_samples}, traces={n_traces}")
    print(f"[OK] RMSDmax = {rmsdmax:.6f}")
    print(f"[OK] Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

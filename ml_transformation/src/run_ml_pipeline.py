# run_ml_pipeline.py
import argparse
import os
import numpy as np

from utils_io import read_txt_matrix, read_segy_matrix, write_txt_matrix, normalize_dt_tmax
from rule_transform import compute_rmsd_positive_global_threshold, adjacent_peak_rule_transform
from ml_features import build_feature_matrix
from viz import plot_density, plot_side_by_side, plot_difference, plot_wiggle_all_traces, plot_spectra

def main():
    p = argparse.ArgumentParser(description="ML transformation (non-proxy if you supply labels) + visualisations.")
    p.add_argument("--input", required=True)
    p.add_argument("--input-type", choices=["auto","segy","txt"], default="auto")
    p.add_argument("--delimiter", default=None)
    p.add_argument("--ntraces", type=int, default=None)
    p.add_argument("--nsamp", type=int, default=None)
    p.add_argument("--dt-ms", type=float, default=None)
    p.add_argument("--tmax-ms", type=float, default=None)

    p.add_argument("--labels", default=None,
                   help="Optional 0/1 flip labels matrix (nsamp x ntr). If omitted, labels are generated from RULE.")
    p.add_argument("--label-delimiter", default=None)

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--win", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--prob-thr", type=float, default=0.70)

    p.add_argument("--clip", type=float, default=60.0)
    p.add_argument("--cmap", default="seismic")
    p.add_argument("--outdir", default="ml_outputs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    itype = args.input_type
    if itype == "auto":
        itype = "segy" if args.input.lower().endswith((".sgy",".segy")) else "txt"

    if itype == "segy":
        sec, dt_us = read_segy_matrix(args.input)
        if args.dt_ms is None and dt_us is not None and dt_us > 0:
            args.dt_ms = dt_us / 1000.0
    else:
        sec = read_txt_matrix(args.input, delimiter=args.delimiter, n_traces_hint=args.ntraces, n_samples_hint=args.nsamp)

    ns, ntr = sec.shape
    dt_ms, tmax_ms = normalize_dt_tmax(args.dt_ms, args.tmax_ms, ns)

    rmsd_max, _ = compute_rmsd_positive_global_threshold(sec)

    # labels
    if args.labels is not None:
        ymat = read_txt_matrix(args.labels, delimiter=args.label_delimiter, n_traces_hint=ntr, n_samples_hint=ns)
        ymat = (ymat > 0.5).astype(np.uint8)
        label_source = "external_labels"
    else:
        _, ymask = adjacent_peak_rule_transform(sec, rmsd_max, min_peak_amp=1e-12, min_peak_distance=1)
        ymat = ymask.astype(np.uint8)
        label_source = "rule_generated_labels"

    ntrain = int(max(1, min(ntr-1, round(args.train_frac * ntr))))
    train_traces = np.arange(ntrain)

    X_list, y_list = [], []
    for k in train_traces:
        Xk = build_feature_matrix(sec[:, k], win=args.win, rmsd_max=rmsd_max)
        yk = ymat[:, k].astype(np.uint8)
        X_list.append(Xk); y_list.append(yk)
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError("scikit-learn required: pip install scikit-learn") from e

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    ml_sec = sec.copy()
    prob_map = np.zeros_like(sec, dtype=np.float32)
    flip_map = np.zeros_like(sec, dtype=bool)

    for k in range(ntr):
        Xk = build_feature_matrix(sec[:, k], win=args.win, rmsd_max=rmsd_max)
        pk = clf.predict_proba(Xk)[:, 1]
        prob_map[:, k] = pk
        flip = (pk >= args.prob_thr) & (sec[:, k] > 0.0)  # one-way (positive only)
        ml_sec[flip, k] *= -1.0
        flip_map[flip, k] = True

    write_txt_matrix(os.path.join(args.outdir, "ml_transformed.txt"), ml_sec, fmt="%.6f")
    np.savetxt(os.path.join(args.outdir, "ml_prob_map.csv"), prob_map, delimiter=",")
    np.savetxt(os.path.join(args.outdir, "ml_flip_map.csv"), flip_map.astype(np.uint8), delimiter=",")

    plot_density(sec, os.path.join(args.outdir, "original_color.png"),
                 "Original", dt_ms, tmax_ms, clip=args.clip, cmap=args.cmap, colorbar=True)
    plot_density(ml_sec, os.path.join(args.outdir, "ml_color.png"),
                 f"ML transformed (prob≥{args.prob_thr:.2f})", dt_ms, tmax_ms, clip=args.clip, cmap=args.cmap, colorbar=True)

    plot_side_by_side(sec, ml_sec, os.path.join(args.outdir, "compare_orig_ml.png"),
                      ("Original", "ML transformed"), dt_ms, tmax_ms, clip=args.clip, cmap=args.cmap)

    plot_difference(ml_sec - sec, os.path.join(args.outdir, "diff_ml_minus_orig.png"),
                    "Difference: ML - Original", dt_ms, tmax_ms, clip=None, cmap=args.cmap)

    plot_wiggle_all_traces(sec, os.path.join(args.outdir, "wiggle_original_alltraces.png"),
                           dt_ms, tmax_ms, scale=None, title="Original wiggle (all traces)")
    plot_wiggle_all_traces(ml_sec, os.path.join(args.outdir, "wiggle_ml_alltraces.png"),
                           dt_ms, tmax_ms, scale=None, title="ML transformed wiggle (all traces)")

    nyq = 500.0 / dt_ms
    plot_spectra({"Original": sec, "ML": ml_sec},
                 os.path.join(args.outdir, "amplitude_spectra_original_ml.png"),
                 dt_ms, fmax=nyq, title="Amplitude spectra: Original vs ML")

    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"nsamp={ns}\n")
        f.write(f"ntraces={ntr}\n")
        f.write(f"dt_ms={dt_ms}\n")
        f.write(f"tmax_ms={tmax_ms}\n")
        f.write(f"RMSD_max_global={rmsd_max}\n")
        f.write(f"label_source={label_source}\n")
        f.write(f"train_traces={ntrain} (0..{ntrain-1})\n")
        f.write(f"prob_threshold={args.prob_thr}\n")

    try:
        import joblib
        joblib.dump(clf, os.path.join(args.outdir, "rf_model.joblib"))
    except Exception:
        pass

    print("[OK] ML pipeline completed:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()

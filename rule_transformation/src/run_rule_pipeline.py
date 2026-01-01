# run_rule_pipeline.py
import argparse
import os
import numpy as np

from utils_io import read_txt_matrix, read_segy_matrix, write_txt_matrix, normalize_dt_tmax
from rule_transform import compute_rmsd_positive_global_threshold, adjacent_peak_rule_transform
from viz import plot_density, plot_side_by_side, plot_difference, plot_wiggle_all_traces, plot_spectra

def main():
    p = argparse.ArgumentParser(description="RMSD adjacent-peak RULE transformation (SEGY or TXT) + visualisations.")
    p.add_argument("--input", required=True, help="Input path (.sgy/.segy or .txt)")
    p.add_argument("--input-type", choices=["auto","segy","txt"], default="auto")
    p.add_argument("--delimiter", default=None)
    p.add_argument("--ntraces", type=int, default=None)
    p.add_argument("--nsamp", type=int, default=None)
    p.add_argument("--dt-ms", type=float, default=None)
    p.add_argument("--tmax-ms", type=float, default=None)
    p.add_argument("--clip", type=float, default=60.0)
    p.add_argument("--cmap", default="seismic")
    p.add_argument("--min-peak-amp", type=float, default=1e-12)
    p.add_argument("--min-peak-distance", type=int, default=1)
    p.add_argument("--outdir", default="rule_outputs")
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

    rmsd_max, rmsd_per_trace = compute_rmsd_positive_global_threshold(sec)
    rule_sec, rule_mask = adjacent_peak_rule_transform(
        sec, rmsd_max,
        min_peak_amp=args.min_peak_amp,
        min_peak_distance=args.min_peak_distance
    )

    write_txt_matrix(os.path.join(args.outdir, "rule_transformed.txt"), rule_sec, fmt="%.6f")

    plot_density(sec, os.path.join(args.outdir, "original_color.png"),
                 "Original", dt_ms, tmax_ms, clip=args.clip, cmap=args.cmap, colorbar=True)
    plot_density(rule_sec, os.path.join(args.outdir, "rule_color.png"),
                 "Rule transformed (adjacent-peak)", dt_ms, tmax_ms, clip=args.clip, cmap=args.cmap, colorbar=True)

    plot_side_by_side(sec, rule_sec, os.path.join(args.outdir, "compare_orig_rule.png"),
                      ("Original", "Rule transformed"), dt_ms, tmax_ms, clip=args.clip, cmap=args.cmap)

    plot_difference(rule_sec - sec, os.path.join(args.outdir, "diff_rule_minus_orig.png"),
                    "Difference: Rule - Original", dt_ms, tmax_ms, clip=None, cmap=args.cmap)

    plot_wiggle_all_traces(sec, os.path.join(args.outdir, "wiggle_original_alltraces.png"),
                           dt_ms, tmax_ms, scale=None, title="Original wiggle (all traces)")
    plot_wiggle_all_traces(rule_sec, os.path.join(args.outdir, "wiggle_rule_alltraces.png"),
                           dt_ms, tmax_ms, scale=None, title="Rule transformed wiggle (all traces)")

    nyq = 500.0 / dt_ms
    plot_spectra({"Original": sec, "Rule": rule_sec},
                 os.path.join(args.outdir, "amplitude_spectra_original_rule.png"),
                 dt_ms, fmax=nyq, title="Amplitude spectra: Original vs Rule")

    np.savetxt(os.path.join(args.outdir, "rmsd_per_trace.csv"), rmsd_per_trace, delimiter=",", header="RMSD_trace", comments="")
    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"nsamp={ns}\n")
        f.write(f"ntraces={ntr}\n")
        f.write(f"dt_ms={dt_ms}\n")
        f.write(f"tmax_ms={tmax_ms}\n")
        f.write(f"RMSD_max_global={rmsd_max}\n")

    print("[OK] Rule pipeline completed:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()

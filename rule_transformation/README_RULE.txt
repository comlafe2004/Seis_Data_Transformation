RMSD Adjacent-Peak RULE Transformation (SEGY or TXT) + Visualisations

WHAT THIS DOES
1) Reads a seismic section from SEG-Y (.sgy/.segy) OR TXT matrix.
2) Computes a GLOBAL RMSD threshold (RMSD_max) from positive samples only.
3) Applies the amended adjacent-peak rule:
   - find positive peaks (local maxima > 0)
   - for each adjacent peak pair, look strictly between peaks
   - if ALL samples between peaks are > 0 and there exist samples <= RMSD_max,
     flip ONLY those samples <= RMSD_max (peaks are not modified)
4) Writes outputs and generates plots (density, side-by-side, difference, wiggle, spectra).

REQUIREMENTS
- Python 3.9+
- numpy, matplotlib
- segyio (only if using SEG-Y input)

RUN (TXT example; WS70 dt=4ms, tmax=6000ms)
python src/run_rule_pipeline.py --input WS70-001_1__21405.txt --input-type txt --dt-ms 4 --tmax-ms 6000 --outdir rule_out

RUN (SEGY example)
python src/run_rule_pipeline.py --input line.sgy --input-type segy --dt-ms 4 --tmax-ms 6000 --outdir rule_out

NOTES
- Wiggle plots use ALL traces (no subset).
- Density plots use --cmap (default 'seismic') and --clip (default ±60).

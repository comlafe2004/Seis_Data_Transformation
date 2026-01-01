ML Transformation (NOT proxy if you provide your own labels) + Visualisations

WHAT THIS DOES
1) Reads a seismic section from SEG-Y OR TXT.
2) Computes RMSD_max (global) and uses it as a feature (amp <= RMSD_max).
3) Trains a RandomForest model:
   - If you provide --labels (0/1 flip matrix), training is NOT proxy.
   - If you omit --labels, labels are generated from the RULE (proxy-like).
4) Applies the model to flip samples:
   - flip if P(flip) >= prob_thr AND original amplitude is positive (one-way flip)
5) Writes outputs and generates plots (density, side-by-side, difference, wiggle, spectra).

REQUIREMENTS
- Python 3.9+
- numpy, matplotlib, scikit-learn
- segyio (only if using SEG-Y input)
- joblib (optional, for saving rf_model.joblib)

RUN (TXT; non-proxy if labels supplied)
python src/run_ml_pipeline.py --input section.txt --input-type txt --dt-ms 4 --tmax-ms 6000 --labels labels.csv --label-delimiter , --outdir ml_out

RUN (TXT; labels generated from rule)
python src/run_ml_pipeline.py --input WS70-001_1__21405.txt --input-type txt --dt-ms 4 --tmax-ms 6000 --outdir ml_out

RUN (SEGY)
python src/run_ml_pipeline.py --input line.sgy --input-type segy --dt-ms 4 --tmax-ms 6000 --outdir ml_out

KEY PARAMETERS
--prob-thr  (default 0.70)
--win       (default 5)
--n-estimators (default 200)

ML (Proxy) Transformation Pipeline (SEGY or TXT input)
=====================================================

What this package does
----------------------
This is a self-contained Python pipeline that:
1) Loads a 2D seismic section from either:
   - SEG-Y (*.sgy / *.segy)  [requires segyio], or
   - TXT (numeric matrix) where traces are in COLUMNS (samples x traces)
2) Computes a GLOBAL RMSDmax:
   - For each trace k: extract positive samples x_i > 0, compute biased mean mu_b = mean(x_i),
     compute RMSD_k = sqrt( mean( (x_i - mu_b)^2 ) )
   - RMSDmax = max_k RMSD_k
3) Builds "rule labels" using the *adjacent-positive-peak rule*:
   - Find positive peaks (local maxima > 0) in each trace
   - For each adjacent peak pair (p_i, p_{i+1}):
       * Consider the interval between the peaks (exclusive of the peaks)
       * If ALL samples in the interval are > 0 (a positive run)
         and there exists at least one sample with 0 < amp <= RMSDmax,
         then ONLY those samples with 0 < amp <= RMSDmax are flipped (multiplied by -1).
4) Trains a RandomForestClassifier to emulate the rule labels (ML proxy).
5) Applies the trained model to generate the ML(proxy) transformed section.
6) Produces plots (density & wiggle) and difference sections:
   - ML(proxy) - Original   (should be white/blue only if proxy flips only positive samples)
   - ML(proxy) - Rule       (can contain blue/white/red because disagreements are two-sided)

Important note about sign in difference plots
---------------------------------------------
"Proxy - Original" is computed as:  D = A_proxy - A_orig
If the proxy only flips positive samples:
- unchanged: A_proxy = A_orig -> D = 0 (white)
- flipped:   A_proxy = -A_orig where A_orig > 0 -> D = -2*A_orig < 0 (blue)
So you should NOT see red in Proxy-Original unless:
- the difference was computed in the wrong order, OR
- negative samples were flipped, OR
- a non-signed difference/absolute value was used.

Requirements
------------
Python 3.9+
Required:
  numpy, scipy, matplotlib, scikit-learn
Optional (only if reading SEG-Y):
  segyio

Install:
  pip install numpy scipy matplotlib scikit-learn
  pip install segyio   (optional, for SEG-Y input)

How to run
----------
Example (TXT input):
  python ml_proxy_pipeline.py --input WS70-001_1__21405.txt --input-type txt --dt-ms 4 --tmax-ms 6000 --outdir out_mlproxy

Example (SEGY input):
  python ml_proxy_pipeline.py --input R_1.SGY --input-type segy --dt-ms 4 --tmax-ms 6000 --outdir out_mlproxy

User-defined parameters
-----------------------
Key options:
  --dt-ms                 Sampling interval in ms (used for time axis)
  --tmax-ms               Trace length in ms (used for time axis; if omitted, uses n_samples*dt_ms)
  --clip                  Symmetric clip for plots (e.g., 60 means [-60, 60]; if omitted uses robust percentiles)
  --peak-prom             Prominence threshold for peak picking (default auto)
  --peak-dist             Minimum peak distance in samples
  --rf-trees              Number of trees in RandomForest
  --rf-max-depth          Max depth for RandomForest (None = unlimited)
  --rf-threshold          Probability threshold for flipping (default 0.70)
  --roll-win              Rolling window (samples) for rolling mean/std features
  --trace-step-wiggle      1=all traces, 2=every 2nd trace, etc.

If your TXT file has traces in ROWS instead of columns
------------------------------------------------------
Use --txt-orientation rows

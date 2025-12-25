SEGY DIRECT RMSD / ML POLARITY TRANSFORMATION PIPELINE: This converts SEGY to .txt before transformation

Scripts:

01_segy_to_txt.py                 (Convert SEGY to TXT)

02_rule_transformation.py         (Deterministic RMSD rule)

03_ml_transformation.py           (ML training)

04_ml_proxy_transformation.py     (ML proxy application)

05_plotting.py                    (Visualization utilities)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

How to run the plot script for the files in seismic_rmsd_pipeline.zip (2):

python 05_plot_sections.py \
  --orig data.txt \
  --rule rule.txt \
  --ml ml.txt \
  --proxy ml_proxy.txt \
  --dt-ms 4 \
  --outdir plots
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The SEGY_DIRECT_RMSD_ML_PIPELINE provides an alternative root to the transformation that does not involve conversion of SEGY data to .txt format before implementation - 

Requirements: 

numpy

scipy

matplotlib

scikit-learn

segyio

Included in the ZIP is the README.md

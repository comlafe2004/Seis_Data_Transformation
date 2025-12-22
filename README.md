SEGY DIRECT RMSD / ML POLARITY TRANSFORMATION PIPELINE

Scripts:
01_segy_to_txt.py           Convert SEGY to TXT
02_rule_transformation.py   Deterministic RMSD rule
03_ml_transformation.py     ML training
04_ml_proxy_transformation.py ML proxy application
05_plotting.py              Visualization utilities

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
How to run the plot script for the files in seismic_rmsd_pipeline.zip (2):
python 05_plot_sections.py \
  --orig data.txt \
  --rule rule.txt \
  --ml ml.txt \
  --proxy ml_proxy.txt \
  --dt-ms 4 \
  --outdir plots

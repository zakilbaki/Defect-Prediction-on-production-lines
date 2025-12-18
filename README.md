Defect Prediction on Production Lines
This project is based on an industrial machine learning challenge provided by Valeo, a global automotive supplier.
Challenge link:
https://challengedata.ens.fr/participants/challenges/36/
Objective
Predict defective products (KO) on a starter motor production line before the final test bench, using sensor measurements collected during assembly.
Each product goes through several production stations where numerical signals (torque, angle, force, electrical measurements) are recorded.
At the end of the line, a test bench assigns:
0 → OK (passed)
1 → KO (failed)
The goal is to learn a function that maps assembly measurements to the final quality result.
Data
34,515 training samples
8,001 test samples
Highly imbalanced target (~0.9% defects)
Each product is identified by a unique PROC_TRACEINFO code encoding the product reference, production date, and incremental index.
Note:
The raw dataset is not included in this repository.
It can be accessed via the official challenge website (link above).
Approach
Feature engineering based on production process understanding
Explicit handling of missing sensor measurements
Time-based validation to avoid data leakage
Optimization focused on AUROC
Model
Elastic Net Logistic Regression (L1 + L2 regularization)
Temporal Z-score features
Feature selection based on coefficient magnitude
Robust preprocessing (imputation and scaling)
Results
ROC-AUC ≈ 0.72 on time-based validation
Performance above the benchmark AUROC (0.675)

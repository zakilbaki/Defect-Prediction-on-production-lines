# Industrial Defect Prediction

Industrial ML project for predicting defective products on a production line before final quality control.

## Business Problem

Defects are rare but costly in manufacturing. The goal is to detect likely failures earlier in the production process using sensor measurements collected during assembly, so operators can reduce scrap, rework, and late-stage quality losses.

## Dataset

- Valeo industrial challenge dataset
- 34,515 training samples
- 8,001 test samples
- target defect rate around 0.9%
- raw data not included in the repo

Challenge source: [Challenge Data](https://challengedata.ens.fr/participants/challenges/36/)

## Method

- feature engineering from process signals
- explicit treatment of missing measurements
- time-based validation to reduce leakage risk
- scaling and imputation
- sparse linear baseline with Elastic Net logistic regression

## Results

- ROC-AUC around `0.72` on time-based validation
- above benchmark score of `0.675`
- strong baseline for a heavily imbalanced industrial problem

## Current Repository State

This repository currently contains the main notebook used for exploration and modeling. The next improvement step is to split the work into:

- `notebooks/` for exploration
- `src/` for reusable preprocessing and training code
- `reports/` for figures and metrics

## How To Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `main.ipynb` and place the challenge files locally before running.

## Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn


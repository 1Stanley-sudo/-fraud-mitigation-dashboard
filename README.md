# Pre-emptive Fraud Mitigation Prototype

A dashboard that scores Nigerian fintech transactions for fraud risk and applies a
graduated intervention before settlement completes. Built as the technical artefact
for the dissertation "Engineering an AI-Driven Pre-emptive Cybersecurity Prototype
for Predictive Fraud Mitigation in the Nigerian Fintech Sector".

## Running it

```
pip install -r requirements.txt
python -m streamlit run app.py
```

The browser opens at http://localhost:8501. The first launch takes 30 to 60 seconds
while the models are fitted. Later launches take about two seconds, because fitted
models are saved to `models_store/` and reloaded.

Requires Python 3.10 or later.

## Folder layout

```
app/
  app.py                  the dashboard, all five tabs
  requirements.txt        dependencies with minimum versions
  README.md               this file
  src/
    data_pipeline.py      loading, feature engineering, train/test split
    models.py             the three models
    evaluation.py         precision/recall/AUC, financial impact, latency
    explanations.py       plain-language reasons for each alert
    actions.py            risk level to intervention tier
    ingest.py             validation and scoring of uploaded files
    persistence.py        saving and loading fitted models
  scripts/
    generate_holdout.py   builds test files from data never used in training
  data/
    nibss_fraud_dataset.csv   the dataset (1,000,000 transactions)
    paysim_dataset.csv        assessed and rejected; not used, safe to delete
  samples/
    holdout_sample.csv        unseen transactions, labels included
    holdout_blind.csv         unseen transactions, labels removed
  models_store/           fitted models, created on first run
```

## The five tabs

| Tab | What it shows |
| --- | --- |
| Sector Context | Published NIBSS and CBN fraud figures for 2021 to 2025 |
| Data & Pipeline | Dataset summary, sample rows, and which columns are excluded |
| Model Evaluation | Performance of all three models on data withheld from training |
| Live Threat Feed | Flagged transactions with explanations and assigned interventions |
| Test on New Data | Upload a file the models have never seen and watch them score it |

## The three models

**Isolation Forest.** The first label-free attempt. It fails on this data (ROC-AUC
0.535, no fraud detected) because fraud here is not an outlier: it sits at elevated
value inside the normal range. Kept in the tool as a documented negative result.

**Behavioural Deviation Scorer.** The label-free model actually used. Risk is the
one-sided standard score of log transaction value against the training baseline,
optionally weighted by channel. It needs no fraud labels at any point, so a firm
with no fraud history can fit and run it immediately.

**Random Forest.** Supervised benchmark, fitted with labels. Not a deployment
candidate, since the target users have no labels. Its purpose is to measure what
labelled history would be worth: the gap between it and the Deviation Scorer is
0.252 ROC-AUC.

## Preparing a demonstration

Generate test files drawn only from rows excluded from training:

```
python scripts/generate_holdout.py --rows 5000 --trained-on 200000
python scripts/generate_holdout.py --rows 5000 --trained-on 200000 --no-labels
```

Files are written to `samples/`. Upload either one on the Test on New Data tab. The
labelled version lets the tool score its own accuracy; the blind version reproduces
the position of a live system that has no ground truth.

Keep `--trained-on` matched to the sidebar sample size, otherwise the excluded rows
will not line up with what the app actually trained on.

## Notes

All interventions are simulated. The tool prints what would happen but performs no
action against any live system.

Randomness is seeded throughout, so repeated runs under the same settings produce
identical figures.

If you change any model code, use "Clear saved models" in the sidebar. The cache key
covers the sidebar settings but not the source files, so a stale model would
otherwise be reused.

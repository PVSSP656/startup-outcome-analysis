# Startup Outcome Analysis & Prediction

A static portfolio case study presenting an exploratory classification study of 100,000 startup records.

## Website

The site is built with plain HTML, CSS and a small JavaScript file. All visualisations are exported from the saved outputs of the accompanying Jupyter notebook.

## Important notes

- The dataset source, licence, collection process and representativeness could not be verified.
- The data does not establish when predictors were measured relative to outcomes.
- The held-out test partition was reused for diagnostics and permutation importance; it is not a completely untouched final estimate.
- The low IPO recall is reported rather than hidden.
- The raw dataset is intentionally not included in this public website repository.

## Local preview

Open `index.html` in a browser, or run a local static server from the repository root:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## Source notebook

The cleaned notebook belongs in `notebooks/startup_funding_analysis_clean.ipynb` once committed.

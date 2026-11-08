# Startup Outcome Analysis & Prediction

### An exploratory data science study of startup outcomes using statistical analysis, machine learning classification, and an interactive portfolio website

<p align="center">
  <strong>Exploratory Data Analysis · Statistical Testing · Feature Engineering · Machine Learning · Portfolio Website</strong>
</p>

<p align="center">
  <a href="https://github.com/PVSSP656/startup-outcome-analysis">GitHub Repository</a>
  ·
  <a href="https://pvssp656.github.io/startup-outcome-analysis/">Project Website</a>
  ·
  <a href="notebooks/startup_funding_analysis.ipynb">Jupyter Notebook</a>
</p>

---

## Overview

Why do some startups fail while others reach acquisition or IPO?

Startup outcomes are influenced by several interacting factors, including funding history, founder experience, team composition, market conditions, product traction, burn rate, revenue, and investor involvement.

**Startup Outcome Analysis & Prediction** is an end-to-end, exploratory classification study that combines rigorous statistical analysis, leakage-resistant machine learning modeling, and an interactive GitHub Pages portfolio website. The project investigates whether startup characteristics can distinguish between three recorded outcomes:

- **Failure** (55.61%)
- **Acquisition** (42.34%)
- **IPO** (2.05%)

The objective is to thoroughly evaluate data quality, discover meaningful distributional patterns, test statistical associations, engineer domain-justified predictors, benchmark multiple classification models with imbalanced-data metrics, and interpret learned relationships transparently.

> [!NOTE]
> This is an exploratory analytical study. The dataset's real-world provenance and temporal structure could not be independently verified. Conclusions apply strictly to the supplied dataset and validation design; results should not be interpreted as evidence of prospective real-world startup success prediction.

---

## Research Objectives

1. **Assess Data Quality:** Verify completeness, data types, duplicate records, IQR-based outlier flags, numerical validity, and categorical consistency prior to modeling.
2. **Explore Distributions & Group Patterns:** Analyze univariate distributions, categorical frequencies, class imbalance, and cross-outcome differences.
3. **Statistical Hypothesis Testing:** Evaluate outcome differences and associations rigorously using non-parametric tests and effect sizes without assuming causality.
4. **Leakage-Resistant Feature Engineering:** Formulate and validate interpretable domain ratios handled safely inside training-fitted pipelines.
5. **Leakage-Free Preprocessing:** Establish stratified train-test splitting and ensure all imputation and scalers fit solely on training data.
6. **Multi-Model Benchmark:** Compare a baseline DummyClassifier against Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and class-weighted variants.
7. **Address Severe Class Imbalance:** Emphasize macro-averaged F1 and minority-class (IPO) precision, recall, and F1 rather than raw accuracy.
8. **Interpret Model Behavior:** Examine class-specific logistic regression coefficients and permutation feature importance scored by macro F1.
9. **Interactive Portfolio Presentation:** Host an accessible, responsive static website via GitHub Pages displaying exported visualisations and analytical summaries.

---

## Dataset

- **File location:** `data/raw/startup_success_dataset.csv`
- **Observations:** 100,000 records
- **Features:** 11 original columns (7 numerical, 3 categorical, 1 target)
- **Target column:** `outcome`

| Feature | Type | Description |
|---|---|---|
| `funding_rounds` | Numerical (integer) | Number of investment rounds raised |
| `founder_experience_years` | Numerical (float) | Founders' cumulative years of experience |
| `team_size` | Numerical (integer) | Full-time equivalent team members |
| `market_size_billion` | Numerical (float) | Total addressable market size (in billions USD) |
| `product_traction_users` | Numerical (integer) | Active user traction count |
| `burn_rate_million` | Numerical (float) | Monthly capital burn rate (in millions USD) |
| `revenue_million` | Numerical (float) | Annualized revenue (in millions USD) |
| `investor_type` | Categorical (nominal) | Lead investor category (Angel, VC, Private Equity, etc.) |
| `sector` | Categorical (nominal) | Startup industry vertical (Fintech, Healthcare, SaaS, etc.) |
| `founder_background` | Categorical (nominal) | Background category of founding team |
| `outcome` | Target (nominal) | Recorded status: **Failure**, **Acquisition**, **IPO** |

### Target Distribution

| Outcome | Records | Proportion |
|---|---:|---:|
| **Failure** | 55,610 | 55.61% |
| **Acquisition** | 42,335 | 42.34% |
| **IPO** | 2,055 | 2.05% |
| **Total** | **100,000** | **100.00%** |

The minority IPO class represents only ~2.05% of startups in the dataset, creating a substantial class imbalance that makes macro-averaged metrics and class-level recall critical for meaningful evaluation.

---

## Analytical Methodology

The project follows a structured 9-stage data science lifecycle:

```text
01 Data Quality Assessment ──► 02 Exploratory Data Analysis ──► 03 Statistical Testing
                                                                        │
06 Model Evaluation & Tuning ◄── 05 Leakage-Free Preprocessing ◄── 04 Feature Engineering
         │
         ▼
07 Comparative Diagnostics  ──► 08 Model Interpretation     ──► 09 Web & Documentation
```

### Stage 1: Data Quality Assessment
- Zero missing values and zero duplicate rows across all 100,000 rows.
- Data types, numerical ranges, and categorical levels verified.
- Outlier candidates examined using IQR boundaries without discarding rows (preserving raw data fidelity).

### Stage 2: Exploratory Data Analysis
- Right-skewed distributions observed in market size, burn rate, and revenue.
- Strong Pearson correlation (~0.714) identified between `product_traction_users` and `revenue_million`.
- Visual comparisons revealed that IPO records exhibit distinctly higher median traction and revenue, while Failure records cluster at the lower end.

### Stage 3: Statistical Hypothesis Testing
- **Kruskal–Wallis tests** followed by Holm-corrected Mann–Whitney U tests confirmed statistically significant group differences across outcomes for `funding_rounds`, `founder_experience_years`, `team_size`, `product_traction_users`, and `revenue_million`.
- `market_size_billion` and `burn_rate_million` showed no meaningful outcome separation.
- **Chi-square tests of independence** and Cramér's V revealed no practically significant associations between nominal categorical predictors (sector, investor type, founder background) and startup outcome.

### Stage 4: Feature Engineering
Three domain-informed ratio features were engineered and evaluated:
1. `revenue_to_burn_ratio` = `revenue_million / burn_rate_million`
2. `traction_per_employee` = `product_traction_users / team_size`
3. `funding_rounds_per_founder_experience` = `funding_rounds / founder_experience_years`

Undefined values (from zero denominators) were handled systematically via training-fitted median imputation. Arbitrary binning and target encoding were deliberately avoided to prevent overfitting and data leakage.

### Stage 5: Leakage-Resistant Preprocessing
- Stratified 80:20 train-test split (`random_state=42`), preserving class ratios.
- Numerical features: median imputation followed by `StandardScaler`.
- Categorical features: `OneHotEncoder(handle_unknown='ignore')`.
- All transformers were fit exclusively on training data within `sklearn.pipeline.Pipeline`.

### Stages 6–7: Machine Learning Benchmark & Validation
- Evaluated six classification architectures using 5-fold stratified cross-validation on training data.
- Tuned class weights and thresholds to improve minority-class detection without collapsing precision.

---

## Model Benchmark Results

Evaluated on the held-out test set (20,000 observations):

| Model | Accuracy | Macro F1 | Weighted F1 | IPO Precision | IPO Recall | IPO F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** *(Selected)* | **74.52%** | **0.6215** | **0.7408** | **65.68%** | **27.01%** | **0.3828** |
| Random Forest (balanced) | 73.01% | 0.6120 | 0.7290 | 53.12% | 28.95% | 0.3748 |
| Random Forest (default) | 73.56% | 0.5928 | 0.7310 | 67.20% | 20.44% | 0.3134 |
| Decision Tree | 70.46% | 0.5685 | 0.7010 | 44.76% | 22.87% | 0.3027 |
| K-Nearest Neighbors | 69.96% | 0.5553 | 0.6946 | 61.34% | 17.76% | 0.2755 |
| Dummy Classifier (majority) | 55.61% | 0.2382 | 0.3975 | 0.00% | 0.00% | 0.00% |

### Key Observations
- **Logistic Regression** achieved the highest cross-validated and held-out Macro F1 (0.6215) and overall accuracy (74.52%).
- **Balanced Random Forest** achieved the highest IPO recall (28.95%) at the cost of lower precision and overall Macro F1.
- Accuracy alone is misleading: the Dummy Classifier achieves 55.61% accuracy while completely failing to detect any IPO or Acquisition. Macro F1 provides an unweighted balance across all three classes.

---

## Model Interpretation

The primary Logistic Regression model was interpreted using two complementary lenses:

1. **Class-Specific Standardized Coefficients:**
   - Higher revenue and product traction strongly shift log-odds toward Acquisition and IPO relative to Failure.
   - High burn rate without commensurate revenue reduces acquisition probability.
2. **Permutation Feature Importance (Macro F1 decrease on shuffling):**
   - `revenue_million`, `product_traction_users`, and `funding_rounds` ranked as the top predictive signals.
   - Categorical attributes showed minimal permutation drop, corroborating EDA and statistical findings.

> [!IMPORTANT]
> Permutation importance and logistic coefficients reflect associative patterns within this specific dataset and model parameterization; they do not establish causal mechanisms.

---

## Project Visualisations

High-resolution visualizations exported from the analytical pipeline:

| Visualisation | Focus Area | File |
|---|---|---|
| **Target Distribution** | Class imbalance overview | [`outputs/visualizations/target_distribution.png`](outputs/visualizations/target_distribution.png) |
| **Revenue by Outcome** | Revenue differences across classes | [`outputs/visualizations/revenue_by_outcome.png`](outputs/visualizations/revenue_by_outcome.png) |
| **Correlation Heatmap** | Multicollinearity & numerical relations | [`outputs/visualizations/correlation_heatmap.png`](outputs/visualizations/correlation_heatmap.png) |
| **Model Comparison** | Cross-model benchmark metrics | [`outputs/visualizations/model_comparison.png`](outputs/visualizations/model_comparison.png) |
| **Selected Confusion Matrix** | Prediction errors & IPO detection | [`outputs/visualizations/selected_model_confusion_matrix.png`](outputs/visualizations/selected_model_confusion_matrix.png) |
| **Permutation Importance** | Macro F1 feature contribution | [`outputs/visualizations/permutation_importance.png`](outputs/visualizations/permutation_importance.png) |

All visual assets are also interactively presented on the **[GitHub Pages Project Website](https://pvssp656.github.io/startup-outcome-analysis/)**.

---

## Repository Structure

The consolidated repository is organized as follows:

```text
startup-outcome-analysis/
├── .gitignore                      # Git exclusion rules (virtualenvs, raw datasets, cache)
├── .nojekyll                       # GitHub Pages bypass for Jekyll processing
├── index.html                      # Interactive project portfolio website
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
│
├── assets/                         # Website styling, scripts and web visual assets
│   ├── css/
│   │   └── style.css               # Editorial CSS typography and layout
│   ├── js/
│   │   └── main.js                 # Lightweight navigation script
│   └── images/                     # Web-optimized figures (cell-*.webp)
│
├── data/
│   └── raw/
│       ├── .gitkeep                # Directory preservation marker
│       └── startup_success_dataset.csv  (git-ignored: raw 100k-row dataset)
│
├── notebooks/
│   └── startup_funding_analysis.ipynb   # Complete 9-stage Jupyter analysis notebook
│
├── outputs/
│   └── visualizations/             # High-resolution exported figures (.png)
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       ├── permutation_importance.png
│       ├── revenue_by_outcome.png
│       ├── selected_model_confusion_matrix.png
│       └── target_distribution.png
│
└── src/
    └── .gitkeep                    # Modular code directory
```

---

## Technologies & Stack

| Domain | Technology / Library | Role |
|---|---|---|
| **Language** | Python 3.11+ | Core programming runtime |
| **Data Manipulation** | `pandas`, `numpy` | Tabular data processing and numerical arrays |
| **Statistical Testing** | `scipy` | Kruskal–Wallis, Mann–Whitney U, Chi-square, Cramér's V |
| **Machine Learning** | `scikit-learn` | Pipelines, cross-validation, classifiers, metrics |
| **Visualization** | `matplotlib`, `seaborn` | Statistical graphing and diagnostic plots |
| **Notebook Environment** | `jupyter`, `ipykernel` | Interactive exploratory notebook execution |
| **Frontend Website** | HTML5, CSS3, JavaScript | Portfolio presentation |
| **Hosting & CI/CD** | GitHub Pages | Static website deployment from root branch |
| **Version Control** | Git, GitHub | Distributed version control and collaboration |

---

## Reproducing the Analysis Locally

### 1. Clone the repository
```bash
git clone https://github.com/PVSSP656/startup-outcome-analysis.git
cd startup-outcome-analysis
```

### 2. Set up the virtual environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
Place `startup_success_dataset.csv` in `data/raw/` (if reproducing from scratch), then launch:
```bash
jupyter notebook notebooks/startup_funding_analysis.ipynb
```
Run all cells from top to bottom. The notebook will automatically execute data quality checks, statistical tests, pipeline training, evaluation, and export diagnostic plots to `outputs/visualizations/`.

### 5. Preview the Website Locally
From the repository root:
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in your web browser.

---

## Limitations

- **Dataset Provenance:** The origin, collection methodology, and real-world licensing of the dataset were not independently verifiable. It may be synthetic or non-representative of the broader startup ecosystem.
- **Absence of Temporal Markers:** The dataset lacks event timestamps; it is unclear whether variables like revenue and traction were measured strictly prior to the outcome. Potential lookahead or measurement leakage cannot be excluded.
- **Extreme Class Imbalance:** IPO represents only 2.05% of observations. While macro F1 mitigates blind accuracy inflation, IPO recall for the selected model is 27.01%, leaving most IPOs undetected.
- **Holdout Test Set Usage:** The test set was utilized across baseline evaluation, class-weight diagnostics, and permutation importance. While pipeline fits were strictly contained, a pristine outer holdout or nested cross-validation is recommended for future iterations.
- **Associative vs. Causal:** Feature coefficients and permutation importance reflect predictive correlations within this model, not causal levers that founders can manipulate.

---

## Future Directions

- Replicate findings on an independently sourced, time-stamped startup dataset (e.g., Crunchbase, PitchBook).
- Implement temporal out-of-time validation splits to test prospective predictability.
- Explore cost-sensitive classification and probability calibration (Platt scaling / isotonic regression) to optimize decision thresholds for venture capital scenarios.
- Investigate survival analysis approaches (e.g., Cox proportional hazards) where time-to-outcome is observed.

---

## Author

**P. V. S. Subramanya Prasad**  
Integrated M.Sc. Mathematics  
Vellore Institute of Technology (VIT), Vellore  
- **GitHub:** [@PVSSP656](https://github.com/PVSSP656)  
- **Repository:** [PVSSP656/startup-outcome-analysis](https://github.com/PVSSP656/startup-outcome-analysis)  
- **Website:** [pvssp656.github.io/startup-outcome-analysis](https://pvssp656.github.io/startup-outcome-analysis/)

---

<p align="center">
  <sub>Released under the MIT License. Built as an exploratory data science and machine learning research project.</sub>
</p>

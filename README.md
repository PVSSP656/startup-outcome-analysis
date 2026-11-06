# Startup Outcome Analysis & Prediction

### An exploratory data science study of startup outcomes using statistical analysis and machine learning

<p align="center">
  <strong>Exploratory Data Analysis · Statistical Testing · Machine Learning · Model Evaluation</strong>
</p>

<p align="center">
  <a href="https://github.com/PVSSP656/startup-outcome-analysis">GitHub Repository</a>
  ·
  <a href="https://pvssp656.github.io/startup-outcome-analysis/">Project Website</a>
</p>

---

## Overview

Why do some startups fail while others reach acquisition or IPO?

Startup outcomes are influenced by several interacting factors, including funding history, founder experience, team composition, market conditions, product traction, revenue, and investor involvement.

**Startup Outcome Analysis & Prediction** explores these relationships through a structured data science workflow. The project combines exploratory data analysis, statistical hypothesis testing, feature engineering, and machine learning classification to investigate patterns associated with three startup outcomes:

- **Failure**
- **Acquisition**
- **IPO**

The objective is to understand the dataset, identify meaningful patterns, evaluate classification approaches, and examine how effectively different models distinguish between startup outcomes.

This is an exploratory analytical study. The dataset's real-world provenance and temporal structure have not been independently verified, so the results should not be interpreted as evidence of validated future startup prediction.

## Research Objectives

1. Examine the distribution and quality of the available startup data.
2. Investigate relationships between startup characteristics and outcomes.
3. Identify statistically significant differences across outcome groups.
4. Prepare and transform features for machine learning.
5. Establish a baseline and compare multiple classification algorithms.
6. Evaluate performance under substantial class imbalance.
7. Investigate model behaviour and feature importance.
8. Document limitations and opportunities for further research.

## Dataset

The dataset contains **100,000 observations and 11 columns**.

| Feature | Description |
|---|---|
| `funding_rounds` | Number of funding rounds |
| `founder_experience_years` | Founders' years of experience |
| `team_size` | Number of team members |
| `market_size_billion` | Market size in billions |
| `product_traction_users` | Product traction measured by users |
| `burn_rate_million` | Burn rate in millions |
| `revenue_million` | Revenue in millions |
| `investor_type` | Investor category |
| `sector` | Startup industry sector |
| `founder_background` | Founders' background category |
| `outcome` | Target classification |

### Target distribution

The target variable contains three classes:

| Outcome | Observations | Percentage |
|---|---:|---:|
| Failure | 55,610 | 55.61% |
| Acquisition | 42,335 | 42.34% |
| IPO | 2,055 | 2.055% |
| **Total** | **100,000** | **100%** |

A major characteristic of this dataset is the underrepresentation of IPO observations. This class imbalance is important when interpreting model performance.

> **Data provenance:** The dataset's original collection process and real-world provenance have not been independently verified. Findings are therefore interpreted within the scope of the available dataset.

## Analytical Workflow

The analysis is organised into nine stages.

| Stage | Analysis |
|---|---|
| 01 | Data quality assessment |
| 02 | Exploratory data analysis |
| 03 | Statistical hypothesis testing |
| 04 | Feature engineering |
| 05 | Data preprocessing |
| 06 | Baseline machine learning |
| 07 | Model comparison and evaluation |
| 08 | Model interpretation |
| 09 | Final documentation and conclusions |

### 1. Data Quality Assessment

- Dataset dimensions and data types
- Missing-value analysis
- Duplicate-record checks
- Initial statistical summaries

### 2. Exploratory Data Analysis

- Target-class distribution
- Numerical feature distributions
- Categorical feature analysis
- Relationships between features
- Outcome-wise comparisons
- Correlation analysis

### 3. Statistical Analysis

Statistical tests are used to investigate differences and associations between startup characteristics and outcomes.

Methods include:

- Kruskal–Wallis test
- Mann–Whitney U test
- Holm multiple-comparison correction
- Chi-square test of independence
- Cramér's V effect size

### 4. Feature Engineering

Feature preparation and transformation are performed to support the subsequent machine learning workflow.

### 5. Preprocessing

- Training and test data separation
- Stratified splitting
- Numerical and categorical preprocessing
- Pipeline-based transformations

### 6–7. Machine Learning and Evaluation

The project explores several classification approaches:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbours
- Balanced Random Forest

Models are evaluated using metrics appropriate for multiclass classification and imbalanced datasets.

### 8. Model Interpretation

The analysis investigates model behaviour using:

- Logistic Regression coefficients
- Permutation feature importance
- Class-wise performance
- Prediction-level diagnostics

Global model coefficients and feature-importance results are interpreted as model-level information, not as individual causal explanations.

## Model Results

The following results were obtained during the analysis.

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 74.52% |
| Macro F1-score | 0.6215 |
| Weighted F1-score | 0.7408 |
| IPO Precision | 65.68% |
| IPO Recall | 27.01% |

### Balanced Random Forest

| Metric | Result |
|---|---:|
| IPO Recall | 28.95% |
| Macro F1-score | 0.6120 |

### Interpreting the Results

The models achieve reasonable overall classification performance, but the minority IPO class remains difficult to identify.

The results demonstrate why accuracy alone is insufficient for imbalanced classification problems. Macro F1-score and class-specific precision and recall provide additional insight into model performance.

The reported results should be interpreted as findings from this particular dataset and experimental setup, rather than generalisable estimates of real-world startup outcomes.

## Evaluation Considerations

The project uses a stratified 80:20 train-test split and training-only cross-validation for model selection.

There is, however, an important evaluation limitation: the held-out test set was also used in subsequent class-weighted diagnostics and permutation-importance analysis.

Consequently, it should not be described as a completely untouched test set. A future experimental version should reserve a separate final evaluation set or use nested validation to provide a more independent estimate of generalisation performance.

Additionally, the dataset does not establish whether every predictor was available before the recorded startup outcome. This prevents a reliable claim of prospective prediction.

## Visualisations

The project website presents selected visualisations generated during the notebook analysis.

These include exploratory distributions, statistical comparisons, model evaluation, and feature-importance figures.

**[Explore the Project Website](https://pvssp656.github.io/startup-outcome-analysis/)**

## Repository Structure

```text
startup-outcome-analysis/
│
├── index.html
├── README.md
├── LICENSE
├── .nojekyll
│
└── assets/
    ├── css/
    │   └── style.css
    │
    ├── js/
    │   └── main.js
    │
    └── images/
        └── analysis visualisations
```

This repository contains the static portfolio website and its visual assets. The analytical notebook and raw dataset are maintained separately and are not included in this website repository.

## Tools & Technologies

| Area | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistical analysis | SciPy |
| Machine learning | Scikit-learn |
| Development | Jupyter Notebook |
| Version control | Git, GitHub |
| Website | HTML, CSS, JavaScript |
| Deployment | GitHub Pages |

## Limitations

This study has several limitations:

- Dataset provenance has not been independently established.
- The dataset does not provide verified temporal information for establishing predictor availability before outcomes.
- IPO observations are substantially underrepresented.
- The held-out test set was reused for some diagnostic and interpretation steps.
- Statistical associations do not establish causation.
- Results may not generalise beyond the supplied dataset.

These limitations are important when interpreting the findings and designing future experiments.

## Future Improvements

Potential extensions include:

- Independent verification of dataset provenance.
- Time-aware data collection and validation.
- A separate final test set.
- More systematic hyperparameter optimisation.
- Further investigation of minority-class performance.
- Calibration and probability-based evaluation.
- External validation on independently sourced startup data.

## Author

**P. V. S. Subramanya Prasad**

Integrated M.Sc. Mathematics  
Vellore Institute of Technology (VIT), Vellore

Interested in machine learning, statistical modelling, and applied data science.

- GitHub: [@PVSSP656](https://github.com/PVSSP656)
- Project Repository: [startup-outcome-analysis](https://github.com/PVSSP656/startup-outcome-analysis)

---

<p align="center">
  <sub>Built as an exploratory data science and machine learning project.</sub>
</p>
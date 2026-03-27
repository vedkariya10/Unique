# SwiftSole Intelligence Dashboard

A data-driven analytics platform for SwiftSole — India's 15-minute premium sneaker delivery startup.

## Dashboard Pages

| Page | Type | Algorithms |
|------|------|-----------|
| 📊 Descriptive Analysis | Descriptive | Distribution plots, correlation matrix, heatmaps |
| 🔍 Diagnostic Analysis | Diagnostic | Chi-square, ANOVA, feature correlation, funnel |
| 🤖 Predictive Modelling | Predictive | Random Forest, Logistic Regression, XGBoost, K-Means, Apriori, Ridge Regression |
| 🎯 Prescriptive Actions | Prescriptive | Segment action matrix, budget allocation, seasonal calendar |
| 📥 New Data Upload | Prediction Engine | Live scoring pipeline for new respondents |

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push all files (no sub-folders) to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click Deploy

## File Structure

```
app.py                        ← Main entry point
utils.py                      ← Shared preprocessing pipeline
requirements.txt              ← All dependencies (pinned)
swiftsole_dataset.csv         ← 2,100 row synthetic dataset (seed=42)
.streamlit/config.toml        ← Theme and server config
pages/
  __init__.py
  page_descriptive.py
  page_diagnostic.py
  page_predictive.py
  page_prescriptive.py
  page_upload.py
```

## Dataset

- 2,000 clean rows + 100 noise rows
- 57 columns: 32 survey questions + 12 engineered features + 3 meta columns
- 5 persona archetypes: Hype Collector, Aspirational Adopter, Comfort Pragmatist, Gift Buyer, Research Driven
- Target variable: `target_binary` (1 = will buy, 0 = won't buy)
- Random seed: 42 (fully reproducible)

## ML Models

- **Classification**: Random Forest · Logistic Regression · XGBoost — predicts buy/won't buy
- **Clustering**: K-Means (k=2–8, elbow + silhouette) — segments customers into tribes
- **Association Rules**: Apriori (adjustable support/confidence) — finds product co-purchase patterns
- **Regression**: Random Forest Regressor · Ridge — predicts WTP midpoint in ₹

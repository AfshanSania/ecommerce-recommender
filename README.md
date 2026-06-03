# 🛒 AI-Powered E-Commerce Product Recommendation Engine

A hybrid machine learning recommendation system built on real retail transaction data, combining collaborative filtering, content-based filtering, and a GenAI explanation layer — structured for Azure ML deployment.

---

## 📌 Project Overview

This project builds an end-to-end product recommendation engine using the [Online Retail II UCI dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) — real UK-based e-commerce transactions from 2009–2011 with over 500,000 rows.

The system recommends products to customers by learning from purchase history patterns and product similarity, then uses an LLM to generate natural language explanations for each recommendation.

---

## 🏗️ Architecture

```
Raw Transaction Data
        │
        ▼
Data Cleaning & Feature Engineering
  ├── RFM Scoring (Recency, Frequency, Monetary)
  └── Customer Segmentation (Champions, Loyal, At Risk...)
        │
        ▼
   ┌────┴────┐
   │         │
   ▼         ▼
SVD-based    TF-IDF
Collaborative  Content-Based
Filtering      Filtering
   │         │
   └────┬────┘
        │  Weighted Ensemble (CF×0.7 + CB×0.3)
        ▼
  Hybrid Recommender
        │
        ▼
  GenAI Explanation Layer
  (Azure OpenAI / OpenAI)
        │
        ▼
  Top-N Recommendations
  with Natural Language Explanations
```

---

## 🤖 Models

| Model | Approach | Library |
|---|---|---|
| SVD Matrix Factorization | Collaborative Filtering | Scikit-learn `TruncatedSVD` |
| TF-IDF Cosine Similarity | Content-Based Filtering | Scikit-learn `TfidfVectorizer` |
| Neural Collaborative Filtering | GMF + MLP Fusion | PyTorch |
| Hybrid Ensemble | Weighted score combination | NumPy / Pandas |
| Recommendation Explanations | LLM prompt engineering | Azure OpenAI / OpenAI |

---

## 📊 Results

Evaluated using leave-one-out offline evaluation across 300 users:

| Metric | Top-5 | Top-10 | Top-20 |
|---|---|---|---|
| Hit Rate@K | — | — | — |
| Precision@K | — | — | — |
| Recall@K | — | — | — |

> Results populate automatically when you run the notebook on the full dataset.

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn, SciPy
- **Deep Learning** — PyTorch (custom NCF model with embeddings)
- **NLP** — TF-IDF vectorization, cosine similarity
- **GenAI** — Azure OpenAI / OpenAI GPT-4o integration
- **Visualization** — Matplotlib, Seaborn, t-SNE embedding plots
- **Cloud** — Azure ML deployment scaffold (ACI/AKS)

---

## 📁 Project Structure

```
ecommerce-recommender/
│
├── ecommerce_ai_recommender.ipynb   # Main notebook (all 13 sections)
├── requirements.txt                 # Python dependencies
└── README.md
```

### Notebook Sections

| # | Section |
|---|---|
| 1 | Install & Import Dependencies |
| 2 | Load & Explore Data |
| 3 | Data Cleaning & Feature Engineering |
| 4 | EDA Dashboard (6-panel visualization) |
| 5 | Model 1 — SVD Collaborative Filtering |
| 6 | Model 2 — TF-IDF Content-Based Filter |
| 7 | Model 3 — Neural Collaborative Filtering (PyTorch) |
| 8 | Hybrid Recommender (CF + CB Ensemble) |
| 9 | GenAI Explanation Layer |
| 10 | Evaluation — Precision@K, Recall@K, Hit Rate@K |
| 11 | t-SNE Product Embedding Visualization |
| 12 | Azure ML Deployment Notes |
| 13 | Summary & Results |

---

## 🚀 How to Run

### On Kaggle (recommended)
1. Open the notebook on [Kaggle](https://www.kaggle.com)
2. Add the [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) dataset from the right panel
3. Click **Run All**

### Locally
```bash
git clone https://github.com/<your-username>/ecommerce-recommender.git
cd ecommerce-recommender
pip install -r requirements.txt
jupyter notebook ecommerce_ai_recommender.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and place `online_retail_II.csv` in the project root.

### Enable GenAI Explanations (optional)
Set your API key as an environment variable or Kaggle Secret:
```bash
# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_KEY="<your-key>"

# Or standard OpenAI
export OPENAI_API_KEY="<your-key>"
```
Then set `USE_LLM = True` in the GenAI cell.

---

## ☁️ Azure ML Deployment

The notebook includes a full Azure ML deployment scaffold:

- Model registration via `azureml.core`
- Scoring script pattern (`init()` + `run()`)
- Environment setup from `requirements.txt`
- Deployment to ACI (dev) or AKS (production)
- Azure OpenAI integration for the explanation layer

---

## 📦 Dataset

**Online Retail II** — UCI Machine Learning Repository  
Real transactions from a UK-based online retailer, December 2009 to December 2011.

| Field | Description |
|---|---|
| Invoice | Transaction ID |
| StockCode | Product ID |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Date and time of transaction |
| Price | Unit price (GBP) |
| Customer ID | Unique customer identifier |
| Country | Customer country |

---

## 🧠 Skills Demonstrated

- Data wrangling with Pandas and NumPy on 500k+ row datasets
- RFM customer segmentation and behavioral feature engineering
- Matrix factorization with Scikit-learn TruncatedSVD
- Custom PyTorch model (Neural Collaborative Filtering)
- NLP-based product similarity with TF-IDF
- GenAI integration using Azure OpenAI API
- Offline recommender evaluation (Precision@K, Recall@K, Hit Rate)
- Dimensionality reduction and visualization with t-SNE
- Azure ML deployment architecture

---

## 📄 License

MIT License — free to use and adapt.

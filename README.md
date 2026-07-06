# QuantixAI - Intelligent Financial & Stock Prediction Dashboard

![QuantixAI](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Django](https://img.shields.io/badge/Django-4.2-darkgreen)

QuantixAI is a production-grade, full-stack financial web application that combines live market data, advanced machine learning (Deep Learning & Gradient Boosting), sentiment analysis, and a Hybrid RAG (Retrieval-Augmented Generation) AI Assistant to provide unparalleled stock market insights.

---

## 📑 Table of Contents
1. [🚀 Features](#-features)
2. [📁 Project Structure](#-project-structure)
3. [🛠️ Tech Stack](#️-tech-stack)
4. [⚙️ Local Setup](#️-local-setup)
5. [🧠 ML Architecture](#-ml-architecture)
6. [🚀 Deployment (Render.com)](#-deployment-rendercom)
7. [📝 License](#-license)

---

## 🚀 Features

### 📊 Live Financial Data
* **Real-time Price Streams:** Multi-ticker prices fetched live via `yfinance` without refreshing the page.
* **Dynamic Watchlist:** Add and track favorite tickers instantly.

### 🤖 Machine Learning Ensemble
* **Deep Learning:** LSTM, BiLSTM, GRU, CNN-LSTM models train on-the-fly.
* **Gradient Boosting:** XGBoost and LightGBM regression models.
* **Multi-Horizon Targets:** Predicts price action for 5m, 15m, 1h, and 1d time horizons.
* **Live Evaluation:** Dynamically compares models using RMSE, MAE, MAPE, R², and Directional Trend Accuracy.

### 🧠 Hybrid AI Assistant (RAG)
* **Agentic Routing:** Automatically categorizes user intents (General, Financial, Technical, Casual).
* **Vector Store Search:** Utilizes ChromaDB + BM25 Hybrid Search for highly accurate financial context retrieval.

### 📈 Market Sentiment
* **Real-time News Scraping:** Pulls latest ticker news and processes it through the VADER NLP engine.
* **Confidence Scoring:** Outputs precise Bullish/Bearish percentages based on compound NLP scores.

---

## 📁 Project Structure

```text
quantixai/
│
├── stockapp/                   # Main Django Application
│   ├── myapp/                  # Core logic, views, and ML pipelines
│   │   ├── api_views.py        # REST endpoints (Prices, Sentiment, Train)
│   │   ├── data_pipeline.py    # yfinance & YahooQuery data fetching
│   │   ├── ml_models.py        # ML architectures (Keras, XGBoost)
│   │   ├── model_engine.py     # Ensemble aggregation logic
│   │   ├── prediction_engine.py# Multi-horizon price targeting
│   │   ├── sentiment_engine.py # VADER NLP processing
│   │   └── training_engine.py  # Automated model training & scaling
│   │
│   ├── rag/                    # AI Assistant & ChromaDB Logic
│   │   ├── llm.py              # Groq integration & prompts
│   │   ├── pipeline.py         # Intent routing & Hybrid RAG execution
│   │   └── retriever.py        # Semantic + BM25 search
│   │
│   ├── static/                 # Frontend Assets (CSS/JS)
│   │   ├── css/app.css         # Modern Flexbox/Grid UI Styling
│   │   └── js/                 # Asynchronous dashboard scripts
│   │
│   └── Templates/              # HTML Django Templates
│       └── index.html          # Main Dashboard
│
├── requirements.txt            # Python dependencies (gunicorn included)
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

| Category         | Technologies Used                               |
|------------------|-------------------------------------------------|
| **Backend**      | Python 3.8+, Django 4.2                         |
| **AI / NLP**     | LangChain, Groq, ChromaDB, VADER Sentiment      |
| **Machine Learning** | TensorFlow/Keras, Scikit-Learn, XGBoost, LightGBM |
| **Data Sources** | yfinance, YahooQuery                            |
| **Frontend**     | HTML5, CSS3, Vanilla JavaScript                 |
| **Deployment**   | Render.com, Gunicorn                            |

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/quantixai.git
cd quantixai
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the `stockapp` directory:
```ini
# stockapp/.env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_django_secret_key_here
DEBUG=True
```

### 4. Run Migrations & Start Server
```bash
cd stockapp
python manage.py migrate
python manage.py runserver
```
Visit `http://127.0.0.1:8000` in your browser.

---

## 🧠 ML Architecture

The automated prediction pipeline follows a rigorous structure:
1. **Data Ingestion:** Fetches historical daily/minute data via `yfinance`.
2. **Feature Engineering:** Scales features using Min-Max normalization and constructs 60-candle lookback sequences.
3. **Training:** Trains Deep Learning models (LSTM/GRU) and Tree-based models (XGBoost/RF) simultaneously.
4. **Evaluation:** Computes regression metrics (RMSE, MAE, MAPE, R², Directional Accuracy) and saves a local JSON report.
5. **Inference:** A weighted ensemble aggregates predictions from all models to generate unified multi-horizon price targets.

---

## 🚀 Deployment (Render.com)

1. Connect your GitHub repository to Render and create a new **Web Service**.
2. **Build Command:**
   ```bash
   pip install -r requirements.txt && cd stockapp && python manage.py collectstatic --noinput
   ```
3. **Start Command:**
   ```bash
   cd stockapp && gunicorn stockapp.wsgi:application
   ```
4. Add your Environment Variables (`GROQ_API_KEY`, `SECRET_KEY`, `DEBUG=False`) in Render.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

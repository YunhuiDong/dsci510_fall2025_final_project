# Ethereum Weekly Volatility Prediction Using Market,Sentiment, and On-chain Data
---
## ● Introduction
This project focuses on predicting Ethereum weekly volatility in the short term. By combining market data (ETH price from Yahoo Finance), investor sentiment (Fear & Greed Index), and on-chain usage metrics (Ethereum transaction fees) over the past 500 days, we develop models to predict whether Ethereum’s price will increase in the following seven days.
The prediction task is formulated as a binary classification problem, where:
- `1` indicates the price increases in the next 7 days,
- `0` indicates the price does not increase.
Though visualization and statistical analysis, the project aims to identify the most influential factors behind price movements, understand the relationships between sentiment, network activity, and market behavior, and evaluate how effectively machine learning models can capture these patterns.
---
## ● Data Sources
| Dataset | Type | Description | Purpose |
|--------|------|-------------|---------|
| ETH price from Yahoo Finance | API Call | Daily Ethereum price data, including Open, High, Low, Close, and Volume | Capture market behavior |
| Fear & Greed Index from CoinMarketCap | API Call | Daily market sentiment index | Quantify investor sentiment |
| Ethereum transaction fees from DefiLlama | API Call | Daily total Ethereum transaction fees | Represent on-chain activity and network usage |

**Date Range:** Most recent 500 days  
**Final usable weekly samples:** 485 rows
---
## ● Analysis
The following types of analysis were performed in this project:

- **Exploratory Data Analysis (EDA):**
  - Price trend visualization
  - Fear & Greed sentiment cycles
  - On-chain transaction fee fluctuations
  - Distribution of daily returns
  - Correlation heatmap of engineered features

- **Feature Engineering:**
  - Daily return and weekly return labels
  - Moving averages (7-day)
  - Volatility measures
  - Changes in sentiment and transaction fees

- **Modeling:**
  - Logistic Regression for baseline interpretable classification
  - Random Forest for capturing nonlinear feature interactions

- **Model Evaluation:**
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - Feature importance analysis
  - Predicted probability distribution
  - True vs predicted timeline comparison

---
## ● Summary of the Results
- Logistic Regression achieved better overall accuracy but struggled to correctly identify “up” weeks.
- Random Forest captured nonlinear relationships but showed weaker generalization.
- On-chain activity and sentiment indicators ranked among the most important predictive features.
- Weekly ETH price prediction remains highly challenging due to:
  - High volatility
  - Noisy short-term signals
  - Limited dataset size after weekly labeling
---
## ● How to Run

### 1. Install Required Packages

Make sure you have Python 3.8+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```
Required libraries include:

- pandas
- numpy
- requests
- yfinance
- scikit-learn
- python-dotenv
- matplotlib

### 2. Create a .env File for API Keys
Create an account and get an api from this website: https://coinmarketcap.com/charts/fear-and-greed-index/
Create a file named .env in the project src directory:
```bash
touch .env
```
Inside .env, add the following:
```env
GREED_FEAR_INDEX_API_KEY = "your_api_key"
```
### 3. Run the Data Collection, Preprocessing Pipeline, Model Training, and Evaluation
From the root directory, run:

```bash
python main.py
```
This will:

- Download ETH price data from Yahoo Finance.
- Fetch Fear & Greed Index values from CoinMarketCap.
- Fetch Ethereum transaction fees from DefiLlama.
- Merge all datasets by date.
- Save datasets to the `data/` directory.
- Perform feature engineering.
- Generate prediction labels.
- Train Logistic Regression and Random Forest models.
- Output classification reports and confusion matrices.
- Generate feature importance and probability distribution plots.
- Save figures to the `results/` directory.

### 4. View Results

- Clean datasets are stored in `./data/`.
- Visualizations are stored in `./results/`.
- Model performance metrics are printed directly in the terminal.

---
## ● In Addition
### 1. Run Jupyter Notebook
From the root directory, run:

```bash
python results.ipynb
```

### 2. Run Testing
From the root directory, run:

```bash
python tests.ipynb
```
It will test most key functions of this project.

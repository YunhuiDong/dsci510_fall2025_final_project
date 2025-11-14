# Ethereum Weekly Volatility Prediction Using Market,Sentiment, and On-chain Data
This project aims to predict Ethereum weekly volatility in the short term utilizing real-time Ethereum information from the Yahoo Finance API and Ethereum on-chain data from the Etherscan API, together with the Kaggle dataset that include sentiment indicators. Based on the information and dataset, a machine learning classification model will be generated to determine whether the following week’s Ethereum’s closing price will rise or fall. The model will apply preprocessing and data cleaning, feature engineering, and evaluation and visualization. This project illustrates how market sentiment and network usage dynamics work together to influence Ethereum volatility movement.

# Data sources
1. Yahoo Finance API (ETH–USD)

Used to obtain daily ETH market data.
Fields: Open, High, Low, Close, Adj Close, Volume
Format: CSV (downloaded via yfinance)
Used in: weekly volatility, weekly returns, weekly volume

2. Crypto Fear & Greed Index (Kaggle Dataset)

Included as a future extension.
Fields: date, fng_value, classification
Format: CSV
Note: Requires Kaggle API configuration.

# Results 
The pipeline generates several weekly-level metrics:

- **weekly_volatility** – standard deviation of daily returns  
- **weekly_return_mean** – average daily return  
- **weekly_volume_sum** – total volume per week  
- **close_last** – final weekly closing price  

Visualizations stored in `results/` include:
- Histogram of weekly closing prices  
- Scatter plot of weekly volatility vs. closing price  

# Installation
## Kaggle API:
```bash
KAGGLE_CONFIG_DIR="/Users/yunhuidong/Desktop/dsci510_fall2025_final_project/data"
```

## Etherscan API:
```bash
ETHERSCAN_API_KEY="YOUR_KEY"
```

## Python Dependencies
Install with:
```bash
pip install -r requirements.txt
```

Dependencies include:
- yfinance  
- pandas  
- matplotlib  
- requests  
- python-dotenv  
- pytest  

# Running analysis 
## Run main pipeline
From project root:
```bash
python src/main.py
```

Outputs:
- Data saved to `data/`
- Plots saved to `results/`

## Run tests
```bash
python -m pytest src/tests.py
```

This validates that the Yahoo Finance API successfully retrieves ETH data.
from src.config import DATA_DIR, RESULTS_DIR, GREED_URL, FEES_URL
from src.load import get_yahoo_eth_data, get_fear_greed_data, get_daily_fees_data
from src.analyze import eda_plot, model_plot
from src.process import merge_data, clean_data_and_feature_engineering, modeling


if __name__ == "__main__":

    # We will use Yahoo Finance daily ETH data
    get_yahoo_eth_data()

    # We will use Fear & Greed Index data from CoinMarketCap
    get_fear_greed_data(GREED_URL)

    # We will use Daily Fees data from DefiLlama
    get_daily_fees_data(FEES_URL)

    # Merge the above three datasets into one dataset
    merge_data()

    # Do data cleaning and feature engineering
    clean_data_and_feature_engineering()

    # Train the data and get the model
    modeling()

    # Plot and save in the result directory
    eda_plot()
    model_plot()


from src.config import DATA_DIR, RESULTS_DIR, GREED_URL, FEES_URL
from src.load import get_yahoo_eth_data, get_fear_greed_data, get_daily_fees_data
from src.analyze import eda_plot, model_plot
from src.process import merge_data, clean_data_and_feature_engineering, modeling


if __name__ == "__main__":
    # Test get_yahoo_eth_data() function
    try:
        get_yahoo_eth_data()
    except Exception as e:
        print(f"Error with get_yahoo_eth_data function: {e}")
        return None 

    # Test get_fear_greed_data() function
    try:
        get_fear_greed_data(GREED_URL)
    except Exception as e:
        print(f"Error with get_fear_greed_data: {e}")
        return None 

    # Test get_daily_fees_data() function
    try:
        get_daily_fees_data(FEES_URL)
    except Exception as e:
        print(f"Error with get_daily_fees_data: {e}")
        return None 

    # Test merge_data function
    try:
        merge_data()
    except Exception as e:
        print(f"Error with merge_data: {e}")
        return None 

    # Test clean_data_and_feature_engineering function
    try:
        clean_data_and_feature_engineering()
    except Exception as e:
        print(f"Error with clean_data_and_feature_engineering: {e}")
        return None 

    # Test modeling function
    try:
        modeling()
    except Exception as e:
        print(f"Error with modeling: {e}")
        return None 

    # Test eda_plot function
    try:
        eda_plot()
    except Exception as e:
        print(f"Error with eda_plot: {e}")
        return None 

    # Test model_plot_plot function
    try:
        model_plot()
    except Exception as e:
        print(f"Error with model_plot: {e}")
        return None 

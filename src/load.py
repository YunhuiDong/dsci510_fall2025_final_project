import requests
import yfinance as yf
import json
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import os


#1 Get Yahoo Finance daily ETH data
def get_yahoo_eth_data():
    try:
        start_date = (date.today() - timedelta(days=500)).strftime("%Y-%m-%d")
        eth = yf.download("ETH-USD", start=start_date)
        eth.to_json("./data/eth_price.json", orient="records", indent=4)
        print("Yahoo ETH data loaded successfully.")
        return None
    except Exception as e:
        print(f"Error loading daily ETH data from Yahoo Finance: {e}")
        return None



#2 Get Fear & Greed Index data
def get_fear_greed_data(url):
    try:
        greed_api_key = os.getenv("GREED_FEAR_INDEX_API_KEY")
        greed_url = url

        params = {
            "limit": 500
        }

        headers = {
            "X-CMC_PRO_API_KEY": greed_api_key
        }

        response = requests.get(greed_url, params=params, headers=headers)
        data = response.json()
        with open("./data/fear_greed_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("Fear & Greed Index data loaded successfully.")
        return None
    except Exception as e:
        print(f"Error loading fear and greed data from CoinMarketCap: {e}")
        return None


#3 Get Daily Fees data
def get_daily_fees_data(url):
    try:
        fees_url = url

        response = requests.get(fees_url)
        data = response.json()

        with open("./data/eth_fees_raw.json", "w") as f:
            json.dump(data, f, indent=4)


        with open("./data/eth_fees_raw.json", "r") as f:
            data = json.load(f)

        points = data["totalDataChart"]
        df = pd.DataFrame(points, columns=["timestamp", "fee"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("date")
        df_500 = df.iloc[-500:].reset_index(drop=True)

        small = df_500[["timestamp", "fee"]].values.tolist()
        with open("./data/eth_fees_500days.json", "w") as f:
            json.dump(small, f, indent=2)
        print("Daily fees data loaded successfully.")
        return None
    except Exception as e:
        print(f"Error loading daily fees data from DefiLlama: {e}")
        return None    


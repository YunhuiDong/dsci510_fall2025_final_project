import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv
import os

from config import (
    DATA_DIR,
    YAHOO_ETH_TICKER,
    FNG_KAGGLE_SLUG,
    ETHERSCAN_BASE_URL,
    ETHERSCAN_API_ENV,
)



#1 Get Yahoo Finance daily ETH data
def get_yahoo_eth_data(period: str = "3y", interval: str = "1d") -> str:
    print(f"--- Loading ETH data from Yahoo Finance ({YAHOO_ETH_TICKER}) ---")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(YAHOO_ETH_TICKER, period=period, interval=interval)
    if df.empty:
        print("Warning: downloaded DataFrame is empty.")

    csv_path = DATA_DIR / "eth_yahoo.csv"
    df.to_csv(csv_path, index_label="Date")
    print(f"Saved Yahoo Finance data to {csv_path}")
    return str(csv_path)



#2 Get Kaggle Fear & Greed Index data
def get_fng_kaggle_data(extract_dir: Path | None = None) -> str | None:
    print(f"--- Loading Fear & Greed Index from Kaggle: {FNG_KAGGLE_SLUG} ---")

    try:
        import kaggle
    except ImportError:
        print("kaggle package is not installed. Skipping Kaggle download.")
        return None

    if extract_dir is None:
        extract_dir = DATA_DIR / "fng_raw"
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        kaggle.api.dataset_download_files(
            FNG_KAGGLE_SLUG, path=str(extract_dir), unzip=True
        )
    except OSError as e:
        print(f"Kaggle authentication error: {e}")
        print("Please check KAGGLE_CONFIG_DIR and kaggle.json. Skipping Kaggle download.")
        return None
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        return None

    csv_files = [p for p in extract_dir.iterdir() if p.suffix.lower() == ".csv"]
    if not csv_files:
        print("No CSV file found in Kaggle extract directory.")
        return None

    raw_path = csv_files[0]
    print(f"Loading Kaggle CSV: {raw_path}")
    df = pd.read_csv(raw_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "fng_index.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved Fear & Greed data to {out_path}")
    return str(out_path)


#3 Get Etherscan
def get_etherscan_gas_oracle() -> str:
    import requests

    print("--- Loading gas oracle from Etherscan API ---")

    load_dotenv()
    api_key = os.getenv(ETHERSCAN_API_ENV)
    if not api_key:
        raise ValueError("ETHERSCAN_API_KEY not set in .env")

    params = {
        "module": "gastracker",
        "action": "gasoracle",
        "apikey": api_key,
    }

    resp = requests.get(ETHERSCAN_BASE_URL, params=params)
    resp.raise_for_status()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "etherscan_gas_oracle.json"
    with open(json_path, "w") as f:
        f.write(resp.text)

    print(f"Saved Etherscan gas oracle data to {json_path}")
    return str(json_path)

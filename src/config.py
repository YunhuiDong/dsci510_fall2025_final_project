# src/config.py
from pathlib import Path
from dotenv import load_dotenv

# directory setup
CURRENT_DIR = Path(__file__).resolve().parent   
BASE_DIR = CURRENT_DIR.parent                  

# load .env（
env_path = CURRENT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# data/ and results/ path
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


# 1 Yahoo Finance
YAHOO_ETH_TICKER = "ETH-USD"

# 2 Kaggle Crypto Fear & Greed Index
FNG_KAGGLE_SLUG = "liiucbs/crypto-fear-and-greed-index"

# 3 Etherscan
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
ETHERSCAN_API_ENV = "ETHERSCAN_API_KEY" 

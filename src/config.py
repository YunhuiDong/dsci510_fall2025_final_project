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

# data sources configuration
GREED_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
FEES_URL = "https://api.llama.fi/overview/fees/Ethereum?excludeTotalDataChart=false"

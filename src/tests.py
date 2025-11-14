import os
import pandas as pd

from config import DATA_DIR
from load import get_yahoo_eth_data


def test_yahoo_eth_api_download():
    csv_path = get_yahoo_eth_data(period="1y")
    assert os.path.exists(csv_path), "CSV file was not created."

    df = pd.read_csv(csv_path)
    assert not df.empty, "Downloaded DataFrame is empty."
    assert "Close" in df.columns, "Expected 'Close' column in Yahoo Finance data."

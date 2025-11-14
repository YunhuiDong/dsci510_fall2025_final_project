# src/process.py
from pathlib import Path
import pandas as pd
from config import DATA_DIR


def process_eth_daily(csv_filename: str = "eth_yahoo.csv") -> pd.DataFrame:
    """
    Get ETH data and calculate daily_return

    """
    csv_path = DATA_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Did you run get_yahoo_eth_data()?")

    df = pd.read_csv(csv_path)

    # Case no date
    if "Price" in df.columns and "Close" in df.columns and "Date" not in df.columns:
        # Drop first two lines
        df = df.iloc[2:].copy()      
        df = df.rename(columns={"Price": "Date"})
    # Case with date
    elif "Date" in df.columns:
        pass
    else:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})

    if "Close" not in df.columns:
        raise ValueError("Expected 'Close' column in ETH Yahoo Finance data.")

    # Turn date into time
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Turn categorical values into numerical values
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop Date and Close when NA
    df = df.dropna(subset=["Date", "Close"])

    # sort
    df = df.sort_values("Date").set_index("Date")

    # calculate daily return
    df["daily_return"] = df["Close"].pct_change()

    # drop NaN daily_return
    df = df.dropna(subset=["daily_return"])

    print("Processed daily ETH data with 'daily_return'.")
    return df



def build_weekly_dataset(include_fng: bool = True,
                         fng_filename: str = "fng_index.csv") -> pd.DataFrame:

    daily = process_eth_daily()

    weekly = daily.resample("W").agg(
        {
            "Close": "last",
            "daily_return": ["std", "mean"],
            "Volume": "sum",
        }
    )

    weekly.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in weekly.columns.values
    ]

    weekly = weekly.rename(
        columns={
            "Close_last": "close_last",
            "daily_return_std": "weekly_volatility",
            "daily_return_mean": "weekly_return_mean",
            "Volume_sum": "weekly_volume_sum",
        }
    )

    # Create target：if next week higher then 1, else 0
    weekly["target"] = (weekly["close_last"].shift(-1) > weekly["close_last"]).astype(int)
    weekly = weekly.dropna(subset=["target"])

    print("Built weekly ETH dataset with volatility & target.")

    # aggregate fear and greed index
    if include_fng:
        fng_path = DATA_DIR / fng_filename
        if fng_path.exists():
            fng_df = pd.read_csv(fng_path)

            if "date" in fng_df.columns:
                date_col = "date"
            else:
                date_col = fng_df.columns[0] 

            fng_df[date_col] = pd.to_datetime(fng_df[date_col])
            fng_df = fng_df.sort_values(date_col).set_index(date_col)

            value_col = None
            for c in ["fng_value", "value", "fng_value_numeric", "fng_value_label"]:
                if c in fng_df.columns:
                    value_col = c
                    break

            if value_col is not None:
                fng_weekly = fng_df[[value_col]].resample("W").mean()
                fng_weekly = fng_weekly.rename(
                    columns={value_col: "fng_value_weekly"}
                )
                weekly = weekly.join(fng_weekly, how="left")
                print("Merged weekly Fear & Greed feature into ETH weekly dataset.")
            else:
                print("Warning: could not find numeric Fear & Greed column, skipping merge.")
        else:
            print(f"Fear & Greed file {fng_path} not found, skipping merge.")

    return weekly

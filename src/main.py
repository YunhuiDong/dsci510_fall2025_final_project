# src/main.py
from config import DATA_DIR, RESULTS_DIR
from load import get_yahoo_eth_data   # ← 不再导入 get_fng_kaggle_data
from analyze import plot_statistics
from process import build_weekly_dataset


if __name__ == "__main__":
    # Ensure the directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    #1 Get ETH daily data from Yahoo Finance API
    csv_path = get_yahoo_eth_data()
    print("\n" + "=" * 50 + "\n")

    #2 Create weekly-feature dataset
    weekly_df = build_weekly_dataset(include_fng=True)
    print("\nWeekly ETH dataset head:")
    print(weekly_df.head())

    #3 Plot and save in the result directory
    plot_statistics(weekly_df.dropna(), "ETH_Weekly", result_dir=str(RESULTS_DIR))

    print("\n--- ETH data pipeline complete. Check 'data/' and 'results/' ---")

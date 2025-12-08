import yfinance as yf
import json
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

#Merge the three datasets into one dataset
def merge_data():
    try:
        # 1. ETH price
        start_date = (date.today() - timedelta(days=500)).strftime("%Y-%m-%d")
        eth = yf.download("ETH-USD", start=start_date)
        eth_df = eth.copy()

        if isinstance(eth_df.columns, pd.MultiIndex):
            eth_df.columns = [col[0] for col in eth_df.columns]

        # turn DatetimeIndex into a column
        eth_df = eth_df.reset_index()              
        eth_df = eth_df.rename(columns={"Date": "date"})
        eth_df = eth_df[["date", "Open", "High", "Low", "Close", "Volume"]]


        # 2. Fear & Greed
        with open("./data/fear_greed_data.json", "r") as f:
            fg_raw = json.load(f)

        fg_df = pd.DataFrame(fg_raw["data"])
        fg_df["timestamp"] = fg_df["timestamp"].astype(int)
        fg_df["date"] = pd.to_datetime(fg_df["timestamp"], unit="s")
        fg_df = fg_df.sort_values("date")[["date", "value"]]

        # 3. Fees 500 days
        with open("./data/eth_fees_500days.json", "r") as f:
            fees_list = json.load(f)

        fees_df = pd.DataFrame(fees_list, columns=["timestamp", "fee"])
        fees_df["date"] = pd.to_datetime(fees_df["timestamp"], unit="s")
        fees_df = fees_df.sort_values("date")[["date", "fee"]]

        # Find common date range
        start = max(eth_df["date"].min(), fg_df["date"].min(), fees_df["date"].min())
        end   = min(eth_df["date"].max(), fg_df["date"].max(), fees_df["date"].max())

        print("Common date range:", start, "→", end)

        eth_cut  = eth_df[(eth_df["date"] >= start) & (eth_df["date"] <= end)]
        fg_cut   = fg_df[(fg_df["date"] >= start) & (fg_df["date"] <= end)]
        fees_cut = fees_df[(fees_df["date"] >= start) & (fees_df["date"] <= end)]


        # Merge
        merged = (
            eth_cut
            .merge(fg_cut, on="date", how="inner")
            .merge(fees_cut, on="date", how="inner")
        )

        print("Rows after merge:", len(merged))
        print(merged.head(), "\n", merged.tail())

        # Save
        merged.to_json("./data/merged_eth_dataset.json", orient="records", indent=4)

        print("We successfully merged three datasets into one dataset.")
        return None
    except Exception as e:
        print(f"Error merging three datasets: {e}")
        return None

def clean_data_and_feature_engineering():
    # 1. Read merged data
    merged_path = "./data/merged_eth_dataset.json"

    df = pd.read_json(merged_path)
    df = df.sort_values("date").reset_index(drop=True)

    print("Loaded merged dataset:", df.shape)
    print(df.head())

    # 2. Missing value check
    print("\n===== Missing Value Check =====")
    print(df.isna().sum())

    df = df.dropna().reset_index(drop=True)


    # 3. Basic Feature Engineering

    # Daily Return
    df["daily_return"] = df["Close"].pct_change()

    # Price Change
    df["close_change"] = df["Close"].diff()

    # Fear & Greed daily change
    df["fg_change"] = df["value"].diff()

    # Fee daily change
    df["fee_change"] = df["fee"].diff()


    # 4. Rolling Features (7-day)

    WINDOW = 7

    # Moving Averages
    df["ma7_close"] = df["Close"].rolling(WINDOW).mean()
    df["ma7_volume"] = df["Volume"].rolling(WINDOW).mean()
    df["ma7_fee"] = df["fee"].rolling(WINDOW).mean()
    df["ma7_fg"] = df["value"].rolling(WINDOW).mean()

    # Volatility (std of daily returns)
    df["volatility_7d"] = df["daily_return"].rolling(WINDOW).std()

    # Rolling fee std
    df["fee_volatility_7d"] = df["fee"].rolling(WINDOW).std()


    # 5. Create the next 7th day price comparison label
    # future close price (shift -7)
    df["close_next7"] = df["Close"].shift(-7)

    # Binary labels (increase = 1, decrease/no increase = 0)
    df["target_up"] = (df["close_next7"] > df["Close"]).astype(int)

    # Multiple category labels (up / down / unchanged)
    df["target_trend"] = df["close_next7"] - df["Close"]
    df["target_trend"] = df["target_trend"].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )


    # 6. Clean up the last row that cannot be trained (because next7 cannot be calculated).
    df_clean = df.dropna().reset_index(drop=True)

    print("\n===== Final Clean Dataset Shape =====")
    print(df_clean.shape)
    print(df_clean.head(), df_clean.tail())


    # 7. Save as JSON and CSV
    df_clean.to_json("./data/clean_eth_dataset.json", orient="records", indent=4)
    df_clean.to_csv("./data/clean_eth_dataset.csv", index=False)

    print("\nSaved cleaned files:")
    print(" - ./data/clean_eth_dataset.json")
    print(" - ./data/clean_eth_dataset.csv")

def modeling():
    # 1. Read cleaned data
    data_path = "./data/clean_eth_dataset.csv"  
    df = pd.read_csv(data_path, parse_dates=["date"])

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # 2. Choose features and labels

    # Use target_up as the binary label for "whether it will rise in the next 7 days".
    target_col = "target_up"

    # Exclude columns that we don't want to be features
    exclude_cols = ["date", "close_next7", "target_up", "target_trend"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("\nUsed features:")
    for c in feature_cols:
        print(" -", c)

    X = df[feature_cols]
    y = df[target_col]

    print("\nLabel (0 = no increase/decrease, 1 = increase):")
    print(y.value_counts(normalize=True))


    # 3. Divided into Train / Test sections in chronological order
    #    （The first 80% is training, and the last 20% is testing）

    n = len(df)
    split_idx = int(n * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    dates_train = df["date"].iloc[:split_idx]
    dates_test = df["date"].iloc[split_idx:]

    print(f"\nTotal number of samples: {n}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training date range: {dates_train.min()} → {dates_train.max()}")
    print(f"Test date range: {dates_test.min()} → {dates_test.max()}")

    # 4. Model 1：Logistic Regression

    log_reg_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    log_reg_clf.fit(X_train, y_train)
    y_pred_lr = log_reg_clf.predict(X_test)

    print("\n========== Logistic Regression ==========")
    print("Classification report:")
    print(classification_report(y_test, y_pred_lr, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    # 5. Model 2：Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
    )

    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    print("\n========== Random Forest ==========")
    print("Classification report:")
    print(classification_report(y_test, y_pred_rf, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # 6. Save the test set prediction results for plotting/report writing.
    results = df.iloc[split_idx:].copy()
    results = results[["date", "Close", target_col]].reset_index(drop=True)
    results["pred_lr"] = y_pred_lr
    results["pred_rf"] = y_pred_rf

    save_path = "./data/test_predictions_baseline.csv"
    results.to_csv(save_path, index=False)
    print(f"\nThe test set prediction results have been saved to: {save_path}")
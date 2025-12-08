import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def eth_plot():
    # Load cleaned dataset
    df = pd.read_csv("./data/clean_eth_dataset.csv", parse_dates=["date"])

    sns.set(style="whitegrid", font_scale=1.1)

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["Close"], label="Close Price", color="blue")
    plt.title("ETH Price Trend (Last 500 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig("./results/key_price_trend.png")
    plt.show() 
    plt.close()
    print("Saved plot in ./results/")

def fear_plot():
    df = pd.read_csv("./data/clean_eth_dataset.csv", parse_dates=["date"])

    sns.set(style="whitegrid", font_scale=1.1)

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["value"], label="Fear & Greed", color="orange")
    plt.title("Fear & Greed Index Trend")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.tight_layout()
    plt.savefig("./results/key_fear_greed_trend.png")
    plt.show() 
    plt.close()
    print("Saved plot in ./results/")

def tran_plot():
    df = pd.read_csv("./data/clean_eth_dataset.csv", parse_dates=["date"])

    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["fee"], label="Fees", color="green")
    plt.title("Ethereum Fees Trend (On-chain Usage Proxy)")
    plt.xlabel("Date")
    plt.ylabel("Fees")
    plt.tight_layout()
    plt.savefig("./results/key_fees_trend.png")
    plt.show() 
    plt.close()
    print("Saved plot in ./results/")

def heat_plot():
    df = pd.read_csv("./data/clean_eth_dataset.csv", parse_dates=["date"])

    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(15, 12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap of Features")
    plt.tight_layout()
    plt.savefig("./results/key_corr_heatmap.png")
    plt.show() 
    plt.close()
    print("Saved plot in ./results/")

def daily_plot():
    df = pd.read_csv("./data/clean_eth_dataset.csv", parse_dates=["date"])

    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["daily_return"], bins=40, kde=True, color="purple")
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.savefig("./results/key_return_histogram.png")
    plt.show() 
    plt.close()
    print("Saved plot in ./results/")


def eda_plot():
    # Load cleaned dataset
    df = pd.read_csv("./data/clean_eth_dataset.csv", parse_dates=["date"])

    sns.set(style="whitegrid", font_scale=1.1)

    # 1. ETH Price Trend
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["Close"], label="Close Price", color="blue")
    plt.title("ETH Price Trend (Last 500 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig("./results/key_price_trend.png")
    plt.show() 
    plt.close()

    # 2. Fear & Greed Trend
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["value"], label="Fear & Greed", color="orange")
    plt.title("Fear & Greed Index Trend")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.tight_layout()
    plt.savefig("./results/key_fear_greed_trend.png")
    plt.show() 
    plt.close()

    # 3. Ethereum Fees Trend
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["fee"], label="Fees", color="green")
    plt.title("Ethereum Fees Trend (On-chain Usage Proxy)")
    plt.xlabel("Date")
    plt.ylabel("Fees")
    plt.tight_layout()
    plt.savefig("./results/key_fees_trend.png")
    plt.show() 
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(15, 12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap of Features")
    plt.tight_layout()
    plt.savefig("./results/key_corr_heatmap.png")
    plt.show() 
    plt.close()

    # 5. Daily Return Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["daily_return"], bins=40, kde=True, color="purple")
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.savefig("./results/key_return_histogram.png")
    plt.show() 
    plt.close()

    print("Saved 5 key plots in ./results/")

def model_plot():
    # 1. Read cleaned data
    data_path = "./data/clean_eth_dataset.csv"
    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("Total rows:", len(df))
    print("Date range:", df["date"].min(), "→", df["date"].max())

    # 2. Features and labels
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "value",                 # Fear & Greed index
        "fee",                   
        "daily_return",
        "close_change",
        "fg_change",
        "fee_change",
        "ma7_close",
        "ma7_volume",
        "ma7_fee",
        "ma7_fg",
        "volatility_7d",
        "fee_volatility_7d",
    ]

    target_col = "target_up"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 3. Divided into Train / Test sections in chronological order

    n = len(df)
    train_size = int(n * 0.8)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    dates_train = df["date"].iloc[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    dates_test = df["date"].iloc[train_size:]

    print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)
    print("Train date range:", dates_train.min(), "→", dates_train.max())
    print("Test  date range:", dates_test.min(), "→", dates_test.max())

    # 4. Logistic Regression & Random Forest
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Prediction
    y_pred_log = log_reg.predict(X_test)
    y_proba_log = log_reg.predict_proba(X_test)[:, 1] 

    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    print("\n========== Logistic Regression ==========")
    print(classification_report(y_test, y_pred_log))

    print("========== Random Forest ==========")
    print(classification_report(y_test, y_pred_rf))

    best_model_name = "logistic"   
    y_pred = y_pred_log if best_model_name == "logistic" else y_pred_rf
    y_proba = y_proba_log if best_model_name == "logistic" else y_proba_rf

    fig_dir = "./results"
    os.makedirs(fig_dir, exist_ok=True)

    # 5. Confusion Matrix

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, f"confusion_matrix_{best_model_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("Saved:", cm_path)

    # 6. Feature Importance

    # Logistic Regression 
    coef_log = np.abs(log_reg.coef_[0]) 
    feat_imp_log = pd.Series(coef_log, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(7, 6))
    plt.barh(feat_imp_log.index, feat_imp_log.values)
    plt.xlabel("Importance (|coefficient|)")
    plt.title("Feature Importance - Logistic Regression")
    plt.tight_layout()
    fi_log_path = os.path.join(fig_dir, "feature_importance_logistic.png")
    plt.savefig(fi_log_path, dpi=150)
    plt.show() 
    plt.close()
    print("Saved:", fi_log_path)

    # Random Forest 
    imp_rf = rf.feature_importances_
    feat_imp_rf = pd.Series(imp_rf, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(7, 6))
    plt.barh(feat_imp_rf.index, feat_imp_rf.values)
    plt.xlabel("Importance")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    fi_rf_path = os.path.join(fig_dir, "feature_importance_rf.png")
    plt.savefig(fi_rf_path, dpi=150)
    plt.show() 
    plt.close()
    print("Saved:", fi_rf_path)

    # 7. Prediction vs Real
    test_result = pd.DataFrame({
        "date": dates_test.values,
        "true": y_test.values,
        "pred": y_pred,
        "proba": y_proba,
    }).reset_index(drop=True)

    plt.figure(figsize=(10, 4))
    plt.plot(test_result["date"], test_result["true"], marker="o", linestyle="-", label="True (target_up)")
    plt.plot(test_result["date"], test_result["pred"], marker="x", linestyle="--", label="Predicted")
    plt.yticks([0, 1])
    plt.xlabel("Date")
    plt.ylabel("Up or Not")
    plt.title(f"True vs Predicted target_up ({best_model_name})")
    plt.legend()
    plt.tight_layout()
    tvp_path = os.path.join(fig_dir, f"true_vs_pred_{best_model_name}.png")
    plt.savefig(tvp_path, dpi=150)
    plt.show() 
    plt.close()
    print("Saved:", tvp_path)

    # 8. Probability Histogram（Probability distribution of prediction "up"）
    plt.figure(figsize=(6, 4))
    plt.hist(y_proba, bins=20)
    plt.xlabel("Predicted probability of Up (class=1)")
    plt.ylabel("Count")
    plt.title(f"Predicted Probability Distribution ({best_model_name})")
    plt.tight_layout()
    prob_hist_path = os.path.join(fig_dir, f"proba_hist_{best_model_name}.png")
    plt.savefig(prob_hist_path, dpi=150)
    plt.show() 
    plt.close()
    print("Saved:", prob_hist_path)

    print("\nAll model analysis figures saved in:", fig_dir)


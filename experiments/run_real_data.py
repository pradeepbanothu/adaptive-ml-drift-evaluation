import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.classifiers import get_models

#SETTINGS
np.random.seed(42)
os.makedirs("results", exist_ok=True)

PLOT_DPI      = 500
ROLLING_WIN   = 75
SMOOTH_WIN    = 20

MODEL_COLOURS = {
    "Random Forest":       "#2ca02c",
    "Logistic Regression": "#1f77b4",
    "Decision Tree":       "#ff7f0e",
}


# HELPERS
def rolling_accuracy(y_true, y_pred, window=ROLLING_WIN):
    scores = []
    for i in range(len(y_pred)):
        start = max(0, i - window + 1)
        score = accuracy_score(
            y_true[start:i + 1],
            y_pred[start:i + 1]
        )
        scores.append(score)
    return scores


# MAIN
def main():

    # LOAD DATA
    data = pd.read_csv("data/processed/creditcard.csv")

    fraud  = data[data["Class"] == 1]
    normal = data[data["Class"] == 0].sample(
        len(fraud) * 5, random_state=42
    )

    data = (
        pd.concat([fraud, normal])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # FEATURES + TARGET
    X = data.drop("Class", axis=1).values
    y = data["Class"].values

    #  Split FIRST, then scale — no data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit on train only
    X_test  = scaler.transform(X_test)         # transform test

    # TRAIN MODELS + COLLECT PREDICTIONS + METRICS   
    models = get_models()

    results          = []
    prediction_store = {}

    for name, model in models.items():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        prediction_store[name] = preds

        acc = accuracy_score(y_test, preds)
        pre = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1  = f1_score(y_test, preds, zero_division=0)

        results.append({
            "Model":     name,
            "Accuracy":  round(acc, 4),
            "Precision": round(pre, 4),
            "Recall":    round(rec, 4),
            "F1 Score":  round(f1,  4),
        })

    # SAVE METRICS + PLOTS + PRINT SUMMARY TABLE 
    df = (
        pd.DataFrame(results)
        .sort_values("F1 Score", ascending=False)
        .reset_index(drop=True)
    )
    df["Rank"] = df.index + 1
    df.to_csv("results/real_data_metrics.csv", index=False)

    print("\nReal Data Results:")
    print(df.to_string(index=False))
    print(f"\nBest Model: {df.iloc[0]['Model']}  (F1={df.iloc[0]['F1 Score']})")

    # GRAPH 1 — Precision, Recall, F1  
    x     = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, df["Precision"], width, color="#1f77b4", label="Precision")
    bars2 = ax.bar(x,         df["Recall"],    width, color="#ff7f0e", label="Recall")
    bars3 = ax.bar(x + width, df["F1 Score"],  width, color="#2ca02c", label="F1 Score")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.60, 1.02)
    ax.set_title(
        "Real Dataset Performance Comparison",
        fontsize=15, fontweight="bold"
    )
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    plt.savefig("results/real_data_performance.png", dpi=PLOT_DPI)
    plt.close()

    # GRAPH 2 — F1 Score Ranking    
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = [MODEL_COLOURS.get(m, "#999") for m in df["Model"]]
    bars = ax.bar(df["Model"], df["F1 Score"], color=colours)

    ax.set_ylim(0.60, 1.02)
    ax.set_title(
        "F1 Score Ranking — Credit Card Fraud Dataset",
        fontsize=15, fontweight="bold"
    )
    ax.set_ylabel("F1 Score")
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig("results/real_data_f1_ranking.png", dpi=PLOT_DPI)
    plt.close()

    # GRAPH 3 — Rolling Accuracy over Simulated Stream
    fig, ax = plt.subplots(figsize=(13, 6))

    for name in df["Model"]:

        preds        = prediction_store[name]
        roll_scores  = rolling_accuracy(y_test, preds, window=ROLLING_WIN)
        smooth_scores = (
            pd.Series(roll_scores)
            .rolling(SMOOTH_WIN, min_periods=1, center=True)
            .mean()
        )

        ax.plot(
            smooth_scores,
            linewidth=2.5,
            label=name,
            color=MODEL_COLOURS.get(name, None)
        )

    ax.set_title(
        "Rolling Window Accuracy on Real Dataset (Simulated Stream)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.60, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/real_data_stream_plot.png", dpi=PLOT_DPI)
    plt.close()

    # SUMMARY
    print("\n Saved:")
    print("  results/real_data_metrics.csv")
    print("  results/real_data_performance.png")
    print("  results/real_data_f1_ranking.png")
    print("  results/real_data_stream_plot.png")


if __name__ == "__main__":
    main()
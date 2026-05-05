import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data.data_generator import SyntheticDriftGenerator
from src.models.classifiers import get_models
from src.drift_detection.ddm import DDM


# SETTINGS
np.random.seed(42)
os.makedirs("results", exist_ok=True)

DRIFT_TYPES  = ["none", "sudden", "gradual"]
N_SAMPLES    = 1000
DRIFT_POINT  = 500
WINDOW_SIZE  = 75
ROLLING_WIN  = 80
SMOOTH_WIN   = 20
PLOT_DPI     = 500

# Consistent colours for dissertation
COLOURS = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree":       "#ff7f0e",
    "Random Forest":       "#2ca02c",
}

# HELPERS
def smooth(values, window=SMOOTH_WIN):
    """
    Centred rolling mean — same length as input, no x-axis shift.
    """
    return (
        pd.Series(values)
        .rolling(window, min_periods=1, center=True)
        .mean()
        .values
    )


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


# SINGLE EXPERIMENT
def run_experiment(drift_type):

    generator = SyntheticDriftGenerator(
        n_samples=N_SAMPLES,
        drift_point=DRIFT_POINT,
        drift_type=drift_type,
        random_state=42
    )
    X, y = generator.generate()

    X_train = X[:DRIFT_POINT]
    y_train = y[:DRIFT_POINT]

    base_models = get_models()

    rows    = {"drift_type": drift_type}
    history = {}

    for model_name, base_model in base_models.items():

        static_model   = copy.deepcopy(base_model)
        adaptive_model = copy.deepcopy(base_model)

        static_model.fit(X_train, y_train)
        adaptive_model.fit(X_train, y_train)

        ddm = DDM()

        static_preds   = []
        adaptive_preds = []

        retrain_count     = 0
        drift_detected_at = None

    
        # STREAM LOOP
    
        for i in range(len(X)):

            xi = X[i].reshape(1, -1)
            yi = y[i]

            sp = static_model.predict(xi)[0]
            ap = adaptive_model.predict(xi)[0]

            static_preds.append(sp)
            adaptive_preds.append(ap)

            error = 0 if ap == yi else 1
            warning, drift = ddm.update(error)

            if drift_type != "none" and drift and i > DRIFT_POINT:

                retrain_count += 1

                if drift_detected_at is None:
                    drift_detected_at = i

                start = max(0, i - WINDOW_SIZE)
                adaptive_model.fit(X[start:i + 1], y[start:i + 1])
                ddm.reset()

    
        # SCORES
        static_scores   = rolling_accuracy(y, static_preds)
        adaptive_scores = rolling_accuracy(y, adaptive_preds)

        if drift_type == "none":
            retrain_count     = 0
            drift_detected_at = None

        history[model_name] = {
            "Static":   static_scores,
            "Adaptive": adaptive_scores,
            "Detected": drift_detected_at,
        }

    
        # METRICS
        rows[f"{model_name}_static_acc"]   = round(static_scores[-1],   4)
        rows[f"{model_name}_adaptive_acc"] = round(adaptive_scores[-1],  4)
        rows[f"{model_name}_improvement"]  = round(
            adaptive_scores[-1] - static_scores[-1], 4
        )

        rows[f"{model_name}_precision"] = round(
            precision_score(y, adaptive_preds, zero_division=0), 4
        )
        rows[f"{model_name}_recall"] = round(
            recall_score(y, adaptive_preds, zero_division=0), 4
        )
        rows[f"{model_name}_f1"] = round(
            f1_score(y, adaptive_preds, zero_division=0), 4
        )

        rows[f"{model_name}_retrain_count"] = retrain_count
        rows[f"{model_name}_drift_delay"]   = (
            None if drift_detected_at is None
            else drift_detected_at - DRIFT_POINT
        )

    plot_results(drift_type, history)
    return rows

# PLOT
def plot_results(drift_type, history):

    title_map = {
        "none":    "No Drift",
        "sudden":  "Sudden Drift",
        "gradual": "Gradual Drift",
    }

    model_names = list(history.keys())
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for idx, model_name in enumerate(model_names):

        ax = axes[idx]

        static_vals   = smooth(history[model_name]["Static"])
        adaptive_vals = smooth(history[model_name]["Adaptive"])
        detected      = history[model_name]["Detected"]
        colour        = COLOURS.get(model_name, "steelblue")

        ax.plot(
            static_vals,
            linestyle="--",
            linewidth=2.3,
            color=colour,
            alpha=0.6,
            label="Static Model"
        )

        ax.plot(
            adaptive_vals,
            linewidth=2.5,
            color=colour,
            label="Adaptive Model (DDM)"
        )

        # True drift / reference line
        if drift_type == "none":
            ax.axvline(
                x=DRIFT_POINT,
                color="grey",
                linestyle="--",
                linewidth=1.5,
                label="Reference Point"
            )
        else:
            ax.axvline(
                x=DRIFT_POINT,
                color="red",
                linestyle="--",
                linewidth=2,
                label="True Drift"
            )

        # DDM detection marker
        if drift_type != "none" and detected is not None:
            ax.axvline(
                x=detected,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"DDM Detected (delay={detected - DRIFT_POINT})"
            )

        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    axes[-1].set_xlabel("Time Step")

    fig.suptitle(
        f"{title_map[drift_type]} Scenario: Static vs Adaptive Model Performance",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"results/{drift_type}_comparison.png", dpi=PLOT_DPI)
    plt.close()



# MAIN

def main():

    all_rows = []

    for drift in DRIFT_TYPES:
        print(f"\nRunning {drift} drift experiment...")
        row = run_experiment(drift)
        all_rows.append(row)
        print(f"  Completed.")

    df = pd.DataFrame(all_rows)
    df.to_csv("results/final_metrics.csv", index=False)

    print("\n All experiments completed.")
    print("Saved: results/final_metrics.csv")
    print("Saved: results/none_comparison.png")
    print("Saved: results/sudden_comparison.png")
    print("Saved: results/gradual_comparison.png")


if __name__ == "__main__":
    main()
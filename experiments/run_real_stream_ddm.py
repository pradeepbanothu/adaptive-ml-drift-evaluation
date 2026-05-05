import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.models.classifiers import get_models
from src.drift_detection.ddm import DDM

# SETTINGS
np.random.seed(42)
os.makedirs("results", exist_ok=True)

DRIFT_POINT    = 1000
WINDOW_SIZE    = 75
ROLLING_WIN    = 80
PLOT_DPI       = 500

# Drift injection parameters (realistic, not destructive)
LABEL_FLIP_RATE = 0.20   # 20% label flip post-drift
FEATURE_NOISE   = 0.80   # Gaussian noise std added to features post-drift

MODEL_COLOURS = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree":       "#ff7f0e",
    "Random Forest":       "#2ca02c",
}


# HELPERS

def rolling_accuracy(y_true, y_pred, window=ROLLING_WIN):
    scores = []
    for i in range(len(y_pred)):
        start = max(0, i - window + 1)
        score = accuracy_score(y_true[start:i + 1], y_pred[start:i + 1])
        scores.append(score)
    return scores


# DATA LOADING + PREPARATION    
def load_data():
    """
    Load creditcard.csv and create a balanced 2000-sample stream.
    400 fraud + 1600 normal = 20 % fraud rate.
    First DRIFT_POINT samples used for initial training;
    remainder form the post-drift evaluation stream.
    """
    data = pd.read_csv("data/processed/creditcard.csv")

    fraud  = data[data["Class"] == 1].sample(400,  random_state=42)
    normal = data[data["Class"] == 0].sample(1600, random_state=42)

    data = (
        pd.concat([fraud, normal])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    X = data.drop("Class", axis=1).values.copy()
    y = data["Class"].values.copy()

    #  Scale train portion only — no leakage into post-drift stream
    scaler = StandardScaler()
    X[:DRIFT_POINT]  = scaler.fit_transform(X[:DRIFT_POINT])
    X[DRIFT_POINT:]  = scaler.transform(X[DRIFT_POINT:])

    return X, y


# Drift Injection — moderate concept + covariate drift  
def inject_drift(X, y):
    """
    Inject moderate concept + covariate drift after DRIFT_POINT.
    - 20% label flip  (concept drift)
    - Feature Gaussian noise std=0.8  (covariate drift)
    """
    rng = np.random.RandomState(42)
    X   = X.copy()
    y   = y.copy()

    for i in range(DRIFT_POINT, len(X)):
        if rng.rand() < LABEL_FLIP_RATE:
            y[i] = 1 - y[i]
        X[i] += rng.normal(0, FEATURE_NOISE, size=X[i].shape)

    return X, y


# MAIN     
def main():

    X_raw, y_raw = load_data()
    X, y         = inject_drift(X_raw, y_raw)

    models  = get_models()
    results = {}
    rows    = []

    for name, base_model in models.items():

        print(f"\nRunning {name}...")

        #  Proper deep copies — no shared state
        static_model   = copy.deepcopy(base_model)
        adaptive_model = copy.deepcopy(base_model)

        X_train = X[:DRIFT_POINT]
        y_train = y[:DRIFT_POINT]

        static_model.fit(X_train, y_train)
        adaptive_model.fit(X_train, y_train)

        ddm = DDM()

        static_preds   = []
        adaptive_preds = []

        drift_detected_at = None
        retrain_count     = 0

        # STREAM LOOP  
        for i in range(len(X)):

            xi = X[i].reshape(1, -1)
            yi = y[i]

            sp = static_model.predict(xi)[0]
            ap = adaptive_model.predict(xi)[0]

            static_preds.append(sp)
            adaptive_preds.append(ap)

            error = 0 if ap == yi else 1
            _, drift = ddm.update(error)

            if drift and i > DRIFT_POINT:

                retrain_count += 1

                if drift_detected_at is None:
                    drift_detected_at = i

                start    = max(0, i - WINDOW_SIZE)
                window_X = X[start:i + 1]
                window_y = y[start:i + 1]

                if len(np.unique(window_y)) > 1:
                    adaptive_model.fit(window_X, window_y)

                ddm.reset()

        # METRICS  
        static_acc   = rolling_accuracy(y, static_preds)
        adaptive_acc = rolling_accuracy(y, adaptive_preds)
        improvement  = adaptive_acc[-1] - static_acc[-1]

        results[name] = {
            "static":   static_acc,
            "adaptive": adaptive_acc,
            "detected": drift_detected_at,
        }

        rows.append({
            "Model":            name,
            "Static Accuracy":  round(static_acc[-1],   4),
            "Adaptive Accuracy":round(adaptive_acc[-1],  4),
            "Improvement":      round(improvement,        4),
            "Detection Point":  drift_detected_at,
            "Detection Delay":  (
                None if drift_detected_at is None
                else drift_detected_at - DRIFT_POINT
            ),
            "Retrain Count":    retrain_count,
        })

        print(f"  Static Acc:   {static_acc[-1]:.4f}")
        print(f"  Adaptive Acc: {adaptive_acc[-1]:.4f}")
        print(f"  Improvement:  {improvement:+.4f}")
        print(f"  Detection:    step {drift_detected_at}  (delay={drift_detected_at - DRIFT_POINT if drift_detected_at else 'N/A'})")

    # Save metrics to CSV + print summary table 
    df = pd.DataFrame(rows)
    df.to_csv("results/real_ddm_metrics.csv", index=False)

    print("\nFinal Metrics:")
    print(df.to_string(index=False))

    # Plot results — rolling accuracy + detection points 
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for idx, name in enumerate(results.keys()):

        ax     = axes[idx]
        colour = MODEL_COLOURS.get(name, "steelblue")

        static_vals   = results[name]["static"]
        adaptive_vals = results[name]["adaptive"]
        detected      = results[name]["detected"]

        ax.plot(
            static_vals,
            linestyle="--",
            linewidth=2,
            color=colour,
            alpha=0.55,
            marker="o",
            markevery=80,
            label="Static Model"
        )

        ax.plot(
            adaptive_vals,
            linewidth=2.5,
            color=colour,
            label="Adaptive Model (DDM)"
        )

        ax.axvline(
            DRIFT_POINT,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Drift Point"
        )

        if detected is not None:
            ax.axvline(
                detected,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"DDM Detection (delay={detected - DRIFT_POINT})"
            )

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Rolling Accuracy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

    axes[-1].set_xlabel("Time Step")

    fig.suptitle(
        "Real Data Drift Adaptation: Static vs Adaptive Models (DDM)",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/real_ddm_subplot.png", dpi=PLOT_DPI)
    plt.close()

    print("\n Saved:")
    print("  results/real_ddm_metrics.csv")
    print("  results/real_ddm_subplot.png")


if __name__ == "__main__":
    main()
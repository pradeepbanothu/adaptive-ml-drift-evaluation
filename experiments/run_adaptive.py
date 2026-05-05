# experiments/run_adaptive.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.data.data_generator import SyntheticDriftGenerator
from src.models.adaptive_model import AdaptiveModel

# =====================================================
# SETTINGS
# =====================================================
np.random.seed(42)
os.makedirs("results", exist_ok=True)

DRIFT_POINT  = 500
N_SAMPLES    = 1000
WINDOW_SIZE  = 100
ROLLING_WIN  = 80
PLOT_DPI     = 500

# =====================================================
# ROLLING ACCURACY
# =====================================================
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


# =====================================================
# MAIN
# =====================================================
def main():

    # ----- Generate data -----
    generator = SyntheticDriftGenerator(
        n_samples=N_SAMPLES,
        drift_point=DRIFT_POINT,
        drift_type="sudden",
        random_state=42
    )
    X, y = generator.generate()

    # ----- Build model -----
    model = AdaptiveModel(window_size=WINDOW_SIZE)

    # ----- Initial training on pre-drift data -----
    model.train(X[:DRIFT_POINT], y[:DRIFT_POINT])

    # ----- Stream loop -----
    predictions = []
    drift_events = []

    for i in range(len(X)):
        pred, warning, drift = model.update(X[i], y[i])
        predictions.append(pred)
        if drift:
            drift_events.append(i)

    # ----- Metrics -----
    rolling_scores = rolling_accuracy(y, predictions)

    final_acc = accuracy_score(y[DRIFT_POINT:], predictions[DRIFT_POINT:])
    print(f"\nPost-drift accuracy : {final_acc:.4f}")
    print(f"Drift events detected at: {drift_events}")

    # ----- Plot -----
    plt.figure(figsize=(12, 5))

    plt.plot(
        rolling_scores,
        linewidth=2.5,
        color="darkorange",
        label="Adaptive Model (DDM)"
    )

    plt.axvline(
        x=DRIFT_POINT,
        color="red",
        linestyle="--",
        linewidth=2,
        label="True Drift Point"
    )

    for ev in drift_events:
        plt.axvline(
            x=ev,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="DDM Detection" if ev == drift_events[0] else ""
        )

    plt.xlabel("Time Step")
    plt.ylabel("Rolling Accuracy")
    plt.title(
        "Adaptive Model Performance Under Sudden Concept Drift",
        fontsize=14,
        fontweight="bold"
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig("results/adaptive_performance.png", dpi=PLOT_DPI)
    plt.close()

    print("Saved: results/adaptive_performance.png")


if __name__ == "__main__":
    main()
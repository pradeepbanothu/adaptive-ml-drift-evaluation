# src/models/adaptive_model.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from src.drift_detection.ddm import DDM


class AdaptiveModel:
    """
    A DDM-driven adaptive classifier wrapper.

    Wraps any scikit-learn classifier and monitors the live prediction
    error stream using DDM.  When drift is confirmed the model retrains
    on a recent window of labelled samples.

    Parameters
    ----------
    base_model  : sklearn estimator  (default: LogisticRegression)
    window_size : int  — number of recent samples kept for retraining
    min_samples : int  — DDM min_samples before detection activates

    Public API
    ----------
    train(X, y)              — initial batch fit
    predict(X)               — return class predictions
    retrain(X, y)            — force a retrain on supplied data
    update(x, y_true)        — online step: predict → feed DDM → maybe retrain
                               returns (prediction, warning, drift)
    """

    def __init__(self, base_model=None, window_size=100, min_samples=50):

        if base_model is None:
            base_model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )

        self.model       = clone(base_model)
        self.window_size = window_size
        self.ddm         = DDM(min_samples=min_samples)
        self.trained     = False

        self._buf_X = []
        self._buf_y = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def train(self, X, y):
        """Initial batch training."""
        X, y = np.array(X), np.array(y)
        if len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X):
        """Return predictions. Returns zeros if not yet trained."""
        if not self.trained:
            return np.zeros(len(X), dtype=int)
        return self.model.predict(np.array(X))

    def retrain(self, X, y):
        """Force retrain on the supplied data window."""
        X, y = np.array(X), np.array(y)
        if len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.trained = True

    def update(self, x, y_true):
        """
        Online single-sample update.

        1. Predict on x
        2. Feed error to DDM
        3. Store (x, y_true) in rolling buffer
        4. Retrain if drift confirmed

        Returns
        -------
        prediction : int
        warning    : bool
        drift      : bool
        """
        x = np.array(x).reshape(1, -1)

        prediction = int(self.predict(x)[0])

        error = 0 if prediction == y_true else 1
        warning, drift = self.ddm.update(error)

        # Rolling buffer — cap to window_size
        self._buf_X.append(x.flatten())
        self._buf_y.append(y_true)

        if len(self._buf_X) > self.window_size:
            self._buf_X = self._buf_X[-self.window_size:]
            self._buf_y = self._buf_y[-self.window_size:]

        if drift:
            self.retrain(self._buf_X, self._buf_y)
            self.ddm.reset()

        return prediction, warning, drift

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"AdaptiveModel(trained={self.trained}, "
            f"buffer={len(self._buf_X)}, "
            f"drifts={self.ddm.drift_count})"
        )
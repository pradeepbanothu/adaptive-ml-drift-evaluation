# src/data/data_generator.py

import numpy as np


class SyntheticDriftGenerator:
    """
    Generates synthetic two-feature binary classification streams
    with controlled concept drift.

    Parameters

    n_samples       : int   — total stream length
    drift_point     : int   — step at which drift begins
    drift_type      : str   — 'none' | 'sudden' | 'gradual'
    transition_width: int   — samples over which gradual drift transitions
    random_state    : int   — seed for reproducibility

    Returns (from generate())

    X : ndarray, shape (n_samples, 2)
    y : ndarray, shape (n_samples,)
    """

    def __init__(
        self,
        n_samples=1000,
        drift_point=500,
        drift_type="sudden",
        transition_width=250,
        random_state=42
    ):
        self.n_samples        = n_samples
        self.drift_point      = drift_point
        self.drift_type       = drift_type
        self.transition_width = transition_width
        self.random_state     = random_state

    def generate(self):

        rng = np.random.RandomState(self.random_state)

        X = np.zeros((self.n_samples, 2))
        y = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_samples):

            # Generate features with drift in distribution after drift_point 
            if self.drift_type == "none" or i < self.drift_point:
                x1 = rng.normal(0, 1)
                x2 = rng.normal(0, 1)
            else:
                x1 = rng.normal(0.3, 1.1)
                x2 = rng.normal(-0.2, 1.1)

            # Generate labels with drift in decision boundary after drift_point 
            if self.drift_type == "none":
                label = 1 if x1 + x2 > 0 else 0

            # SUDDEN DRIFT — abrupt change in decision boundary at drift_point
            elif self.drift_type == "sudden":
                if i < self.drift_point:
                    label = 1 if x1 + x2 > 0 else 0
                else:
                    label = 1 if 0.8 * x1 - 1.2 * x2 > 0 else 0

            # GRADUAL DRIFT — smooth transition in decision boundary over transition_width
            elif self.drift_type == "gradual":
                if i < self.drift_point:
                    label = 1 if x1 + x2 > 0 else 0
                else:
                    transition = min(
                        1.0,
                        (i - self.drift_point) / self.transition_width
                    )
                    if rng.rand() < transition:
                        label = 1 if 0.8 * x1 - 1.2 * x2 > 0 else 0
                    else:
                        label = 1 if x1 + x2 > 0 else 0

            else:
                raise ValueError(
                    f"drift_type must be 'none', 'sudden', or 'gradual'. "
                    f"Got: '{self.drift_type}'"
                )

            X[i] = [x1, x2]
            y[i] = label

        return X, y
import math

class DDM:
    """
    Drift Detection Method (DDM) — Gama et al. (2004)

    Monitors online prediction error rate using a Bernoulli model.
    Signals a warning when error rate rises above 2.5 standard
    deviations from the minimum observed, and confirms drift at 3.0.

    Parameters
    
    min_samples : int
     Minimum number of samples before drift detection activates.

    Attributes
    
    drift_count   : int  — total confirmed drift signals emitted
    warning_count : int  — total warning signals emitted
    """

    def __init__(self, min_samples=50):
        self.min_samples = min_samples
        self.reset()

    def update(self, error):
        """
        Update detector with a new prediction result.

        Parameters
    
        error : int
            0 = correct prediction, 1 = incorrect prediction

        Returns
    
        warning : bool — error rate approaching drift threshold
        drift   : bool — drift confirmed
        """

        self.n += 1

        # online update of error rate p
        self.p = self.p + (error - self.p) / self.n

        # Guard: degenerate cases where s = 0
        if self.p == 0.0 or self.p == 1.0:
            return False, False

        # Standard deviation of Bernoulli error stream
        self.s = math.sqrt(self.p * (1 - self.p) / self.n)

        # Wait until enough observations
        if self.n < self.min_samples:
            return False, False

        # Initialise best historical state
        if self.p_min is None:
            self.p_min = self.p
            self.s_min = self.s

        # Update minimum error boundary
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        warning = False
        drift   = False

        # Warning zone  : p + s > p_min + 2.5 * s_min
        if self.p + self.s > self.p_min + 2.5 * self.s_min:
            warning = True
            self.warning_count += 1

        # Drift confirmed: p + s > p_min + 3.0 * s_min
        if self.p + self.s > self.p_min + 3.0 * self.s_min:
            drift = True
            self.drift_count += 1

        return warning, drift

    def reset(self):
        """Reset detector state after confirmed drift."""
        self.n     = 0
        self.p     = 0.0
        self.s     = 0.0
        self.p_min = None
        self.s_min = None

        self.drift_count   = 0
        self.warning_count = 0

    def __repr__(self):
        return (
            f"DDM(n={self.n}, p={self.p:.4f}, "
            f"p_min={self.p_min}, drifts={self.drift_count})"
        )
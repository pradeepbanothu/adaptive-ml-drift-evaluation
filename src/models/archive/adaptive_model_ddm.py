import numpy as np
from sklearn.linear_model import LogisticRegression
from src.drift_detection.ddm import DDM
 
 
class AdaptiveModelDDM:
    def __init__(self):
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0
        )
 
        self.detector = DDM()
 
        self.buffer_X = []
        self.buffer_y = []
 
        self.trained = False
        self.last_drift_t = -1000
 
        # expose drift to main loop (no double update)
        self.last_detected_drift = None
 
    def update(self, x, y_true, t=None):
        x_input = x

        # WARM-UP PHASE (collect data until we can train a reasonable model)
        if not self.trained:
            self.buffer_X.append(x.flatten())
            self.buffer_y.append(y_true)

            if len(self.buffer_y) > 80 and len(set(self.buffer_y)) > 1:
                X_train = np.array(self.buffer_X)
                y_train = np.array(self.buffer_y)

                self.model.fit(X_train, y_train)
                self.trained = True

            #  NEW: smarter warm-up prediction
            if len(self.buffer_y) > 50:
                X_temp = np.array(self.buffer_X)
                y_temp = np.array(self.buffer_y)

                temp_model = LogisticRegression(
                    class_weight='balanced',
                    max_iter=200
                )
                temp_model.fit(X_temp, y_temp)

                return temp_model.predict(x_input)[0]

            return 0

       # PREDICTION
        y_pred = self.model.predict(x_input)[0]

       # DDM UPDATE
        error = 0 if y_pred == y_true else 1
        warning, drift = self.detector.update(error)

        # expose drift event
        if drift:
            self.last_detected_drift = t

        # ======================
        # 4. STORE
        # ======================
        self.buffer_X.append(x.flatten())
        self.buffer_y.append(y_true)

        # ======================
        # 5. DRIFT HANDLING
        # ======================
        if drift and len(self.buffer_y) > 120 and (t - self.last_drift_t > 150):
            print(f" Drift detected at timestep {t}")

            self.last_drift_t = t

            #  reset DDM after drift (critical)
            self.detector.reset()

            # recent window
            X_train = np.array(self.buffer_X[-200:])
            y_train = np.array(self.buffer_y[-200:])

            # skip bad batches
            if len(set(y_train)) < 2 or np.sum(y_train) < 10:
                return y_pred

            # balance
            cls0 = np.where(y_train == 0)[0]
            cls1 = np.where(y_train == 1)[0]

            if len(cls1) > 0:
                n_pos = len(cls1)
                idx0 = np.random.choice(
                    cls0,
                    size=min(len(cls0), n_pos * 3),
                    replace=False
                )
                idx = np.concatenate([idx0, cls1])
                X_train = X_train[idx]
                y_train = y_train[idx]

            # retrain (no full reset)
            self.model.fit(X_train, y_train)

            # keep recent memory
            self.buffer_X = self.buffer_X[-300:]
            self.buffer_y = self.buffer_y[-300:]

        return y_pred
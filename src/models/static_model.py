import numpy as np
from sklearn.linear_model import LogisticRegression
 
class StaticModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
 
    def fit(self, X, y):
        self.model.fit(X, y)
 
    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        return self.model.predict(x)[0]
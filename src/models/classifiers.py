# src/models/classifiers.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_models():
    """
    Return the three classifiers used throughout all experiments.
    All models use balanced class weights and fixed random state
    for reproducibility. Tested with scikit-learn==1.4.2.
    """

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    }


def get_model_descriptions():
    """
    Return plain-text descriptions for dissertation tables.
    """

    return {
        "Logistic Regression": (
            "Linear classifier, L2 regularisation, "
            "balanced class weights, lbfgs solver"
        ),
        "Decision Tree": (
            "Non-linear CART classifier, max depth 8, "
            "balanced class weights"
        ),
        "Random Forest": (
            "100-tree ensemble, max depth 8, "
            "balanced class weights, all cores"
        ),
    }
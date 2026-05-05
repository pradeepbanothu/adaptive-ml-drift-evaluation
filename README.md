# Adaptive Machine Learning Under Concept Drift

This project came out of a question that bothered me for a while — what actually happens to a machine learning model after it gets deployed? It gets trained, it performs well, it goes live. But the world keeps changing. Fraud patterns shift. User behaviour evolves. And the model just sits there, frozen, slowly becoming less accurate without anyone noticing until the damage is done.

That problem is called concept drift. This project is my attempt to study it properly and test whether a simple, lightweight detection method can automatically catch drift and trigger retraining before performance falls too far.

# What This Project Does

Three classifiers are compared — Logistic Regression, Decision Tree and Random Forest — in two versions each. One version is static, trained once and never updated. The other is adaptive, monitored by the Drift Detection Method (DDM) which watches the live prediction error stream and triggers retraining when it detects that something has changed.

Six experiments were run in total. Three used synthetic data where drift type, timing and severity were controlled precisely. Three used a real credit card fraud dataset from Kaggle with injected distributional shift.

# The Headline Result

Under gradual drift, Logistic Regression went from an accuracy of 0.46 to 0.96 after DDM-triggered retraining. That is a gain of 0.50 from a 75-sample retraining window. In stable conditions, DDM produced zero false alarms across all three classifiers. Under sudden drift, detection happened in as little as 1 step.

# Project Structure

adaptive-ml-drift-evaluation/
│
├── src/
│   ├── drift_detection/
│   │   └── ddm.py                  # DDM built from scratch - Gama et al. 2004
│   ├── models/
│   │   ├── classifiers.py          # LR, DT, RF configurations
│   │   └── adaptive_model.py       # DDM wrapper around any classifier
│   └── data/
│       └── data_generator.py       # Synthetic stream generator
│
├── experiments/
│   ├── run_adaptive.py             # Single model demo
│   ├── run_comparison.py           # All three drift scenarios
│   ├── run_real_data.py            # Static baseline on fraud data
│   └── run_real_stream_ddm.py      # DDM streaming on fraud data
│
├── results/                        # All plots and CSVs saved here
├── data/processed/                 # Put creditcard.csv here
└── requirements.txt

# Setup

```bash
# Clone
git clone https://github.com/YOURUSERNAME/adaptive-ml-drift-evaluation.git
cd adaptive-ml-drift-evaluation

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac or Linux

# Install
pip install -r requirements.txt
```

# Dataset

Download the credit card fraud dataset from Kaggle and place it at `data/processed/creditcard.csv`

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset has 284,807 transactions with 492 fraud cases. All features are PCA anonymised.

## Running the Experiments

Run all from the project root directory.

```bash
python experiments/run_adaptive.py
python -m experiments.run_comparison
python -m experiments.run_real_data
python -m experiments.run_real_stream_ddm
```

Results save automatically to the results folder.

# How DDM Works

DDM watches every prediction the model makes. Correct gets a 0. Wrong gets a 1. It tracks a running error rate and compares it against the best error rate the model has ever achieved. When the current error climbs significantly above that historical best, drift is confirmed and the model retrains on the most recent 75 samples. Then the detector resets and starts building a new baseline.

The warning fires at 2.5 standard deviations above the minimum. Drift is confirmed at 3.0. A minimum of 50 samples must be seen before detection activates, to avoid early false alarms.


# Results Summary

| Scenario | Classifier | Static | Adaptive | Change | DDM Delay |

| No Drift | All three | ~1.00 | ~1.00 | 0.00 | Zero alarms |
| Sudden | Logistic Regression | 0.54 | 0.51 | -0.025 | 1 step |
| Sudden | Decision Tree | 0.56 | 0.49 | -0.075 | 9 steps |
| Sudden | Random Forest | 0.55 | 0.51 | -0.038 | 9 steps |
| Gradual | Logistic Regression | 0.46 | 0.96 | +0.500 | 33 steps |
| Gradual | Decision Tree | 0.45 | 0.64 | +0.188 | 36 steps |
| Gradual | Random Forest | 0.45 | 0.69 | +0.238 | 38 steps |
| Real Data | Logistic Regression | 0.69 | 0.74 | +0.050 | 44 steps |
| Real Data | Decision Tree | 0.71 | 0.54 | -0.175 | 10 steps |
| Real Data | Random Forest | 0.73 | 0.76 | +0.038 | 16 steps |


# Dependencies

numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-learn==1.4.2
scipy==1.13.1

## Reference

Gama, J., Medas, P., Castillo, G. and Rodrigues, P. (2004) Learning with drift detection. Advances in Artificial Intelligence, SBIA 2004. Lecture Notes in Computer Science, vol. 3171. Springer, Berlin, pp. 286-295.
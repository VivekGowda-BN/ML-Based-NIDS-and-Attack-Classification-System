"""
nids — ML-Based Network Intrusion Detection System (Mode 1).

Package layout
--------------
config          Global paths and hyperparameter constants
dataset         Raw-data loading and validation
cleaning        Data-quality fixes (nulls, duplicates, types)
preprocessing   Encoding, scaling, train/test split
features        Feature selection and engineering
train           Model training (binary + multiclass)
evaluate        Metrics, confusion matrices, ROC curves
predict         Single-record and batch inference
replay          Offline scenario replay loop
explain         SHAP-based explainability
schemas         Pydantic data models / validators
"""

__version__ = "0.1.0"
__author__ = "Vivek Gowda B N"

# Models Directory

Trained model artefacts are stored here at runtime.
They are excluded from git (see `.gitignore`).

## Naming Convention

| File | Description |
|---|---|
| `binary_clf.joblib` | Binary classifier (Normal vs Attack) |
| `multi_clf.joblib` | Multiclass classifier (attack category) |
| `scaler.joblib` | Fitted `StandardScaler` used during training |
| `label_encoder.joblib` | Fitted `LabelEncoder` for attack categories |

## Reproducing Models

Run `make train` after completing `make data-prep` and `make preprocess`.

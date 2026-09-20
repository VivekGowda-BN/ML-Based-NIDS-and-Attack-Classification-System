# Data Directory

## Structure

| Folder | Contents |
|---|---|
| `raw/` | Original, unmodified UNSW-NB15 CSV files |
| `interim/` | Cleaned but not yet feature-engineered data |
| `processed/` | Final train / test splits ready for modelling |
| `scenarios/` | Hand-crafted replay scenario CSVs |

## Downloading UNSW-NB15

1. Go to <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
2. Request access and download all four CSV parts:
   - `UNSW-NB15_1.csv`
   - `UNSW-NB15_2.csv`
   - `UNSW-NB15_3.csv`
   - `UNSW-NB15_4.csv`
3. Also download the **features list** (`NUSW-NB15_features.csv`)
   and the **ground truth** file (`UNSW-NB15_GT.csv`).
4. Place all files in `data/raw/`.

## Important

**Data files are excluded from git** (see `.gitignore`).
Never commit raw network traffic data or personally identifiable information.

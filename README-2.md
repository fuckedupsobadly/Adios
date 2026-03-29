# Sentence Length Predictor (Kazakhstan Cases)

A machine learning tool that predicts criminal sentence lengths based on case features, finds similar historical cases using KNN, and highlights potential sentencing bias.

---

## Features

- **Sentence prediction** using a Random Forest Regressor
- **Similar case lookup** via K-Nearest Neighbors (manhattan distance)
- **Bias insight** comparing average sentences by dependent status
- **Deviation analysis** — flags how far a prediction strays from similar cases
- **Pretrained model included** — no dataset or training required to run

---

## Quick Start (Recommended)

The pretrained model is included in this repo. Just install dependencies and run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
python sentence_predictor.py
```

That's it — no CSV file or training needed.

---

## Retrain the Model (Optional)

If you have new data and want to retrain:

1. Place your updated `cases_kz_500.csv` in the project root
2. Run:

```bash
python save_model.py
```

This overwrites `sentence_model.pkl` with the newly trained model.

---

## Dataset Format

If retraining, your CSV must have these columns:

| Column | Type | Description |
|---|---|---|
| `crime_type` | string | Type of crime (e.g. `theft`, `fraud`) |
| `severity` | int | Severity score of the offense |
| `prior_convictions` | int | Number of prior convictions |
| `age` | int | Age of the defendant |
| `employment_status` | string | `employed` or `unemployed` |
| `has_dependents` | int | `1` if has dependents, `0` otherwise |
| `sentence_length` | float | **Target** — sentence in years |

---

## Usage in Code

```python
from sentence_predictor import analyze_case

analyze_case({
    "crime_type": "theft",
    "severity": 3,
    "prior_convictions": 1,
    "age": 30,
    "employment_status": "employed",
    "has_dependents": 1
})
```

### Example output

```
Model loaded successfully

Predicted sentence: 3.47 years

Similar cases:
  - theft, 3.0 years
  - theft, 4.0 years
  - robbery, 3.5 years

Deviation: LOW

Bias insight:
    Avg sentence (no dependents): 3.21
    Avg sentence (has dependents): 2.89
```

---

## Project Structure

```
├── sentence_predictor.py  # Main script — load model and run predictions
├── save_model.py          # Run this to retrain and overwrite the model
├── sentence_model.pkl     # Pretrained model (ready to use)
├── requirements.txt
└── README.md
```

---

## Requirements

See `requirements.txt`. Main dependencies: `pandas`, `scikit-learn`.

---

## License

MIT

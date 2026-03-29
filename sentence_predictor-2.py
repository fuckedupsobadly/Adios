"""
Sentence Length Predictor — loads a pretrained model, no retraining needed.
Make sure sentence_model.pkl is in the same folder as this script.
"""

import pickle
import pandas as pd

# ── Load pretrained model ────────────────────────────────────────────────────
with open("sentence_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
knn = saved["knn"]
df = saved["df"]
preprocessor = model.named_steps["preprocessing"]

print("Model loaded successfully")


def analyze_case(new_case_dict):
    """
    Predict sentence length and show similar cases + bias insight.

    Parameters:
        new_case_dict (dict): Feature values for a single case.

    Example:
        analyze_case({
            "crime_type": "theft",
            "severity": 3,
            "prior_convictions": 1,
            "age": 30,
            "employment_status": "employed",
            "has_dependents": 1
        })
    """
    new_case = pd.DataFrame([new_case_dict])
    predicted = model.predict(new_case)[0]

    new_processed = preprocessor.transform(new_case)
    distances, indices = knn.kneighbors(new_processed)
    similar_cases = df.iloc[indices[0]]

    avg_similar = similar_cases["sentence_length"].mean()
    deviation = abs(predicted - avg_similar)

    if deviation < 1:
        deviation_label = "LOW"
    elif deviation < 2:
        deviation_label = "MEDIUM"
    else:
        deviation_label = "HIGH"

    group_avg = df.groupby("has_dependents")["sentence_length"].mean()
    bias_msg = (
        f"Avg sentence (no dependents): {group_avg[0]:.2f}\n"
        f"    Avg sentence (has dependents): {group_avg[1]:.2f}"
    )

    print("\nPredicted sentence:", round(predicted, 2), "years")
    print("\nSimilar cases:")
    for i, row in similar_cases.iterrows():
        print(f"  - {row['crime_type']}, {row['sentence_length']} years")
    print("\nDeviation:", deviation_label)
    print("\nBias insight:")
    print(f"    {bias_msg}")


# Example usage
if __name__ == "__main__":
    analyze_case({
        "crime_type": "theft",
        "severity": 3,
        "prior_convictions": 1,
        "age": 30,
        "employment_status": "employed",
        "has_dependents": 1
    })

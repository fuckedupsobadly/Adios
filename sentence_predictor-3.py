

import pickle
import pandas as pd


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



if __name__ == "__main__":
    print("\n─────────────────────────────────────")
    print("   Sentence Length Predictor (KZ)")
    print("─────────────────────────────────────")

   
    valid_crimes = ["theft", "fraud", "assault", "robbery"]
    while True:
        crime_type = input(f"\nCrime type ({' / '.join(valid_crimes)}): ").strip().lower()
        if crime_type in valid_crimes:
            break
        print(f"  Please enter one of: {', '.join(valid_crimes)}")

    
    while True:
        try:
            severity = int(input("Severity score (1-5): "))
            if 1 <= severity <= 5:
                break
            print("  Please enter a number between 1 and 5.")
        except ValueError:
            print("  Please enter a whole number.")

    
    while True:
        try:
            prior_convictions = int(input("Prior convictions (0, 1, 2, ...): "))
            if prior_convictions >= 0:
                break
            print("  Please enter 0 or more.")
        except ValueError:
            print("  Please enter a whole number.")

    
    while True:
        try:
            age = int(input("Defendant age (18-60): "))
            if 18 <= age <= 60:
                break
            print("  Please enter an age between 18 and 60.")
        except ValueError:
            print("  Please enter a whole number.")

    
    valid_employment = ["employed", "unemployed", "student"]
    while True:
        employment_status = input(f"Employment status ({' / '.join(valid_employment)}): ").strip().lower()
        if employment_status in valid_employment:
            break
        print(f"  Please enter one of: {', '.join(valid_employment)}")

    
    while True:
        has_dependents_input = input("Has dependents? (yes / no): ").strip().lower()
        if has_dependents_input in ("yes", "no"):
            has_dependents = 1 if has_dependents_input == "yes" else 0
            break
        print("  Please enter 'yes' or 'no'.")

    analyze_case({
        "crime_type": crime_type,
        "severity": severity,
        "prior_convictions": prior_convictions,
        "age": age,
        "employment_status": employment_status,
        "has_dependents": has_dependents
    })

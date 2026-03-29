import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# Load data
path = "cases_kz_500.csv"  # Place the CSV in the same folder as this script
df = pd.read_csv(path)

X = df.drop("sentence_length", axis=1)
y = df["sentence_length"]

categorical_cols = ["crime_type", "employment_status"]
numeric_cols = ["severity", "prior_convictions", "age", "has_dependents"]

# Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully")

# Build KNN on full dataset (manhattan distance)
preprocessor = model.named_steps["preprocessing"]
X_processed = preprocessor.transform(X)

knn = NearestNeighbors(n_neighbors=3, metric="manhattan")
knn.fit(X_processed)


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

"""
Run this script ONCE to train and save the model.
After running, a file called sentence_model.pkl will appear in the project folder.
You only need to run this again if you retrain on new data.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

path = "/Users/ald/Documents/python/adios/cases_kz_500.csv"
df = pd.read_csv(path)  

X = df.drop("sentence_length", axis=1)
y = df["sentence_length"]

categorical_cols = ["crime_type", "employment_status"]
numeric_cols = ["severity", "prior_convictions", "age", "has_dependents"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

preprocessor_fitted = model.named_steps["preprocessing"]
X_processed = preprocessor_fitted.transform(X)

knn = NearestNeighbors(n_neighbors=3, metric="manhattan")
knn.fit(X_processed)

# Save everything needed for inference
with open("sentence_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "knn": knn,
        "df": df  # needed for similar case lookup
    }, f)

print("Model saved to sentence_model.pkl")
print(f"File size: {__import__('os').path.getsize('sentence_model.pkl') / 1024 / 1024:.2f} MB")

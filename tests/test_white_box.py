# White box tests for app.py
# Test 1: register() password length check (line 146)
# Test 2: reset_password() token expiry check (line 216)
# Test 3: analyze_case() deviation label (lines 96-103)

from datetime import datetime, timedelta

import pandas as pd

import app as app_module
from app import User, db


# Test 1 - password length when registering

def register(client, password):
    return client.post("/register", data={"email": "student@nis.kz",
                                          "password": password,
                                          "confirm_password": password})


def saved_user():
    with app_module.app.app_context():
        return User.query.filter_by(email="student@nis.kz").first()


def test_1_1_normal_password(client):
    response = register(client, "Kazakh2026")
    user = saved_user()
    print("\n  result:", response.status_code, response.location, "| saved:", user is not None)
    assert response.location == "/login"
    assert user is not None
    assert user.password_hash != "Kazakh2026"


def test_1_2_extreme_password(client):
    response = register(client, "Abcd1234")
    print("\n  result:", response.status_code, response.location, "| saved:", saved_user() is not None)
    assert response.location == "/login"
    assert saved_user() is not None


def test_1_3_erroneous_password(client):
    response = register(client, "Abc1234")
    page = response.get_data(as_text=True)
    print("\n  result:", response.status_code, "| saved:", saved_user() is not None)
    assert "Password must be at least 8 characters." in page
    assert saved_user() is None


# Test 2 - reset link expires after 1 hour

def make_token(time_left):
    with app_module.app.app_context():
        user = User(email="reset@nis.kz")
        user.set_password("OldPass2026")
        token = user.generate_reset_token()
        user.reset_token_expires = datetime.utcnow() + time_left
        db.session.add(user)
        db.session.commit()
        return token


def test_2_1_normal_token(client):
    token = make_token(timedelta(minutes=50))
    response = client.get(f"/reset-password/{token}")
    print("\n  result:", response.status_code, "| form shown:", "Update password" in response.get_data(as_text=True))
    assert response.status_code == 200


def test_2_2_extreme_token(client):
    token = make_token(timedelta(seconds=1))
    response = client.get(f"/reset-password/{token}")
    print("\n  result:", response.status_code, "| form shown:", "Update password" in response.get_data(as_text=True))
    assert response.status_code == 200


def test_2_3_erroneous_token(client):
    token = make_token(timedelta(minutes=-1))
    response = client.get(f"/reset-password/{token}", follow_redirects=True)
    print("\n  result: went to", response.request.path)
    assert response.request.path == "/forgot-password"
    assert "Reset link is invalid or has expired." in response.get_data(as_text=True)


# Test 3 - deviation label
# The model is replaced with a fake one that always predicts the same number.
# The 3 similar cases are all 4.0 years, so the average is 4.0.

class FakePreprocessor:
    def transform(self, data):
        return data


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.named_steps = {"preprocessing": FakePreprocessor()}

    def predict(self, data):
        return [self.prediction]


class FakeKNN:
    def kneighbors(self, data):
        return [[0, 0, 0]], [[0, 1, 2]]


similar_cases = pd.DataFrame({
    "crime_type": ["theft", "theft", "fraud"],
    "has_dependents": [0, 1, 0],
    "sentence_length": [4.0, 4.0, 4.0],
})

case = {"crime_type": "theft", "severity": 3, "prior_convictions": 1,
        "age": 30, "employment_status": "employed", "has_dependents": 0}


def get_label(monkeypatch, prediction):
    monkeypatch.setattr(app_module, "model_data", {
        "model": FakeModel(prediction), "knn": FakeKNN(), "df": similar_cases})
    return app_module.analyze_case(case)["deviation_label"]


def test_3_1_normal_deviation(monkeypatch):
    label = get_label(monkeypatch, 4.5)
    print("\n  result: deviation 0.5 gives", label)
    assert label == "LOW"


def test_3_2_extreme_deviation(monkeypatch):
    label = get_label(monkeypatch, 5.0)
    print("\n  result: deviation 1.0 gives", label)
    assert label == "MEDIUM"


def test_3_3_erroneous_deviation(monkeypatch):
    label = get_label(monkeypatch, 1.0)
    print("\n  result: difference -3.0 gives", label)
    assert label == "HIGH"

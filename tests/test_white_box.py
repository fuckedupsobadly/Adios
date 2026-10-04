"""
WHITE BOX TESTING
Tests are designed from the source code of app.py: each one targets a specific
condition/branch and inspects internal state (database rows, token timestamps,
the computed deviation) that a normal user cannot see.

  WB1  register()        app.py line 146   len(password) < 8          String   Length check
  WB2  reset_password()  app.py line 216   reset_token_expires < now   DateTime Range check
  WB3  analyze_case()    app.py lines 96-103  abs() + LOW/MEDIUM/HIGH  Float    Range check
"""
from datetime import datetime, timedelta

import pandas as pd

import app as app_module
from app import User, db


# ─────────────────────────────────────────────────────────────────────────────
# WB1 — register(): password length branch  (app.py line 146)
# ─────────────────────────────────────────────────────────────────────────────

def register(client, password):
    return client.post("/register", data={"email": "student@nis.kz",
                                          "password": password,
                                          "confirm_password": password})


def stored_user():
    with app_module.app.app_context():
        return User.query.filter_by(email="student@nis.kz").first()


def test_WB1_1_normal_password_10_chars(client):
    response = register(client, "Kazakh2026")
    user = stored_user()
    print(f"\n  ACTUAL: status={response.status_code} -> {response.location}, "
          f"user saved={user is not None}, hash starts '{user.password_hash[:7]}'")
    assert response.status_code == 302 and response.location == "/login"
    assert user is not None
    assert user.password_hash != "Kazakh2026"          # never stored as plain text
    assert user.password_hash.startswith("scrypt:")


def test_WB1_2_extreme_password_exactly_8_chars(client):
    response = register(client, "Abcd1234")
    user = stored_user()
    print(f"\n  ACTUAL: status={response.status_code} -> {response.location}, "
          f"user saved={user is not None}")
    assert response.status_code == 302                 # len 8 -> "< 8" is False
    assert user is not None


def test_WB1_3_erroneous_password_7_chars(client):
    response = register(client, "Abc1234")
    page = response.get_data(as_text=True)
    user = stored_user()
    print(f"\n  ACTUAL: status={response.status_code}, user saved={user is not None}, "
          f"message shown={'Password must be at least 8 characters.' in page}")
    assert response.status_code == 200                 # stays on the form
    assert "Password must be at least 8 characters." in page
    assert user is None                                # nothing written to DB


# ─────────────────────────────────────────────────────────────────────────────
# WB2 — reset_password(): token expiry check  (app.py line 216)
# ─────────────────────────────────────────────────────────────────────────────

def make_token(expires_in):
    """Create a user and give them a reset token that expires after `expires_in`."""
    with app_module.app.app_context():
        user = User(email="reset@nis.kz")
        user.set_password("OldPass2026")
        token = user.generate_reset_token()
        user.reset_token_expires = datetime.utcnow() + expires_in
        db.session.add(user)
        db.session.commit()
        return token


def test_WB2_1_normal_token_issued_10_min_ago(client):
    token = make_token(timedelta(minutes=50))
    response = client.get(f"/reset-password/{token}")
    print(f"\n  ACTUAL: status={response.status_code}, "
          f"reset form shown={'Update password' in response.get_data(as_text=True)}")
    assert response.status_code == 200
    assert "Update password" in response.get_data(as_text=True)


def test_WB2_2_extreme_token_expires_in_1_second(client):
    token = make_token(timedelta(seconds=1))
    response = client.get(f"/reset-password/{token}")
    print(f"\n  ACTUAL: status={response.status_code}, "
          f"reset form shown={'Update password' in response.get_data(as_text=True)}")
    assert response.status_code == 200
    assert "Update password" in response.get_data(as_text=True)


def test_WB2_3_erroneous_token_expired_1_min_ago(client):
    token = make_token(timedelta(minutes=-1))
    response = client.get(f"/reset-password/{token}", follow_redirects=True)
    page = response.get_data(as_text=True)
    print(f"\n  ACTUAL: redirected to {response.request.path}, "
          f"message shown={'Reset link is invalid or has expired.' in page}")
    assert response.request.path == "/forgot-password"
    assert "Reset link is invalid or has expired." in page


# ─────────────────────────────────────────────────────────────────────────────
# WB3 — analyze_case(): deviation label  (app.py lines 96-103)
# The real model is replaced by a stub that returns a fixed prediction, so the
# deviation reaching the if/elif/else is known exactly.
# The 3 "similar cases" all have sentence_length 4.0, so avg_similar = 4.00.
# ─────────────────────────────────────────────────────────────────────────────

class StubPreprocessor:
    def transform(self, frame):
        return frame


class StubModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.named_steps = {"preprocessing": StubPreprocessor()}

    def predict(self, frame):
        return [self.prediction]


class StubKNN:
    def kneighbors(self, processed):
        return [[0, 0, 0]], [[0, 1, 2]]


SIMILAR_CASES = pd.DataFrame({
    "crime_type": ["theft", "theft", "fraud"],
    "has_dependents": [0, 1, 0],
    "sentence_length": [4.0, 4.0, 4.0],
})

CASE = {"crime_type": "theft", "severity": 3, "prior_convictions": 1,
        "age": 30, "employment_status": "employed", "has_dependents": 0}


def run_with_prediction(monkeypatch, predicted):
    monkeypatch.setattr(app_module, "model_data", {
        "model": StubModel(predicted), "knn": StubKNN(), "df": SIMILAR_CASES})
    return app_module.analyze_case(CASE)


def test_WB3_1_normal_deviation_0_50(monkeypatch):
    result = run_with_prediction(monkeypatch, 4.50)
    print(f"\n  ACTUAL: |4.50 - 4.00| = 0.50 -> {result['deviation_label']}")
    assert result["deviation_label"] == "LOW"


def test_WB3_2_extreme_deviation_exactly_1_00(monkeypatch):
    result = run_with_prediction(monkeypatch, 5.00)
    print(f"\n  ACTUAL: |5.00 - 4.00| = 1.00 -> {result['deviation_label']}")
    assert result["deviation_label"] == "MEDIUM"       # 1.00 < 1 is False


def test_WB3_3_erroneous_negative_difference(monkeypatch):
    result = run_with_prediction(monkeypatch, 1.00)
    print(f"\n  ACTUAL: 1.00 - 4.00 = -3.00, abs() -> 3.00 -> {result['deviation_label']}")
    assert result["deviation_label"] == "HIGH"         # without abs(), -3 < 1 -> LOW

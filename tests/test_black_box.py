"""
BLACK BOX TESTING
Tests are designed only from the specification (documentation section 2.2 and
section 9): the tester fills in the Predict form and checks what appears on the
page. No source code or database is inspected.

  BB1  Age                Integer        Range check   (18-60)
  BB2  Prior convictions  Integer/text   Type check    (whole number, 0 or more)
  BB3  Has dependents     Flag 0/1       Lookup check  (No = 0, Yes = 1)
"""
import re

VALID_CASE = {"crime_type": "theft", "severity": "3", "prior_convictions": "1",
              "age": "30", "employment_status": "employed", "has_dependents": "0"}


def submit(client, **changes):
    form = {**VALID_CASE, **changes}
    form = {field: value for field, value in form.items() if value is not None}
    response = client.post("/predict", data=form)
    page = response.get_data(as_text=True)
    prediction = re.search(r'class="result-number">([^<]+)<', page)
    errors = re.findall(r'class="alert alert-error">([^<]+)<', page)
    return response.status_code, prediction.group(1) if prediction else None, errors


def report(status, prediction, errors):
    shown = f"prediction {prediction} years" if prediction else "no prediction"
    print(f"\n  ACTUAL: HTTP {status}, {shown}, errors={errors}")


# ─────────────────────────────────────────────────────────────────────────────
# BB1 — Age (Range check: 18 to 60)
# ─────────────────────────────────────────────────────────────────────────────

def test_BB1_1_normal_age_35(logged_in_client):
    status, prediction, errors = submit(logged_in_client, age="35")
    report(status, prediction, errors)
    assert status == 200 and prediction is not None and errors == []


def test_BB1_2_extreme_age_60(logged_in_client):
    status, prediction, errors = submit(logged_in_client, age="60")
    report(status, prediction, errors)
    assert status == 200 and prediction is not None and errors == []


def test_BB1_3_erroneous_age_75(logged_in_client):
    status, prediction, errors = submit(logged_in_client, age="75")
    report(status, prediction, errors)
    assert prediction is None
    assert errors == ["Age must be 18–60."]


# ─────────────────────────────────────────────────────────────────────────────
# BB2 — Prior convictions (Type check: whole number, 0 or more)
# ─────────────────────────────────────────────────────────────────────────────

def test_BB2_1_normal_prior_convictions_2(logged_in_client):
    status, prediction, errors = submit(logged_in_client, prior_convictions="2")
    report(status, prediction, errors)
    assert status == 200 and prediction is not None and errors == []


def test_BB2_2_extreme_prior_convictions_0(logged_in_client):
    status, prediction, errors = submit(logged_in_client, prior_convictions="0")
    report(status, prediction, errors)
    assert status == 200 and prediction is not None and errors == []


def test_BB2_3_erroneous_prior_convictions_text(logged_in_client):
    status, prediction, errors = submit(logged_in_client, prior_convictions="two")
    report(status, prediction, errors)
    assert prediction is None
    assert errors == ["Prior convictions must be 0 or more."]


# ─────────────────────────────────────────────────────────────────────────────
# BB3 — Has dependents (Lookup check: only "0" = No or "1" = Yes)
# ─────────────────────────────────────────────────────────────────────────────

def test_BB3_1_normal_has_dependents_yes(logged_in_client):
    status, prediction, errors = submit(logged_in_client, has_dependents="1")
    report(status, prediction, errors)
    assert status == 200 and prediction is not None and errors == []


def test_BB3_2_extreme_has_dependents_not_sent(logged_in_client):
    _, prediction_no, _ = submit(logged_in_client, has_dependents="0")
    status, prediction, errors = submit(logged_in_client, has_dependents=None)
    report(status, prediction, errors)
    print(f"  (explicit 'No' gave {prediction_no} years)")
    assert status == 200 and errors == []
    assert prediction == prediction_no                 # treated the same as "No"


def test_BB3_3_erroneous_has_dependents_abc(logged_in_client):
    status, prediction, errors = submit(logged_in_client, has_dependents="abc")
    report(status, prediction, errors)
    assert status == 200                               # page must not crash
    assert prediction is None
    assert len(errors) == 1                            # a clear error message

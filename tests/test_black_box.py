# Black box tests for the Predict page
# Test 1: Age (must be 18 to 60)
# Test 2: Prior convictions (must be a whole number, 0 or more)
# Test 3: Has dependents (must be Yes or No)

import re

normal_case = {"crime_type": "theft", "severity": "3", "prior_convictions": "1",
               "age": "30", "employment_status": "employed", "has_dependents": "0"}


def send_form(client, **changes):
    form = dict(normal_case)
    form.update(changes)
    form = {key: value for key, value in form.items() if value is not None}
    page = client.post("/predict", data=form).get_data(as_text=True)
    prediction = re.search(r'class="result-number">([^<]+)<', page)
    errors = re.findall(r'class="alert alert-error">([^<]+)<', page)
    prediction = prediction.group(1) if prediction else None
    print("\n  result: prediction =", prediction, "| errors =", errors)
    return prediction, errors


# Test 1 - Age

def test_1_1_normal_age(logged_in_client):
    prediction, errors = send_form(logged_in_client, age="35")
    assert prediction is not None and errors == []


def test_1_2_extreme_age(logged_in_client):
    prediction, errors = send_form(logged_in_client, age="60")
    assert prediction is not None and errors == []


def test_1_3_erroneous_age(logged_in_client):
    prediction, errors = send_form(logged_in_client, age="75")
    assert prediction is None
    assert errors == ["Age must be 18–60."]


# Test 2 - Prior convictions

def test_2_1_normal_prior_convictions(logged_in_client):
    prediction, errors = send_form(logged_in_client, prior_convictions="2")
    assert prediction is not None and errors == []


def test_2_2_extreme_prior_convictions(logged_in_client):
    prediction, errors = send_form(logged_in_client, prior_convictions="0")
    assert prediction is not None and errors == []


def test_2_3_erroneous_prior_convictions(logged_in_client):
    prediction, errors = send_form(logged_in_client, prior_convictions="two")
    assert prediction is None
    assert errors == ["Prior convictions must be 0 or more."]


# Test 3 - Has dependents

def test_3_1_normal_dependents(logged_in_client):
    prediction, errors = send_form(logged_in_client, has_dependents="1")
    assert prediction is not None and errors == []


def test_3_2_extreme_dependents(logged_in_client):
    prediction_no, _ = send_form(logged_in_client, has_dependents="0")
    prediction, errors = send_form(logged_in_client, has_dependents=None)
    assert errors == []
    assert prediction == prediction_no


def test_3_3_erroneous_dependents(logged_in_client):
    prediction, errors = send_form(logged_in_client, has_dependents="abc")
    assert prediction is None
    assert errors == ["Has dependents must be Yes or No."]

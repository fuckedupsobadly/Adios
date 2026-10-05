# Regression tests for bugs found while debugging (not part of the 9 + 9 tests
# in the white box and black box testing documents)


def test_huge_prior_convictions_is_rejected_not_crashed(logged_in_client):
    # 3,000,000,000 is past PostgreSQL's INTEGER limit: saving it crashed the live site with HTTP 500
    form = {"crime_type": "theft", "severity": "3", "prior_convictions": "3000000000",
            "age": "30", "employment_status": "employed", "has_dependents": "0"}
    response = logged_in_client.post("/predict", data=form)
    print("\n  result:", response.status_code)
    assert response.status_code == 200
    assert "Prior convictions must be 0–20." in response.get_data(as_text=True)

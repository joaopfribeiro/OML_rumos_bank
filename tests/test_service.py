import json
import pytest
import requests

with open('./config/app.json') as f:
    config = json.load(f)

def test_default_payment_prediction():
    """
    Test for the /has_diabetes endpoint with valid input data.
    It should return a prediction in the response.
    """
    response = requests.post(f"http://localhost:{config['service_port']}/default_payment", json={
        'LIMIT_BAL': 80000.0,
        'SEX': 2,       
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 34,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': -1,
        'PAY_6': -1,
        'BILL_AMT1': 55933.0,
        'BILL_AMT2': 11865.0,
        'BILL_AMT3': 4602.0,
        'BILL_AMT4': 34197.0,
        'BILL_AMT5': 27398.0,
        'BILL_AMT6': 28646.0,
        'PAY_AMT1': 4000.0,
        'PAY_AMT2': 2333.0,
        'PAY_AMT3': 3032.0,
        'PAY_AMT4': 28298.0,
        'PAY_AMT5': 2000.0,
        'PAY_AMT6': 2000.0
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))
    assert response.json()["prediction"] == 0
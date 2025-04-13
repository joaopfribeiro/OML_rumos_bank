import json
import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    model_name = config["model_name"]
    model_version = config["model_version"]
    
    model = f"models:/{config['model_name']}@{config['model_version']}"
    print(f"Modelo carregado com sucesso: {model}") 
    
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 320000.0,
        'SEX': 1,
        'EDUCATION': 1,
        'MARRIAGE': 1,
        'AGE': 49,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': -1,
        'PAY_5': -1,
        'PAY_6': -1,
        'BILL_AMT1': 253286.0,
        'BILL_AMT2': 246536.0,
        'BILL_AMT3': 194663.0,
        'BILL_AMT4': 70074.0,
        'BILL_AMT5': 5856.0,
        'BILL_AMT6': 195599.0,
        'PAY_AMT1': 10358.0,
        'PAY_AMT2': 10000.0,
        'PAY_AMT3': 75940.0,
        'PAY_AMT4': 20000.0,
        'PAY_AMT5': 195599.0,
        'PAY_AMT6': 50000.0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_dir(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 250000.0,
        'SEX': 1,
        'EDUCATION': 3,
        'MARRIAGE': 2,
        'AGE': 24,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 2,
        'PAY_4': 2,
        'PAY_5': 2,
        'PAY_6': 2,
        'BILL_AMT1': 15376.0,
        'BILL_AMT2': 18010.0,
        'BILL_AMT3': 17428.0,
        'BILL_AMT4': 18338.0,
        'BILL_AMT5': 17905.0,
        'BILL_AMT6': 19104.0,
        'PAY_AMT1': 3200.0,
        'PAY_AMT2': 0.0,
        'PAY_AMT3': 1500.0,
        'PAY_AMT4': 0.0,
        'PAY_AMT5': 1650.0,
        'PAY_AMT6': 50000.0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 60000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 50,
        'PAY_0': -1,
        'PAY_2': 0,
        'PAY_3': -1,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 8617.0,
        'BILL_AMT2': 5670.0,
        'BILL_AMT3': 35835.0,
        'BILL_AMT4': 20940.0,
        'BILL_AMT5': 19146.0,
        'BILL_AMT6': 19131.0,
        'PAY_AMT1': 2000.0,
        'PAY_AMT2': 36681.0,
        'PAY_AMT3': 10000.0,
        'PAY_AMT4': 9000.0,
        'PAY_AMT5': 689.0,
        'PAY_AMT6': 679.0
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )
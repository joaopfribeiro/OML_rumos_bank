import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from pydantic import BaseModel, Field
import pandas as pd
import json
import uvicorn

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)


# Define the inputs expected in the request body as JSON
class Request(BaseModel):
    """
    Request model for the API, defining the input structure.

    Attributes:
        LIMIT_BAL (float): Amount of the given credit.
        SEX (int): Gender (1 = male, 2 = female).
        EDUCATION (int): Education (1 = graduate school, 2 = university, 3 = high school, 4 = others).
        MARRIAGE (int): Marital status (1 = married, 2 = single, 3 = divorced).
        AGE (int): Age (years).        
        PAY_0 (int): Repayment status in September, 2005.
        PAY_2 (int): Repayment status in August, 2005.                
        PAY_3 (int): Repayment status in July, 2005.                
        PAY_4 (int): Repayment status in June, 2005.                        
        PAY_5 (int): Repayment status in May, 2005.                        
        PAY_6 (int): Repayment status in April, 2005.                        
        BILL_AMT1 (float): Amount of bill statement in September, 2005.                        
        BILL_AMT2 (float): Amount of bill statement in August, 2005.                        
        BILL_AMT3 (float): Amount of bill statement in July, 2005.                        
        BILL_AMT4 (float): Amount of bill statement in June, 2005.                        
        BILL_AMT5 (float): Amount of bill statement in May, 2005.                        
        BILL_AMT6 (float): Amount of bill statement in April, 2005.                        
        PAY_AMT1 (float): Amount of previous payment in September, 2005.                        
        PAY_AMT2 (float): Amount of previous payment in August, 2005.                        
        PAY_AMT3 (float): Amount of previous payment in July, 2005.                        
        PAY_AMT4 (float): Amount of previous payment in June, 2005.                        
        PAY_AMT5 (float): Amount of previous payment in May, 2005.                        
        PAY_AMT6 (float): Amount of previous payment in April, 2005.            
    """
    LIMIT_BAL: float = Field(..., ge=0)
    SEX: int = 1
    EDUCATION: int = 1
    MARRIAGE: int = 1
    AGE: int = 25
    PAY_0: int = 1
    PAY_2: int = 1
    PAY_3: int = 1
    PAY_4: int = 1
    PAY_5: int = 1
    PAY_6: int = 1
    BILL_AMT1: float = 45.8
    BILL_AMT2: float = 45.8
    BILL_AMT3: float = 45.8
    BILL_AMT4: float = 45.8
    BILL_AMT5: float = 45.8
    BILL_AMT6: float = 45.8
    PAY_AMT1: float = 45.8
    PAY_AMT2: float = 45.8
    PAY_AMT3: float = 45.8
    PAY_AMT4: float = 45.8
    PAY_AMT5: float = 45.8
    PAY_AMT6: float = 45.8

# Create a FastAPI application
app = fastapi.FastAPI()

# Add CORS middleware to allow all origins, methods, and headers for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """
        
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri)
    
    print(f"Loaded model {model_uri}")


@app.post("/default_payment")
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}

# Run the app on port 5002
uvicorn.run(app=app, port=config["service_port"], host="0.0.0.0")
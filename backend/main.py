import pandas as pd
import io
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from fbprophet import Prophet

# Create FastAPI instance
app = FastAPI()

# Initiate MLflow client
client = MlflowClient()

# Load best Prophet model from MLflow
all_exps = [exp.experiment_id for exp in client.list_experiments()]
runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)

best_run = runs.loc[runs['metrics.rmse'].idxmin()]
run_id, exp_id = best_run['run_id'], best_run['experiment_id']

print(f'Loading best model: Run {run_id} of Experiment {exp_id}')
best_model = mlflow.pyfunc.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

def prepare_data(df):
    """ Prepare Data for Prophet """
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "ds", "sales": "y"})
    return df

@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    
    # Load CSV file into DataFrame
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)

    # Prepare data for Prophet
    test_df = prepare_data(test_df)

    # Generate Predictions
    future = best_model.make_future_dataframe(periods=len(test_df))
    forecast = best_model.predict(future)

    # Extract relevant predictions
    preds_final = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(test_df)).to_dict(orient="records")

    # Convert predictions into JSON format
    json_compatible_item_data = jsonable_encoder(preds_final)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the Demand Forecasting API using Prophet </h2>
    <p> The Prophet model and FastAPI instances have been set up successfully </p>
    <p> You can upload CSV files for demand forecasting </p>
    </body>
    """
    return HTMLResponse(content=content)

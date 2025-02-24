
import argparse
import pandas as pd
import mlflow
import mlflow.prophet
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Prophet AutoML Training and MLflow Tracking")
    
    parser.add_argument('--name', '--experiment_name',
                        metavar='', 
                        default='automl-demand-forecasting',
                        help='Name of Experiment. Default is automl-demand-forecasting',
                        type=str)
    
    parser.add_argument('--target', '--t',
                        metavar='', 
                        required=True,
                        help='Name of Target Column (y)',
                        type=str)
    
    parser.add_argument('--date_col', '--d',
                        metavar='',
                        required=True,
                        help='Name of Date Column (ds)',
                        type=str)
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Initiate MLflow
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(args.name)
    
    # Print experiment details
    experiment = mlflow.get_experiment_by_name(args.name)
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    
    # Load dataset
    data = pd.read_csv('data/processed/train.csv')
    
    # Ensure correct data types
    data[args.date_col] = pd.to_datetime(data[args.date_col])
    
    # Train-test split (80-20)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Start MLflow experiment
    with mlflow.start_run():
        # Initialize and train Prophet model
        model = Prophet()
        model.fit(train[[args.date_col, args.target]].rename(columns={args.date_col: 'ds', args.target: 'y'}))
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        
        # Evaluate model
        test_forecast = forecast[-len(test):]['yhat'].values
        test_actual = test[args.target].values
        mae = mean_absolute_error(test_actual, test_forecast)
        
        # Log metrics and model
        mlflow.log_metric("MAE", mae)
        mlflow.prophet.log_model(model, artifact_path="model")
        
        model_uri = mlflow.get_artifact_uri("model")
        print(f'Prophet model saved in {model_uri}')
        
        # Save forecast results
        forecast.to_csv(f'mlruns/{experiment.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/model/forecast.csv', index=False)
        print('Forecasting complete. Results saved.')

if __name__ == "__main__":
    main()

# =========================================
# Prophet AutoML Training with MLflow Tracking
# Author: KAZI SAKIB
# Last Modified: 15th Feb 2025
# =========================================
# Command to execute script locally: streamlit run app.py
# Command to run Docker image: docker run -d -p 8501:8501 <streamlit-app-name>:latest

import streamlit as st
import requests
import pandas as pd
import io
import json

st.title('End-to-End AutoML Project: Demand-Forecasting-Retail')

# FastAPI Endpoint (Update for Docker Compatibility)
endpoint = 'http://host.docker.internal:8000/predict'  # Windows/Mac Docker
# endpoint = 'http://localhost:8000/predict'  # For local testing

test_csv = st.file_uploader('Upload Test CSV', type=['csv'], accept_multiple_files=False)

if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())

    # Convert DataFrame to BytesIO (for sending to FastAPI)
    test_bytes_obj = io.StringIO()
    test_df.to_csv(test_bytes_obj, index=False)
    test_bytes_obj.seek(0)  # Reset pointer

    files = {"file": ('test_dataset.csv', test_bytes_obj.getvalue())}  # Encode properly

    if st.button('Start Prediction'):
        if test_df.empty:
            st.warning("Please upload a valid test dataset!")
        else:
            with st.spinner('Prediction in Progress...'):
                try:
                    response = requests.post(endpoint, files=files, timeout=30)
                    response.raise_for_status()  # Raise error for bad response
                    result_json = response.json()
                    
                    st.success('Prediction Completed! Download results below:')
                    st.download_button(
                        label='Download Predictions',
                        data=json.dumps(result_json, indent=4),
                        file_name='predictions.json'
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")

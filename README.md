# Demand Forecasting for Retail

## 📦 Deployment
### 1️ **Run FastAPI Server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2️ **Run Streamlit UI**
```bash
streamlit run app.py
```

### 3️ **Dockerization**
```bash
# Build Docker Image
docker build -t demand-forecasting .

# Run Container
docker run -p 8501:8501 demand-forecasting
```





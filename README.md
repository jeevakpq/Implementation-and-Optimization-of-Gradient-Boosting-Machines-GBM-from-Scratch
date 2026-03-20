# GBM MLOps Project

**Project Overview:**  
This project implements a **Gradient Boosting Machine (GBM) classifier from scratch** using NumPy and deploys it as a **FastAPI REST API**. The application is containerized with Docker for production-ready deployment.

The project demonstrates:

- Custom GBM implementation with CART regression trees
- Hyperparameter tuning and comparison with XGBoost
- Model serialization using pickle
- API deployment using FastAPI
- Containerization with Docker

---

## Features

- Build GBM from scratch with Python & NumPy
- Train and evaluate on synthetic datasets
- FastAPI endpoints for real-time predictions
- Dockerized environment for scalable deployment
- Comparison with XGBoost to validate performance

---

## Project Structure


project 3/
├─ src/
│ ├─ model.py # Custom GBM model implementation
│ ├─ train.py # Train and save GBM model
│ ├─ predict.py # Load model & make predictions
│ └─ init.py
├─ api/
│ └─ app.py # FastAPI app with endpoints
├─ models/
│ └─ gbm_model.pkl # Serialized trained model
├─ Dockerfile # Docker instructions
├─ requirements.txt # Python dependencies
└─ README.md


---

## Installation & Setup

1. Clone the repository:

git clone https://github.com/<your-username>/gbm-mlops-project.git
cd gbm-mlops-project

2.Create a virtual environment (recommended):

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

3.Install dependencies:

pip install -r requirements.txt

4.Train the model:

python -m src.train

This will generate the gbm_model.pkl file in the models/ folder.

Running FastAPI Locally

1.Start the API server:

python -m uvicorn api.app:app --reload

2.Open Swagger UI in your browser:

http://127.0.0.1:8000/docs

3.Test /predict endpoint:

{
  "features": [0.5, -1.2, 0.3, 0.7, -0.1]
}

The API will return the predicted class for the input features.

Testing via Python

You can test the API with Python:

import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "features": [0.5, -1.2, 0.3, 0.7, -0.1]
}

response = requests.post(url, json=data)
print(response.json())

Expected output:

{"prediction": 1}
Testing via curl
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"features\": [0.5, -1.2, 0.3, 0.7, -0.1]}"
Docker Deployment

1.Build Docker image:

docker build -t gbm-api .

2.Run container:

docker run -p 8000:8000 gbm-api

3.Open Swagger UI:

http://localhost:8000/docs

Requirements

Python 3.10+

NumPy, scikit-learn

FastAPI, Uvicorn

Docker (optional for containerization)

Author

Jeevanandham KP

LinkedIn: https://linkedin.com/in/jeevanandham-kp-a93a51187

GitHub: https://github.com/jeevakpq


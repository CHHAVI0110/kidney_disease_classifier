# Kidney_Disease_Classifier-MLflow-DVC

# Project Overview
This project classifies kidney images into four categories:  

- **Cyst**  
- **Normal**  
- **Stone**  
- **Tumor**  

It leverages:  
- **TensorFlow** for image classification  
- **MLflow** for experiment tracking  
- **DVC** for data version control  
- **FastAPI** to serve predictions via REST API  
- **Docker** for containerized deployment  

# Tech Stack
- Python 3.10  
- TensorFlow (CPU)  
- FastAPI  
- MLflow  
- DVC  
- Docker  
- Conda

# HOW TO RUN?
### STEPS:

Clone the repository

```bash
https://github.com/CHHAVI0110/Kidney_Disease_Classifier
```
### STEP 01- Create a conda environment after opening repository: 

```bash
conda create -n cnncls python=3.10 -y
```

```bash
conda activate cnncls
```


### STEP 02- Install the requirements:
```bash
pip install -r requirements.txt
``` 

### STEP 03- Set MLflow environment variables:

```bash
export MLFLOW_TRACKING_URI:
export MLFLOW_TRACKING_USERNAME:
export MLFLOW_TRACKING_PASSWORD:
```
### STEP 04- Run DVC pipeline:
```bash

1. dvc init 
2. dvc repro
3. dvc dag
```

### STEP 05– Start FastAPI server
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

### STEP 06: Docker Deployment
```bash
docker build -t kidneyclassifier .
docker run -p 5000:5000 kidneyclassifier
```
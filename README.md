 Kidney Disease Classifier – MLflow + DVC + FastAPI + Docker

**Project by [Chhavi Sharma](https://github.com/CHHAVI0110/Kidney_Disease_Classifier)**

---

## 🚀 Project Overview
This project predicts the risk of kidney disease based on patient features using a TensorFlow model.  
It leverages **MLflow** for experiment tracking and **DVC** for data version control.  
The API is built using **FastAPI** and the project is **Docker-ready** for production deployment.

---

## 🔧 Tech Stack
- Python 3.10  
- TensorFlow  
- FastAPI  
- MLflow for experiment tracking  
- DVC for data version control  
- Docker for containerization  
- Conda for environment management  

---

## 🏃 How to Run Locally

### Step 0 – Clone the repository
```bash
git clone https://github.com/CHHAVI0110/Kidney_Disease_Classifier.git
cd Kidney_Disease_Classifier
Step 1 – Create and activate a conda environment
bash
Copy code
conda create -n cnncls python=3.10 -y
conda activate cnncls
Step 2 – Install the requirements
bash
Copy code
pip install -r requirements.txt
Step 3 – Set MLflow environment variables
bash
Copy code
export MLFLOW_TRACKING_URI=<your_mlflow_uri>
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_password>
Step 4 – Initialize and run DVC
bash
Copy code
# Initialize DVC
dvc init

# Reproduce the pipeline
dvc repro

# Visualize pipeline
dvc dag
Step 5 – Run the FastAPI application
bash
Copy code
uvicorn app:app --host 0.0.0.0 --port 5000
Step 6 – Test the API
Example using curl:

bash
Copy code
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 45, "blood_pressure": 80, "blood_sugar": 120}'
📂 Repository Structure
bash
Copy code
├── .dvc/                  # DVC configuration
├── config/                # Configuration files (params.yaml etc)
├── logs/                  # Training/inference logs
├── research/              # Experiment notebooks
├── src/                   # Source code (model, data, inference)
├── Dockerfile             # Docker build instructions
├── requirements.txt       # Python dependencies
├── app.py / main.py       # FastAPI application
├── inputImage.jpg         # Sample input image (optional)
├── scores.json            # Model evaluation metrics
└── …
📈 Model & Inference
Model trained using [dataset name / source]

Example performance metrics:

makefile
Copy code
Accuracy: XX%  
AUC-ROC: XX  
Confusion Matrix: (include screenshot or text)
Example API output:

json
Copy code
{ "prediction": "High risk of kidney disease" }
🐳 Docker Deployment (Optional)
Build the Docker image:

bash
Copy code
docker build -t kidneyclassifier .
Run the Docker container:

bash
Copy code
docker run -p 5000:5000 kidneyclassifier
Access the FastAPI API at http://localhost:5000/predict

⚠️ Note: The current image size is ~7.6 GB, may need optimization for free-tier cloud deployment.

🎯 Future Work
Optimize Docker image size (multi-stage build, model compression)

Deploy to free cloud services (e.g., Railway, Render, or Cloud Run)

Add frontend dashboard for interactive input

Add authentication, logging, and monitoring for production


# Kidney_Disease_Classifier-MLflow-DVC



# HOW TO RUN?
### STEPS:

Clone the repository

```bash
https://github.com/CHHAVI0110/Kidney_Disease_Classifier
```
### STEP 01- Create a conda environment after opening repository

```bash
conda create -n cnncls python=3.10 -y
```

```bash
conda activate cnncls
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt

### STEP 03- Run This To Export As Env Variables:

```bash
export MLFLOW_TRACKING_URI:
export MLFLOW_TRACKING_USERNAME:
export MLFLOW_TRACKING_PASSWORD:
```
### STEP 04- DVC cmd

1. dvc init 
2. dvc repro
3. dvc dag
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os
import subprocess

from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

# Environment setup
os.putenv('LANG', 'en_us.UTF-8')
os.putenv('LC_ALL', 'en_us.UTF-8')

app = FastAPI(
    title="CNN Classifier API",
    description="API for training and predicting using CNN Classifier",
    version="1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates (for index.html)
templates = Jinja2Templates(directory="templates")


# Class equivalent to ClientApp
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
async def trainRoute():
    """Trigger model training."""
    subprocess.run(["python", "main.py"], check=True)
    return {"message": "Training Done Successfully!!"}

@app.post("/predict")
async def predict_route(request: Request):
    try:
        data = await request.json()
        image = data.get("image")
        if not image:
            raise ValueError("No image provided")

        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()

        # result is already {"class": "...", "confidence": ...}
        return {"result": result}

    except Exception as e:
        return JSONResponse(content={"error": f"Unable to process image: {str(e)}"}, status_code=500)


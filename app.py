from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Malaria Detection API", version="2.0.0")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Load your trained model
try:
    model = load_model("model1_cnn.h5")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Constants
IMG_SIZE = (32, 32)
CLASS_NAMES = ["Parasitized", "Uninfected"]
CONFIDENCE_THRESHOLD = 0.65

# Clinical guidance database
CLINICAL_GUIDANCE = {
    "Parasitized": {
        "summary": "⚠️ Plasmodium infection detected",
        "treatment": """
        <strong>Immediate Clinical Actions:</strong><br>
        • Confirm with rapid diagnostic test (RDT) or microscopy<br>
        • Start Artemisinin-based Combination Therapy (ACT) immediately<br>
        • Recommended: Artemether-Lumefantrine (Coartem)<br>
        • For severe malaria: IV Artesunate<br>
        • Monitor for complications, especially in children and pregnant women
        """,
        "follow_up": "Schedule follow-up in 3 days to check treatment response"
    },
    "Uninfected": {
        "summary": "✅ No malaria parasites detected",
        "treatment": """
        <strong>Clinical Recommendation:</strong><br>
        • No antimalarial treatment required<br>
        • If symptoms persist, test for other febrile illnesses:<br>
          - Dengue fever<br>
          - Typhoid fever<br>
          - Respiratory infections<br>
        • Supportive care for symptom relief
        """,
        "follow_up": "Return if symptoms worsen or persist beyond 48 hours"
    }
}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model inference
    Matches training preprocessing exactly
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image format: {str(e)}")

async def run_prediction(image_bytes: bytes) -> Dict[str, Any]:
    """Run model prediction asynchronously"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    loop = asyncio.get_event_loop()
    
    def predict():
        img_array = preprocess_image(image_bytes)
        prediction = model.predict(img_array, verbose=0)[0][0]
        return prediction
    
    try:
        prediction = await loop.run_in_executor(executor, predict)
        
        if prediction > 0.5:
            label = "Uninfected"
            confidence = float(prediction)
        else:
            label = "Parasitized"
            confidence = float(1 - prediction)
        
        # Apply confidence threshold
        is_confident = confidence >= CONFIDENCE_THRESHOLD
        
        return {
            "prediction": label,
            "confidence": round(confidence, 3),
            "is_confident": is_confident,
            "raw_score": float(prediction)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def home():
    """Health check endpoint"""
    return {
        "message": "Malaria Detection API is running",
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "version": "2.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict malaria from blood smear image
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file with size limit (10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Get prediction
        result = await run_prediction(contents)
        
        # Get clinical guidance
        guidance = CLINICAL_GUIDANCE.get(result["prediction"], {})
        
        # Prepare response
        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "is_confident": result["is_confident"],
            "summary": guidance.get("summary", ""),
            "treatment": guidance.get("treatment", ""),
            "follow_up": guidance.get("follow_up", ""),
            "raw_score": result["raw_score"]
        }
        
        # Add warning for low confidence
        if not result["is_confident"]:
            response["warning"] = "Low confidence prediction. Manual review recommended."
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "input_shape": list(IMG_SIZE) + [3],
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("API shutdown complete")
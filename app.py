from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
import logging
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Malaria Detection API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)
PORT = int(os.environ.get("PORT", 8000))

# Load ONNX model
MODEL_PATH = os.environ.get("MODEL_PATH", "model1_cnn.onnx")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    logger.info("ONNX model loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL_LOADED = False
    session = None

IMG_SIZE = (32, 32)
CONFIDENCE_THRESHOLD = 0.65

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
        • Supportive care for symptom relief when okay
        """,
        "follow_up": "Return if symptoms worsen or persist beyond 48 hours"
    }
}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model inference"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image format: {str(e)}")

async def run_prediction(image_bytes: bytes) -> Dict[str, Any]:
    """Run model prediction asynchronously"""
    if not MODEL_LOADED or session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    loop = asyncio.get_event_loop()
    
    def predict():
        img_array = preprocess_image(image_bytes)
        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: img_array})[0][0][0]
        return float(prediction)
    
    try:
        prediction = await loop.run_in_executor(executor, predict)
        
        if prediction > 0.5:
            label = "Uninfected"
            confidence = float(prediction)
        else:
            label = "Parasitized"
            confidence = float(1 - prediction)
        
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

# Serve static files (if you have any CSS/JS files in static folder)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve index.html from the same directory as app.py"""
    try:
        # Get the directory where app.py is located
        current_dir = Path(__file__).parent
        index_path = current_dir / "index.html"
        
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            logger.warning(f"index.html not found at {index_path}")
            return HTMLResponse(content="""
            <html>
                <body>
                    <h1>Malaria Detection API</h1>
                    <p>API is running but index.html not found.</p>
                    <p>Expected location: {}</p>
                </body>
            </html>
            """.format(index_path))
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return HTMLResponse(content=f"<h1>Error loading frontend</h1><p>{str(e)}</p>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "version": "2.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        result = await run_prediction(contents)
        guidance = CLINICAL_GUIDANCE.get(result["prediction"], {})
        
        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "is_confident": result["is_confident"],
            "summary": guidance.get("summary", ""),
            "treatment": guidance.get("treatment", ""),
            "follow_up": guidance.get("follow_up", ""),
            "raw_score": result["raw_score"]
        }
        
        if not result["is_confident"]:
            response["warning"] = "Low confidence prediction. Manual review recommended."
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("API shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

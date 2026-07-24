from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import os

from database import get_db, engine, Base
from models import PredictionRecord
from llm_utils import generate_llm_medical_report

# Ensure Database tables are initialized
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform API",
    version="1.0.0",
    description="REST API providing PyTorch Deep Learning Pneumonia Diagnostics, LLM Medical Reports, and Prediction History."
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING LOGIC ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/pneumonia_resnet18.pth"  # Path to saved weights

def load_trained_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    
    # Matching the exact layer architecture saved in state_dict:
    # fc.0 -> Linear, fc.1 -> ReLU, fc.2 -> Dropout, fc.3 -> Linear
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ PyTorch Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model file {MODEL_PATH} not found in root. Using uninitialized weights for API testing.")
        
    model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- API ENDPOINTS ---

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Advanced AI Medical Intelligence Platform",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(DEVICE)}

@app.post("/api/v1/predict")
async def predict_xray(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts an X-ray image file, performs PyTorch ResNet18 inference,
    generates an LLM diagnostic summary, and logs the result into SQLite.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_img = transform(image).unsqueeze(0).to(DEVICE)

        # PyTorch Inference
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        normal_prob = float(probabilities[0] * 100)
        pneumonia_prob = float(probabilities[1] * 100)
        
        predicted_class = "PNEUMONIA" if pneumonia_prob > normal_prob else "NORMAL"
        confidence = max(normal_prob, pneumonia_prob)

        # Generate LLM Report
        llm_summary = generate_llm_medical_report(
            prediction=predicted_class,
            confidence=confidence,
            normal_prob=normal_prob,
            pneumonia_prob=pneumonia_prob
        )

        # Save entry into Database
        record = PredictionRecord(
            filename=file.filename,
            prediction=predicted_class,
            confidence=confidence,
            normal_probability=normal_prob,
            pneumonia_probability=pneumonia_prob,
            llm_summary=llm_summary
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "id": record.id,
            "filename": record.filename,
            "prediction": record.prediction,
            "confidence": round(record.confidence, 2),
            "probabilities": {
                "NORMAL": round(record.normal_probability, 2),
                "PNEUMONIA": round(record.pneumonia_probability, 2)
            },
            "llm_report": record.llm_summary,
            "timestamp": record.created_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

@app.get("/api/v1/history")
def get_prediction_history(limit: int = 10, db: Session = Depends(get_db)):
    """Fetch past prediction logs stored in the SQLite database."""
    records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "prediction": r.prediction,
            "confidence": round(r.confidence, 2),
            "normal_probability": round(r.normal_probability, 2),
            "pneumonia_probability": round(r.pneumonia_probability, 2),
            "llm_summary": r.llm_summary,
            "timestamp": r.created_at.isoformat()
        }
        for r in records
    ]
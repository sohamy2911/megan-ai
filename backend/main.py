from pathlib import Path
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml" / "models" / "sentiment_pipeline.joblib"

app = FastAPI(
    title="MEGAN AI API",
    description="Machine-learning sentiment intelligence API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze",
    )


@app.get("/")
def home():
    return {
        "name": "MEGAN AI",
        "status": "online",
        "message": "Sentiment Intelligence API",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: SentimentRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python ml/train.py first.",
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Please enter non-empty text.",
        )

    prediction = int(model.predict([text])[0])

    probabilities = model.predict_proba([text])[0]
    classes = list(model.named_steps["classifier"].classes_)
    probability_map = {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    confidence = probability_map.get(prediction, 0.0)

    return {
        "project": "MEGAN AI",
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "negative": round(probability_map.get(0, 0.0) * 100, 2),
            "positive": round(probability_map.get(1, 0.0) * 100, 2),
        },
        "character_count": len(text),
        "note": "Confidence is the model's estimated class probability, not a guarantee of correctness.",
    }

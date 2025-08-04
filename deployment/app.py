# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import shap
import numpy as np
import json

from scraper import extract_article

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API", description="Detects misinformation in news articles using RoBERTa")

# Load model and tokenizer
MODEL_PATH = "./roberta_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Input schema
class ArticleRequest(BaseModel):
    url: str

# SHAP explainer
def explain_prediction(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Move model to GPU if available
    model.to(device)

    # SHAP explainer
    explainer = shap.Explainer(
        lambda x: torch.softmax(model(torch.tensor(x).to(device), attention_mask=torch.tensor(
            np.tile(attention_mask.cpu().numpy(), (len(x), 1))
        ).to(device)).logits.cpu().detach().numpy(), axis=1),
        tokenizer
    )
    shap_values = explainer([text])
    return shap_values

@app.post("/predict", response_model=Dict)
async def predict(request: ArticleRequest):
    url = request.url.strip()

    # Scrape article
    article = extract_article(url)
    if "error" in article:
        raise HTTPException(status_code=400, detail=article["error"])

    title = article["title"]
    body = article["body"]
    processed_text = article["spacy_text"]

    if not processed_text:
        raise HTTPException(status_code=400, detail="No readable content extracted from the article.")

    # Predict
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])

    # Interpretation via SHAP (only for explanation; may be slow)
    try:
        shap_explainer = shap.Explainer(
            lambda x: torch.softmax(model(torch.tensor(x).to(device), attention_mask=torch.ones_like(torch.tensor(x))).logits.cpu().detach().numpy(), axis=1),
            tokenizer
        )
        shap_values = shap_explainer([processed_text])
        shap_exp = shap_values[0].values.tolist()  # Simplified for JSON
    except Exception as e:
        shap_exp = f"Explanation failed: {str(e)}"

    result = {
        "url": url,
        "title": title,
        "prediction": "Real (Credible)" if pred_label == 1 else "Fake (Misinformation)",
        "confidence": confidence,
        "probabilities": {
            "fake": float(probs[0]),
            "real": float(probs[1])
        },
        "explanation": shap_exp  # Note: Full SHAP visualization not returned here
    }

    return result

@app.get("/")
async def root():
    return {"message": "Fake News Detection API is running. Use POST /predict with JSON {'url': '...'}"}
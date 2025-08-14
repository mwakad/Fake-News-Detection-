# main.py (Backend)

# Imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Union
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer
from scraper import extract_article 

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API", description="Detects misinformation in news articles using RoBERTa")

# Load model and tokenizer
MODEL_PATH = "./roberta_model" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
try:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval() 
except Exception as e:
    print(f"Error loading model or tokenizer from {MODEL_PATH}: {e}")
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Input schema for the API request
class ArticleRequest(BaseModel):
    url: str

# Define LIME-compatible predictor function
def lime_predictor(texts: List[str]) -> np.ndarray:
    # Tokenize batch of texts
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**encodings)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
    
    return probs

# Initialize LIME explainer
class_names = ['Fake Article (Label 0)', 'Real Article (Label 1)']
lime_explainer = LimeTextExplainer(class_names=class_names)

@app.post("/predict", response_model=Dict)
async def predict(request: ArticleRequest):
    url = request.url.strip()

    # Scrape article content
    article = extract_article(url)
    if "error" in article:
        raise HTTPException(status_code=400, detail=article["error"])

    title = article["title"]
    body = article["body"]
    processed_text = article["spacy_text"]

    if not processed_text:
        raise HTTPException(status_code=400, detail="No readable content extracted or processed from the article.")

    # Perform prediction with the finetuned RoBERTa model
    inputs = tokenizer(
        processed_text,
        return_tensors="pt", 
        truncation=True,    
        padding=True,        
        max_length=512    
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
     
        # Apply prediction threshold rules
        pred_label = None
        prediction_str = ""
        confidence = max(fake_prob, real_prob)

        if 0.40 <= fake_prob <= 0.45 and 0.55 <= real_prob <= 0.60:
            prediction_str = "Flagged for Human Review"
        elif fake_prob > 0.45: 
            pred_label = 0
            prediction_str = class_names[pred_label]
            confidence = fake_prob
        elif real_prob > 0.60: 
            pred_label = 1
            prediction_str = class_names[pred_label]
            confidence = real_prob
        else:
            prediction_str = "Uncertain Prediction (Thresholds Not Met)"

    # Prepare result dictionary 
    result = {
        "url": url,
        "title": title,
        "prediction": prediction_str,
        "confidence": confidence,
        "probabilities": {
            "fake": fake_prob,
            "real": real_prob
        },
        "processed_text": processed_text,
        "pred_label": pred_label
    }

    return result

class ExplainRequest(BaseModel):
    processed_text: str
    pred_label: Union[int, None]

@app.post("/explain", response_model=Dict)
async def explain(request: ExplainRequest):
    processed_text = request.processed_text
    pred_label = request.pred_label
    top_features: List[Tuple[str, float]] = []   
    if pred_label in [0, 1]:
        try:
            explanation = lime_explainer.explain_instance(
                processed_text,
                classifier_fn=lime_predictor,
                num_features=10,
                num_samples=500,
                labels=[pred_label]
            )
            raw_features = explanation.as_list(label=pred_label)
            for feature_item in raw_features:
                if isinstance(feature_item, (list, tuple)) and len(feature_item) == 2:
                    try:
                        word = str(feature_item[0])
                        weight = float(feature_item[1])
                        top_features.append((word, weight))
                    except (ValueError, TypeError):
                        print(f"Skipping malformed LIME feature item during type conversion: {feature_item}")
                else:
                    print(f"Skipping malformed LIME feature item (not a 2-tuple): {feature_item}")
            if not top_features:
                top_features = [("No specific features found for this prediction.", 0.0)]
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            top_features = [(f"LIME explanation failed: {str(e)}", 0.0)]
    else:
        top_features = [("No prediction passed the thresholds for explanation.", 0.0)]
         
    return {"explanation": top_features}

# Root endpoint 
@app.get("/")
async def root():
    return {"message": "Fake News Detection API is running. Use POST /predict with JSON {'url': '...'}"}
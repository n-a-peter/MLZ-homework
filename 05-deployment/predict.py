import pickle
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="lead-conversion")

# load the pipeline with pickle
with open("pipeline_v2.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict_lead(customer: Dict[str, Any]):
    lead_score = pipeline.predict_proba(customer)[0, 1]
    lead_score = float(lead_score)
    return {
        "conversion_probability": lead_score,
        "converted" : bool(lead_score >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from typing import Annotated
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model_path = os.path.join(os.path.dirname(__file__), '../models/flan-t5-small-finetuned/processed_data_clean')
prefix = "summarize: "

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

class impressionBody(BaseModel):
    findingsText: Annotated[str, StringConstraints(max_length=256)]

@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health():
    return {"message": "API up and running"}

@app.post("/impression/")
def get_impression(requestBody: impressionBody):
    try:
        input_ids = tokenizer(prefix + requestBody.findingsText, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**input_ids)
        
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"impression": text}
    except Exception as e:
        return {"exception": e}

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

model_path = "../models/flan-t5-small-finetuned/processed_data_clean"
prefix = "summarize: "

class impressionBody(BaseModel):
    findingsText: str

@app.get("/")
async def root():
    return {"message": "API up and running"}

@app.post("/impression/")
async def get_impression(requestBody: impressionBody):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    input_ids = tokenizer(prefix + requestBody.findingsText, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text

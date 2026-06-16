from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from typing import Annotated
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import torch
import os

app = FastAPI()

api_dir = Path(__file__).resolve().parent
static_dir = api_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

repo_model_dir = api_dir.parent / "models/flan-t5-small-finetuned/processed_data_clean"
volume_model_dir = Path(os.getenv("MODEL_DIR", "/models/flan-t5-small-finetuned/processed_data_clean"))
prefix = "summarize: "


def _is_real_model_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1_000_000


def resolve_model_path() -> Path:
    volume_model_file = volume_model_dir / "model.safetensors"
    repo_model_file = repo_model_dir / "model.safetensors"
    print(volume_model_file)

    if _is_real_model_file(volume_model_file):
        runtime_model_dir = Path("/tmp/radiology-model")
        runtime_model_dir.mkdir(parents=True, exist_ok=True)

        for source in repo_model_dir.iterdir():
            target = runtime_model_dir / source.name
            if target.exists() or target.is_symlink():
                target.unlink()
            if source.name != "model.safetensors":
                target.symlink_to(source)

        (runtime_model_dir / "model.safetensors").symlink_to(volume_model_file)
        return runtime_model_dir

    if _is_real_model_file(repo_model_file):
        return repo_model_dir

    raise RuntimeError(
        "Model weights not found. Mount the Railway volume at /models with "
        "flan-t5-small-finetuned/processed_data_clean/model.safetensors, "
        "or set MODEL_DIR to the mounted model directory."
        f"{repo_model_dir}"
    )


model_path = resolve_model_path()
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

class impressionBody(BaseModel):
    findingsText: Annotated[str, StringConstraints(max_length=256)]

@app.get("/")
def root():
    return FileResponse(static_dir / "index.html")


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

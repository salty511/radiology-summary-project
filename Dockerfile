FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r api/requirements.txt

COPY . .

RUN model_file="models/flan-t5-small-finetuned/processed_data_clean/model.safetensors" \
    && test -f "$model_file" \
    && test "$(wc -c < "$model_file")" -gt 1000000 \
    || (echo "Missing real model file. Ensure Git LFS files are available during the Railway build." && exit 1)

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

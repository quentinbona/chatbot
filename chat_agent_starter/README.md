# TensorFlow Chat Agent Starter

A starter Python repository for an MVP AI chat agent with:

- FastAPI backend
- TensorFlow / Keras intent model
- Retrieval-based response fallback
- Short-term conversation memory
- SQLite transcript storage
- Versioned model export and inference loading

This starter is intentionally narrow-scope and practical. It is meant to be a foundation you can extend into:
- domain assistants
- FAQ bots
- support copilots
- internal knowledge chat layers

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m ml.train
python -m app.main
```

Open the API docs at:
- http://127.0.0.1:8000/docs

## Example chat request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user",
    "message": "How do I reset my password?"
  }'
```

## Project goals

This project optimizes for:
- simplicity
- local development
- clear separation between app and ML layers
- easy retraining and redeployment

It does **not** try to train a frontier LLM from scratch.

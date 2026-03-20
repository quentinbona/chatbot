# Technical Architecture Document
## TensorFlow Chat Agent MVP

## 1. Purpose

This system is a production-minded MVP for a Python-based AI chat agent using TensorFlow as the training and inference framework. The design intentionally favors a narrow, reliable architecture over an oversized experimental stack.

The reference implementation in this repository uses:

- **FastAPI** for the API surface
- **TensorFlow / Keras** for text classification
- **SQLite** for transcript persistence
- **Simple response retrieval** for deterministic responses
- **Config-driven deployment** through environment variables

The result is an end-to-end system that can train, export, load, serve, and log a conversational assistant.

---

## 2. MVP Scope

### Included
- REST chat API
- Local model training
- Exported TensorFlow SavedModel
- Short-term memory through recent transcript lookup
- SQLite persistence
- Intent prediction plus fallback logic
- Starter tests and Docker support

### Explicitly not included
- Frontier-scale generative LLM training
- Tool-use orchestration
- Web browsing agents
- Vector database memory
- Multi-tenant auth and billing
- Human feedback loops such as RLHF
- Distributed training or GPU scheduling

---

## 3. Product Definition

The MVP behaves as a **domain chatbot**.

A user sends a message to `/chat`.  
The backend loads recent context from SQLite.  
The TensorFlow model predicts the user intent.  
The service returns a deterministic reply associated with that intent, unless confidence is below threshold, in which case it returns a fallback response.

This is a strong MVP pattern because it is:

- easy to train
- inexpensive to host
- easy to evaluate
- easy to extend into retrieval or generation later

---

## 4. High-Level Architecture

```text
[Client / UI]
     |
     v
[FastAPI App]
     |
     +--> [Chat Service]
     |        |
     |        +--> [Transcript Store / SQLite]
     |        |
     |        +--> [Model Manager]
     |                   |
     |                   +--> [TensorFlow SavedModel]
     |
     +--> [/train endpoint]
              |
              v
         [Training Pipeline]
              |
              +--> preprocess.py
              +--> dataset.py
              +--> model.py
              +--> evaluate.py
              +--> export.py
```

---

## 5. Architectural Layers

## 5.1 Interface Layer

The interface layer can be any of the following:

- web frontend
- mobile frontend
- CLI
- third-party system calling the API

The repository currently exposes a REST API only.

### Key endpoints
- `GET /health`
- `GET /model-info`
- `POST /train`
- `POST /chat`

This keeps the first version simple and operable.

---

## 5.2 Application Layer

The application layer lives in `app/`.

### Responsibilities
- request validation
- inference orchestration
- transcript read/write
- config loading
- response shaping
- runtime model management

### Components

#### `app/main.py`
Entrypoint for the FastAPI service. Defines all public endpoints and wires together services.

#### `app/chat_service.py`
Core orchestration logic. Applies business rules like:
- how many context turns to fetch
- when to trigger fallback
- how to store user and assistant messages

#### `app/memory.py`
Provides SQLite transcript persistence. Handles:
- initialization
- inserts
- recent message lookup
- context formatting

#### `app/inference.py`
Loads the exported TensorFlow model and performs predictions. Also loads response metadata so the app can map predicted intents to final replies.

#### `app/config.py`
Defines all environment-driven settings in one place.

#### `app/schemas.py`
Declares request and response contracts using Pydantic.

---

## 5.3 ML Layer

The ML layer lives in `ml/`.

### Responsibilities
- load training data
- encode labels
- split train and validation sets
- define the TensorFlow model
- train and evaluate
- export the model and metadata

### Components

#### `ml/preprocess.py`
Loads raw JSON training data and returns:
- text samples
- labels
- response dictionary

#### `ml/dataset.py`
Handles label encoding and train/validation splitting.

#### `ml/model.py`
Defines the TensorFlow Keras network.

Current network structure:
- string input
- `TextVectorization`
- `Embedding`
- `GlobalAveragePooling1D`
- dense hidden layer
- dropout
- softmax output

This is intentionally small, cheap, and easy to retrain.

#### `ml/evaluate.py`
Generates a scikit-learn classification report.

#### `ml/export.py`
Exports a TensorFlow SavedModel and writes a `metadata.json` file containing:
- class labels
- intent-to-response mapping

#### `ml/train.py`
Runs the full training pipeline.

---

## 5.4 Data Layer

### Training data
Stored in `data/raw/training_data.json`.

It contains:
- labeled text examples
- response templates per intent
- fallback responses

### Transcript data
Stored in SQLite at:
- `data/transcripts/chat_agent.db`

The transcript schema is intentionally simple:

```sql
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

This is sufficient for:
- short-term context
- debugging
- analytics seeds
- future supervised fine-tuning data

---

## 6. Training Flow

## 6.1 Input
The training job reads labeled examples from JSON.

## 6.2 Preprocessing
- load texts
- load intent labels
- encode class ids
- split into train and validation sets

## 6.3 Vectorization
The model includes a `TextVectorization` layer so the text preprocessing graph is embedded into the model itself.

Benefits:
- simpler serving
- no external tokenizer dependency
- lower production drift risk between training and inference

## 6.4 Model Training
The model is trained with:
- optimizer: Adam
- loss: sparse categorical crossentropy
- metric: accuracy
- callback: early stopping

## 6.5 Evaluation
Validation predictions are converted into a classification report for offline review.

## 6.6 Export
The trained model is exported to `models/latest/` and accompanied by metadata.

---

## 7. Runtime Inference Flow

```text
User message
   |
   v
/chat endpoint
   |
   v
ChatService.handle_message()
   |
   +--> fetch recent context from SQLite
   +--> predict intent with TensorFlow model
   +--> compare confidence to threshold
   +--> choose intent response or fallback
   +--> save user + assistant messages
   +--> return structured API response
```

### Example response fields
- `reply`
- `predicted_intent`
- `confidence`
- `used_fallback`
- `context_messages`

This structure is useful for:
- debugging
- admin consoles
- evaluation dashboards

---

## 8. Folder Structure

```text
chat_agent_starter/
├── .env.example
├── Dockerfile
├── README.md
├── TECHNICAL_ARCHITECTURE.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── chat_service.py
│   ├── config.py
│   ├── inference.py
│   ├── main.py
│   ├── memory.py
│   └── schemas.py
├── data/
│   └── raw/
│       └── training_data.json
├── ml/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── export.py
│   ├── model.py
│   ├── preprocess.py
│   └── train.py
└── tests/
    ├── test_inference_placeholder.py
    └── test_memory.py
```

---

## 9. Key Design Decisions

## 9.1 Why intent classification first
A full generative chat model is not the most practical MVP. Intent classification is dramatically simpler to:
- train
- debug
- evaluate
- trust in production

It also creates a clean upgrade path toward:
- retrieval-augmented responses
- small generative decoders
- external tool routing

## 9.2 Why SQLite
SQLite is enough for an MVP and reduces:
- setup friction
- deployment complexity
- local development overhead

Later upgrades can move transcript storage into Postgres or another operational store.

## 9.3 Why `TextVectorization`
Keeping vectorization inside the TensorFlow graph avoids a common production bug where training and serving tokenize text differently.

## 9.4 Why deterministic responses
For an MVP, deterministic responses are easier to review and less likely to hallucinate than generated text.

---

## 10. Security and Operational Considerations

### Current baseline
- no authentication
- no rate limiting
- no moderation layer
- local storage only

### Recommended next upgrades
- API key or OAuth auth
- request logging and correlation ids
- rate limits
- abuse filtering
- secrets management
- structured logging
- centralized metrics

---

## 11. Observability Recommendations

For the next version, track:
- request count
- latency by endpoint
- model confidence distribution
- fallback rate
- transcript volume
- per-intent frequency
- training version currently loaded

This data helps identify:
- weak intents
- insufficient training coverage
- broken deployments
- support topic drift

---

## 12. Testing Strategy

### Unit tests
Focus on:
- transcript insert and retrieval
- config parsing
- label handling
- response selection logic

### Integration tests
Focus on:
- `/train` successfully exports a model
- `/chat` returns a valid response once the model is loaded
- model metadata is loaded correctly

### Offline evaluation
Review:
- confusion matrix
- per-class recall
- fallback trigger rate
- hard negative examples

---

## 13. Extension Roadmap

## Phase 2
- larger dataset
- better intent coverage
- richer response templates
- admin training dashboard
- Postgres storage
- confidence calibration

## Phase 3
- semantic retrieval over knowledge documents
- embedding search
- answer citation support
- session analytics

## Phase 4
- hybrid routing between retrieval and generation
- tool invocation
- human handoff workflow
- role-based access control

---

## 14. Failure Modes

Common failure cases:

### Low confidence predictions
Handled by fallback response.

### Missing exported model
Handled by returning `503` from `/chat` until training occurs.

### Sparse training data
Leads to unstable class boundaries and overconfidence. Fix by improving coverage and adding adversarial examples.

### Transcript DB corruption
Mitigate with backups or migration to a stronger operational store for production.

---

## 15. Deployment Model

### Local development
- create virtual environment
- install requirements
- run `python -m ml.train`
- run `python -m app.main`

### Container deployment
The repository includes a simple Dockerfile. In production, place it behind:
- reverse proxy
- HTTPS termination
- process supervision
- external logging

---

## 16. Suggested Next Refactors

To mature this codebase:

1. separate training and serving images
2. add model version directories with active symlink
3. add migrations for DB schema
4. add structured logs
5. add richer response generation layer
6. add confidence calibration and threshold tuning
7. replace JSON training data with a data contract and validation pipeline

---

## 17. Final MVP Definition

This repository implements a complete MVP architecture for a TensorFlow-based AI chat agent in Python:

- training pipeline
- model export
- model loading
- inference service
- transcript persistence
- deployment entrypoint

It is intentionally small, but it is a true end-to-end architecture rather than a toy script.

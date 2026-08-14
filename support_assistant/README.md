# Zepto Support Assistant

## Overview

This module implements a small Retrieval-Augmented Generation (RAG) customer-support assistant for Zepto.

The application uses:

- Sentence Transformers for local embeddings
- ChromaDB for vector storage and similarity retrieval
- LangGraph for intent routing and orchestration
- Pydantic for structured response validation
- FastAPI for the HTTP API
- A deterministic offline mock LLM mode for the graded baseline
- Docker for containerization

The required graded path does not require an API key or an LLM provider.

---

## 1. Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 2. Architecture

The application follows this RAG pipeline:

```text
User Query
    |
    v
FastAPI POST /ask
    |
    v
LangGraph StateGraph
    |
    v
classify_intent
    |
    +-----------------------------+
    |                             |
    v                             v
policy_question              general_question
    |                             |
    v                             v
retrieve_and_answer          direct_answer
    |                             |
    v                             |
Query embedding                  |
    |                             |
    v                             |
ChromaDB similarity search       |
    |                             |
    v                             |
Top 3 policy chunks              |
    |                             |
    +-------------+---------------+
                  |
                  v
            Pydantic validation
                  |
                  v
             JSON response
```

### Ingestion

The eight Zepto policy documents are stored in the `docs/` directory.

The `ingest_documents()` function in `main.py` loads the `.txt` files.

For this small corpus, one document is treated as one chunk.

Each chunk receives an ID such as `doc_01_chunk_01`.

### Embedding

The application uses the open-source Sentence Transformers model:

`all-MiniLM-L6-v2`

The model generates embeddings for all policy chunks and for incoming queries.

### Vector Storage

ChromaDB stores the embeddings in the collection:

`zepto_policies`

Cosine similarity is used for retrieval.

### Retrieval

The `retrieve_and_answer()` LangGraph node performs retrieval.

For a policy question, the query is embedded and the top three most similar chunks are retrieved from ChromaDB.

### Generation

In the required mock mode, no external LLM is called.

The `retrieve_and_answer()` node returns an answer beginning with:

`Based on the retrieved context:`

The `direct_answer()` node returns the fixed mock response for general questions.

---

## 3. LangGraph

The LangGraph `StateGraph` contains three required nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

The graph starts at `classify_intent`.

A conditional edge routes policy questions to `retrieve_and_answer` and general questions to `direct_answer`.

---

## 4. Intent Classification

In mock mode, classification uses a deterministic keyword heuristic.

Policy keywords:

`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`

If one of these keywords occurs in the lowercased query, the query becomes `policy_question`.

Otherwise it becomes `general_question`.

No LLM call is made during this classification in mock mode.

---

## 5. Structured Prompt

The application contains a structured prompt template using the required:

- ROLE
- CONTEXT
- TASK
- FORMAT
- LENGTH

The prompt also includes an explicit negative constraint and a few-shot example.

The negative constraint instructs the model not to use information outside the supplied Zepto policy context.

The prompt is used by the optional real-LLM path.

---

## 6. MOCK_LLM

The required graded baseline uses `MOCK_LLM=1`, or leaves the variable unset.

In mock mode:

- intent classification is deterministic
- embeddings are generated locally
- ChromaDB retrieval is real
- policy answer generation is deterministic
- general answer generation is deterministic
- no LLM API call is made
- no LLM API key is required

The optional real-LLM branch can be selected with `MOCK_LLM=0`.

---

## 7. Structured Output

The final response is validated using the Pydantic `ZeptoResponse` model.

It contains exactly three fields:

- `answer` — string
- `sources` — list of chunk/document IDs
- `confidence` — float between 0 and 1

Mock mode uses a deterministic confidence value of `1.0`.

Policy questions contain retrieved chunk IDs in `sources`.

General questions return an empty `sources` list.

The optional real-LLM branch retries validation up to two additional times if the raw LLM response does not match the schema.

---

## 8. FastAPI

The application exposes:

`POST /ask`

Request:

```json
{"query": "How much does delivery cost?"}
```

---

## 9. Example API Calls

The following calls were run using the required offline mock mode.

### Example 1 — Policy Question

Request:

```json
{"query": "How much does delivery cost?"}
```

Raw response:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01_chunk_01","doc_05_chunk_01","doc_02_chunk_01"],"confidence":1.0}
```

This query contains the keyword `delivery`, so it is routed to `retrieve_and_answer`.

### Example 2 — General Question

Request:

```json
{"query": "What is the capital of India?"}
```

Raw response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

This query does not contain a policy keyword, so it is routed to `direct_answer`.

---

## 10. Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

The API will be available on port 7860.

---

## 11. Docker

Build:

```bash
docker build -t zepto-support-assistant .
```

Run:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

The Dockerfile starts the FastAPI application with Uvicorn on port 7860.

The standard Google Colab runtime used during development did not provide a Docker daemon, so Docker build/run was not executed inside Colab. The Dockerfile is included and configured for a local Docker environment.

---

## 12. Offline Graded Baseline

The required graded baseline is the deterministic mock mode.

`MOCK_LLM` unset or `MOCK_LLM=1`

No LLM API key or LLM provider network access is required.

Embeddings use `all-MiniLM-L6-v2` locally and retrieval uses ChromaDB.

---

## 13. Optional Real-LLM Extension

The code contains a separate `MOCK_LLM=0` branch for optional real-LLM practice.

The optional branch uses the structured prompt and validates the generated JSON using Pydantic.

If validation fails, the implementation retries up to two additional times with corrective instructions.

This extension is not required for the graded baseline.

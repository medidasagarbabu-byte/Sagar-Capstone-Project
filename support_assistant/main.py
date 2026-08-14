
import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from fastapi import FastAPI
from langgraph.graph import StateGraph, END


# ============================================================
# CONFIGURATION
# ============================================================

# Graded baseline:
# MOCK_LLM unset or set to "1" -> deterministic offline mock mode
# MOCK_LLM="0" -> optional real-LLM mode
MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def ingest_documents():
    """
    Load the 8 Zepto policy documents, create one chunk per
    document, embed them and store them in ChromaDB.
    """

    existing = collection.count()

    if existing >= 8:
        return

    documents = []
    ids = []
    metadatas = []

    for filename in sorted(os.listdir(DOCS_DIR)):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(DOCS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        # One document = one chunk
        chunk_id = filename.replace(".txt", "_chunk_01")

        documents.append(text)
        ids.append(chunk_id)

        metadatas.append({
            "source": filename
        })

    if documents:

        embeddings = embedding_model.encode(
            documents
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )


# Run ingestion when the application starts
ingest_documents()


# ============================================================
# PYDANTIC RESPONSE MODEL
# ============================================================

class ZeptoResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# FASTAPI REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    query: str


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ZeptoState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# STRUCTURED PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant.
You answer customer questions using only the provided Zepto policy context.

CONTEXT:
{context}

TASK:
Answer the user's question using the supplied context.
If the answer is not present in the context, clearly say that the
provided Zepto policy context does not contain the answer.

FORMAT:
Return a JSON object with exactly these fields:
- answer: a string containing the answer
- sources: a list of document or chunk IDs used
- confidence: a number between 0 and 1

LENGTH:
Keep the answer concise and preferably under 100 words.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, deadlines, fees, or support procedures.

FEW-SHOT EXAMPLE:

Question:
What is the delivery fee for orders below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee.

Example Answer:
{{
  "answer": "Orders below INR 149 incur a flat INR 25 delivery fee.",
  "sources": ["doc_01_chunk_01"],
  "confidence": 1.0
}}

USER QUESTION:
{question}
"""


# ============================================================
# OPTIONAL REAL LLM HELPER
# ============================================================

def real_llm_call(prompt: str) -> str:
    """
    Optional extension for MOCK_LLM=0.

    The graded baseline never calls this function.

    To keep the required submission fully offline, this function
    deliberately raises an error unless a real LLM backend is
    implemented by the student.
    """

    raise RuntimeError(
        "Real LLM mode is optional. "
        "Set MOCK_LLM=1 for the required offline graded mode."
    )


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(state: ZeptoState) -> ZeptoState:

    query = state["query"]

    # Required deterministic mock behavior
    if MOCK_LLM != "0":

        policy_keywords = [
            "delivery",
            "return",
            "refund",
            "membership",
            "tracking",
            "cancel",
            "gift card",
            "support hours"
        ]

        lower_query = query.lower()

        if any(
            keyword in lower_query
            for keyword in policy_keywords
        ):
            intent = "policy_question"
        else:
            intent = "general_question"

        return {
            **state,
            "intent": intent
        }

    # Optional real LLM path
    prompt = f"""
Classify the following user question as exactly one of:
policy_question
general_question

Question:
{query}

Return only the classification.
"""

    raw_output = real_llm_call(prompt)

    cleaned = raw_output.strip()

    if cleaned not in [
        "policy_question",
        "general_question"
    ]:
        cleaned = "general_question"

    return {
        **state,
        "intent": cleaned
    }


# ============================================================
# RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: ZeptoState) -> ZeptoState:

    query = state["query"]

    # Retrieval always runs in BOTH modes.
    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_ids = results["ids"][0]

    if not retrieved_documents:
        return {
            **state,
            "answer": "No relevant Zepto policy context was found.",
            "sources": [],
            "confidence": 0.0
        }

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM != "0":

        top_chunk = retrieved_documents[0]

        top_chunk_snippet = top_chunk[:200]

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

        return {
            **state,
            "answer": answer,
            "sources": retrieved_ids,
            "confidence": 1.0
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    context_parts = []

    for chunk_id, document in zip(
        retrieved_ids,
        retrieved_documents
    ):
        context_parts.append(
            f"[{chunk_id}]\n{document}"
        )

    context = "\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    # Retry up to 2 additional times if validation fails
    last_error = None

    for attempt in range(3):

        try:

            raw_output = real_llm_call(prompt)

            # Expected real LLM output would be JSON.
            import json

            parsed = json.loads(raw_output)

            validated = ZeptoResponse(**parsed)

            return {
                **state,
                "answer": validated.answer,
                "sources": validated.sources,
                "confidence": validated.confidence
            }

        except Exception as e:

            last_error = str(e)

            prompt = f"""
Your previous answer failed JSON/schema validation.

Corrective instruction:
Return ONLY valid JSON with exactly these fields:
answer (string)
sources (list of strings)
confidence (number between 0 and 1)

Do not include markdown.
Do not include explanations.

Original prompt:
{prompt}
"""

    return {
        **state,
        "answer": (
            "ERROR: Real LLM output failed schema validation "
            f"after 3 attempts. {last_error}"
        ),
        "sources": [],
        "confidence": 0.0
    }


# ============================================================
# DIRECT ANSWER
# ============================================================

def direct_answer(state: ZeptoState) -> ZeptoState:

    # Required mock mode
    if MOCK_LLM != "0":

        return {
            **state,
            "answer": (
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0
        }

    # Optional real LLM mode
    prompt = f"""
Answer the following question directly.

Question:
{state["query"]}

Return a concise answer.
"""

    raw_output = real_llm_call(prompt)

    return {
        **state,
        "answer": raw_output,
        "sources": [],
        "confidence": 1.0
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_intent(state: ZeptoState) -> str:

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(ZeptoState)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.set_entry_point(
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

zepto_graph = graph_builder.compile()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline mock RAG support assistant for Zepto policies",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running",
        "mock_llm": MOCK_LLM
    }


@app.post(
    "/ask",
    response_model=ZeptoResponse
)
def ask(request: AskRequest):

    result = zepto_graph.invoke({
        "query": request.query
    })

    response = ZeptoResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )

    return response

"""
AI in Science — RAG Research Assistant

Ask questions about the OECD "Artificial Intelligence in Science" (2023) report,
with a live web-search fallback for anything the report doesn't cover.

Retrieval pipeline: BM25 + vector search fused via Reciprocal Rank Fusion
(EnsembleRetriever), then narrowed by a cross-encoder re-ranker
(ContextualCompressionRetriever) down to the final top-k passed to the LLM.

IMPORTANT: this module is imported directly by generate_golden_dataset.py and
evaluate.py (they need build_vector_store / _load_indexed_documents / chunk_id /
final_retriever / ask). Everything above the `if __name__ == "__main__":` guard
runs on import -- that's intentional. The Gradio UI itself must stay inside that
guard: if demo.launch() ran at import time, importing this module from an eval
script (or from CI) would block forever on a live Gradio server instead of
returning control to the caller.
"""

import os
import hashlib
import yaml
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
import gradio as gr

# Loads LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc. from your .env file
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "AIinScience.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_langchain_db")
COLLECTION_NAME = "example_collection"

CHUNK_SIZE_TOKENS = 650
CHUNK_OVERLAP_TOKENS = 100

CANDIDATE_K = 20 # how many chunks bm25 and vector search each return before fusion
FINAL_K = 4  # how many re-ranked results actually get passed to the LLM

# Bump this and add a new entry in prompts.yaml to change the active prompt without
# touching code -- the eval script logs which version produced each run's numbers.
PROMPT_VERSION = "v1.2"

gemini_api_key = os.getenv("GEMINI_API_KEY")

model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite",
    api_key=gemini_api_key,
)


def chunk_id(doc) -> str:
    """Deterministic ID from normalized content + source + page, so re-running
    ingestion never creates duplicate chunks, and the golden eval set can reference
    a chunk by an ID that stays stable across runs."""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content = " ".join(doc.page_content.split())
    value = f"{source}|{page}|{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    existing = store.get(limit=1)
    if existing["ids"]:
        return store

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    all_splits = splitter.split_documents(documents)

    ids = [chunk_id(doc) for doc in all_splits]
    store.add_documents(documents=all_splits, ids=ids)
    return store


def _load_indexed_documents(store: Chroma) -> list[Document]:
    """Reconstruct Document objects from what's already in Chroma, so BM25 is built
    from the exact same chunks as the vector index -- one ingestion path, two
    retrievers."""
    raw = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(raw["documents"], raw["metadatas"])
    ]


def build_hybrid_retriever(store: Chroma) -> EnsembleRetriever:
    """BM25 keyword search + vector search, fused via Reciprocal Rank Fusion."""
    indexed_docs = _load_indexed_documents(store)

    bm25_retriever = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever.k = CANDIDATE_K

    vector_retriever = store.as_retriever(search_kwargs={"k": CANDIDATE_K})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def build_reranking_retriever(hybrid: EnsembleRetriever, final_k: int) -> ContextualCompressionRetriever:
    """Wrap the hybrid retriever with a cross-encoder re-ranker: takes the wide fused
    candidate set and scores each one directly against the query, keeping only the
    top final_k most relevant."""
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=final_k)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)


vector_store = build_vector_store()
hybrid_retriever = build_hybrid_retriever(vector_store)
final_retriever = build_reranking_retriever(hybrid_retriever, FINAL_K)


@tool
def retrieve_from_pdf(query: str) -> str:
    """Retrieve information from the OECD 'Artificial Intelligence in Science' report."""
    try:
        retrieved_docs = final_retriever.invoke(query)
    except Exception as e:
        return (
            f"[RETRIEVAL ERROR] The PDF search failed: {e}. "
            f"Do not answer from report content — tell the user retrieval failed "
            f"and, if relevant, offer to try a web search instead."
        )

    relevant_docs = retrieved_docs[:FINAL_K]

    if not relevant_docs:
        return (
            "[NO RESULTS] No relevant content was found in the 'AI in Science' "
            "report for this query. Do not fabricate an answer from the report — "
            "either say nothing relevant was found, or use TavilySearch if the "
            "question might be answerable from the web."
        )

    docs_content = ""
    for i, doc in enumerate(relevant_docs, start=1):
        page_num = doc.metadata.get("page", 0) + 1  # PyPDFLoader pages are 0-indexed
        docs_content += (
            f"--- Document Chunk {i} ---\n"
            f"Citation Tag: [Page {page_num}]\n"
            f"Content: {doc.page_content}\n\n"
        )
    return docs_content


tavily_api_key = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    tavily_api_key=tavily_api_key,
)


def load_system_prompt(version: str, file_path: str = "prompts.yaml") -> str:
    """Load a specific version of the system prompt from a YAML file, so changing the
    active prompt is a config edit + version bump, not a code change."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return data["versions"][version]["text"]
    except KeyError as e:
        raise ValueError(f"Prompt version '{version}' not found in {file_path}") from e


system_prompt = load_system_prompt(PROMPT_VERSION)

agent = create_agent(
    model=model,
    tools=[retrieve_from_pdf, web_search_tool],
    system_prompt=system_prompt,
)


@traceable(run_type="chain", name="Main Agent Invocation")
def _invoke_agent(query: str):
    """Single place that actually calls the agent -- ask() and ask_with_sources()
    both build on this so evaluation never invokes the agent twice per question
    (which would silently double Gemini calls against the daily quota)."""
    return agent.invoke({"messages": [{"role": "user", "content": query}]})


def _extract_answer_text(response) -> str:
    content = response["messages"][-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(t for t in texts if t) or "[No text content returned by the model]"
    return str(content)


def ask(query: str) -> str:
    return _extract_answer_text(_invoke_agent(query))


def ask_with_sources(query: str) -> dict:
    """Same call as ask(), but also reports which tools actually ran and what they
    returned. This is what evaluate.py should judge faithfulness against -- the
    REAL context that grounded this specific answer, not a separately recomputed
    retrieval call that might use a different query or might not reflect whether
    the agent called retrieve_from_pdf at all."""
    response = _invoke_agent(query)
    answer = _extract_answer_text(response)

    tool_calls = [
        {"tool": msg.name, "output": msg.content}
        for msg in response["messages"]
        if isinstance(msg, ToolMessage)
    ]

    return {"answer": answer, "tool_calls": tool_calls}


demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about AI in Science...", label="Query"),
    outputs=gr.Textbox(lines=10, placeholder="Response will appear here...", label="Response"),
    title="AI in Science Research Assistant",
    description="Ask questions about the 'AI in Science' research paper or recent developments in AI.",
)

if __name__ == "__main__":
    demo.launch()

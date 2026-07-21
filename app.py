import os
import hashlib
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
import gradio as gr

load_dotenv()

CANDIDATE_K = 8  # how many candidates EACH retriever (BM25, vector) contributes before fusion
FINAL_K = 4      # how many fused results actually get passed to the LLM

def chunk_id(doc):
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content = " ".join(doc.page_content.split())  # normalize whitespace

    value = f"{source}|{page}|{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def build_vector_store():

    file_path = "AIinScience.pdf"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db", 
    )

    existing = store.get(limit=1)
    if existing["ids"]:
            return store
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=100
    )

    all_splits = text_splitter.split_documents(documents)
    ids = [chunk_id(doc) for doc in all_splits]

    store.add_documents(
        documents=all_splits,
        ids=ids,
    )
    return store

def _load_indexed_documents(store: Chroma) -> list[Document]:
    """Reconstruct Document objects from what's already sitting in Chroma, so BM25 is
    built from the exact same chunks as the vector index -- one ingestion path, two
    retrievers, rather than re-parsing the PDF a second time just for BM25."""
    raw = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(raw["documents"], raw["metadatas"])
    ]
 
def build_hybrid_retriever(store: Chroma) -> EnsembleRetriever:
    """Combine BM25 keyword search with vector similarity search via Reciprocal Rank
    Fusion. RRF gives each retriever's rank-i document a score of 1/(k_rrf + i) (k_rrf is
    a constant, typically 60) and sums those scores across retrievers -- so a document
    that ranks well in EITHER method surfaces near the top, not just one that wins on
    both. Weights start even; sweep them against the golden eval set in Phase 3 rather
    than hand-tuning now."""
    indexed_docs = _load_indexed_documents(store)
 
    bm25_retriever = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever.k = CANDIDATE_K
 
    vector_retriever = store.as_retriever(search_kwargs={"k": CANDIDATE_K})
 
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

vector_store = build_vector_store()
hybrid_retriever = build_hybrid_retriever(vector_store)

api_key= os.getenv('GEMINI_API_KEY')
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=api_key,
)

@tool
def retrieve_from_pdf(query: str) -> str:
    """Retrieve information from the AI in Science research paper."""
    try:
        retrieved_docs = hybrid_retriever.invoke(query)
    except Exception as e:
        return (
            f"[RETRIEVAL ERROR] The PDF search failed: {e}. "
            f"Do not answer from paper content — tell the user retrieval failed "
            f"and, if relevant, offer to try a web search instead."
        )

    threshold = 0.8  
    relevant_docs = retrieved_docs[:FINAL_K]

    if not relevant_docs:
        return (
            "[NO RESULTS] No relevant content was found in the 'AI in Science' "
            "paper for this query. Do not fabricate an answer from the paper — "
            "either say nothing relevant was found, or use TavilySearch if the "
            "question might be answerable from the web."
        )

    docs_content = ""
    
    for doc in relevant_docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown")
        docs_content += (
            f"Source: {source}, Page: {page}\n"
            f"Content: {doc.page_content}\n\n"
        )
    return docs_content

tavily_api_key = os.getenv('TAVILY_API_KEY')
web_search_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    tavily_api_key=tavily_api_key
)

system_prompt = """You are a helpful research assistant in AI in Science with access to two tools:
1. retrieve_from_pdf: Use this to find information from the
   "AI in Science" research paper
2. TavilySearch: Use this to find current information
   not in the paper (recent events, updates, etc.)
Strategy:
- For questions about the paper content → use retrieve_from_pdf
- For questions about recent events or topics not in the paper → use TavilySearch
"""
agent = create_agent(
    model=model,
    tools=[retrieve_from_pdf, web_search_tool],
    system_prompt=system_prompt
)

def ask(query: str) -> str:
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = response["messages"][-1].content

    # Content can come back as a plain string or as a list of content blocks depending on the model/agent version
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if isinstance(block, dict)]
        joined = "\n".join(t for t in texts if t)
        return joined or "[No text content returned by the model]"
    return str(content)

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about AI in Science...", label="Query"),
    outputs=gr.Textbox(lines=10, placeholder="Response will appear here...", label="Response"),
    title="AI in Science Research Assistant",
    description="Ask questions about the 'AI in Science' research paper or recent developments in AI.",
)

demo.launch()
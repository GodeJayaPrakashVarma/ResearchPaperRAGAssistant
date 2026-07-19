#! pip install -qU  langchain langchain-huggingface sentence_transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
import os

file_path = "AIinScience.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=650,
    chunk_overlap=100
)

all_splits = text_splitter.split_documents(documents)

def chunk_id(doc):
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content = " ".join(doc.page_content.split())  # normalize whitespace

    value = f"{source}|{page}|{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

ids = [chunk_id(doc) for doc in all_splits]

document_ids = vector_store.add_documents(
    documents=all_splits,
    ids=ids,
)

#!pip install -qU langchain-google-genai
#!pip install langchain langchain-tavily
from langchain.chat_models import init_chat_model
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

api_key= os.getenv('GEMINI_API_KEY')
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=api_key,
)

@tool
def retrieve_from_pdf(query: str) -> str:
    """Retrieve information from the AI in Science research paper."""
    try:
        retrieved_docs = vector_store.similarity_search_with_score(query, k=2)
    except Exception as e:
        return (
            f"[RETRIEVAL ERROR] The PDF search failed: {e}. "
            f"Do not answer from paper content — tell the user retrieval failed "
            f"and, if relevant, offer to try a web search instead."
        )

    threshold = 0.8  
    relevant_docs = []

    for doc, score in retrieved_docs:
        if score < threshold:
            relevant_docs.append(doc)

    if not relevant_docs:
        return (
            "[NO RESULTS] No relevant content was found in the 'AI in Science' "
            "paper for this query. Do not fabricate an answer from the paper — "
            "either say nothing relevant was found, or use TavilySearch if the "
            "question might be answerable from the web."
        )

    docs_content = ""
    
    for doc in relevant_docs:
        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"
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

demo.launch(share=True)
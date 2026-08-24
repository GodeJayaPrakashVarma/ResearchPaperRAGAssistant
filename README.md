# AI in Science — RAG Research Assistant

Ask questions about the OECD *Artificial Intelligence in Science* (2023) report and get
answers grounded in specific pages — with a live web-search fallback for anything the
report doesn't cover.

## Problem

Long technical reports are hard to query directly — answers are buried across
hundreds of pages, and generic chat models will happily hallucinate an answer instead of
admitting they don't know. This assistant retrieves the actual supporting passages before
answering, cites the page number for every claim, and explicitly says so when it can't
find support in the report.

## Live demo

- **Try it**: https://huggingface.co/spaces/JayaPrakashVarma/ResearchPaperRAGAssistant

## Architecture

```
User query
   │
   ▼
Agent (Gemini 2.5 Flash) decides which tool to call
   │
   ├── retrieve_from_pdf ─▶ Hybrid retrieval (BM25 keyword + vector search,
   │                        fused via Reciprocal Rank Fusion) ─▶ page-tagged chunks
   │
   └── TavilySearch ─▶ live web results (for anything outside the report)
   │
   ▼
Answer with page citations, via Gradio UI
```

Ingestion (one-time, idempotent): PDF → token-based chunking (500–800 tokens, 100 overlap)
→ embeddings (`sentence-transformers/all-mpnet-base-v2`) → Chroma vector store. The BM25
index is rebuilt from those same stored chunks — one source of truth, two retrieval paths.

## Roadmap

- [x] **Phase 1** — PDF ingestion, chunking, vector store, agentic tool routing
- [x] **Phase 2a** — Idempotent ingestion, token-based chunking, defensive response
      parsing, page-citation enforcement in the system prompt
- [x] **Phase 2b** — Hybrid retrieval: BM25 + vector search combined via RRF
      (`EnsembleRetriever`)
- [x] **Phase 2c** — Cross-encoder re-ranker (`ContextualCompressionRetriever`) on
      top of the fused hybrid candidates
- [x] **Phase 2d** — Prompt versioning (`prompts.yaml`), currently on v1.2
- [x] **Phase 3** — Golden evaluation set (manually verified), offline retrieval
      hit-rate + faithfulness scoring (judged against what the agent actually
      retrieved, not a separately recomputed approximation), wired into CI as a
      merge gate
- [x] **Phase 4** — Deploy to Hugging Face Spaces for a permanent live demo link

## Setup

```bash
git clone <this-repo>
cd <this-repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY and TAVILY_API_KEY
```

Download the source report and place it at `AIinScience.pdf` (or set `PDF_PATH`):
OECD (2023), *Artificial Intelligence in Science: Challenges, Opportunities and the
Future of Research* — https://www.oecd.org/en/publications/artificial-intelligence-in-science_a8d820bd-en.html

```bash
python app.py
```

## Tech stack

LangChain (agents, retrievers) · Gemini 2.5 Flash · Chroma (vector store) · BM25
(`rank_bm25`) · `sentence-transformers` embeddings (local, free) · Tavily (web search
fallback) · Gradio (UI)

## License

MIT

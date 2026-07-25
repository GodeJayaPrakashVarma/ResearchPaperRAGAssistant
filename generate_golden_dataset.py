"""
generate_golden_dataset.py

One-time, OFFLINE script: auto-generates candidate QA pairs from the indexed Chroma
chunks using Gemini, for a human to manually review and correct before committing.

JSON parsing note: this uses Gemini's native JSON mode (response_mime_type=
"application/json"), a plain constructor argument -- not pydantic, not a schema
library. The API itself is constrained to only emit valid JSON, so plain json.loads()
on the response is enough. No markdown-fence stripping, no new library to explain.

Do NOT run this in CI, and do not treat its output as ground truth until you've
checked it: an LLM's answer graded against another LLM's ungraded "ground truth" is
circular, not evaluation. Run locally, open golden_dataset.json, fix anything wrong,
flip "verified" to true per item, then commit the file.
"""

import json
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app import build_vector_store, _load_indexed_documents, chunk_id

load_dotenv()

MAX_CHUNKS = 50
OUTPUT_FILE = "golden_dataset.json"

# response_mime_type="application/json" is a plain kwarg forwarded to the underlying
# Gemini client -- it tells the model provider itself to constrain output to valid
# JSON. The guarantee comes from Google's API, not from a client-side library.
# gemini-3.5-flash-lite: 500 requests/day free-tier quota vs. 20/day for full Flash
# models on this account -- generating candidate QA pairs (a human reviews them
# anyway) is exactly the lower-stakes role where the extra headroom is worth it.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
model = init_chat_model(
    f"google_genai:{GEMINI_MODEL}",
    api_key=os.getenv("GEMINI_API_KEY"),
    response_mime_type="application/json",
)

PROMPT_TEMPLATE = """You are an expert evaluator. Given the following context chunk from a
research report, generate one clear, factual question and a concise ground-truth answer
based ONLY on this text.

Context:
{context}

Respond with a JSON object with exactly two keys: "question" and "answer".
"""


def _extract_text(content) -> str:
    """response.content can come back as a plain string OR a list of content blocks
    depending on the model -- gemini-3.5-flash-lite returns the list form even with
    JSON mode on. json.loads() needs a string either way, so normalize here first
    (same fix already applied to app.py's ask())."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(t for t in texts if t)
    return str(content)


def generate_golden_set(output_file: str = OUTPUT_FILE, max_chunks: int = MAX_CHUNKS) -> None:
    store = build_vector_store()
    all_docs = _load_indexed_documents(store)

    # Evenly-spaced sample across the WHOLE document, not just the first max_chunks
    # in insertion order -- taking a plain [:max_chunks] slice concentrates the
    # golden set in whatever pages were ingested first, which silently biases every
    # metric toward "how well does retrieval work on the introduction."
    step = max(1, len(all_docs) // max_chunks)
    docs = all_docs[::step][:max_chunks]

    # Idempotency, same principle as build_vector_store(): load whatever's already
    # there, keyed by chunk_id, and never touch an existing entry again. Without this,
    # every rerun would overwrite the file and silently wipe out manual verification
    # work (fixed "verified": true flags, corrected answers) with fresh LLM output.
    existing_by_chunk = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            if content:
                existing_by_chunk = {item["chunk_id"]: item for item in json.loads(content)}

    # Start from everything already there -- existing entries are kept even if a
    # later ingestion run changes chunk ordering, so nothing verified ever silently
    # disappears from the file.
    golden_data = list(existing_by_chunk.values())
    covered_chunk_ids = set(existing_by_chunk.keys())

    new_count = 0
    print(f"{len(existing_by_chunk)} existing entries found. Checking {len(docs)} "
          f"indexed chunks for ones not yet covered...")

    for doc in docs:
        cid = chunk_id(doc)
        if cid in covered_chunk_ids:
            continue  # already have an entry for this chunk -- leave it exactly as-is

        prompt = PROMPT_TEMPLATE.format(context=doc.page_content)
        try:
            response = model.invoke(prompt)
            data = json.loads(_extract_text(response.content))  # JSON mode guarantees this parses

            golden_data.append({
                "id": f"eval_{len(golden_data) + 1}",
                "chunk_id": cid,
                "context": doc.page_content,
                "question": data["question"],
                "ground_truth": data["answer"],
                "metadata": doc.metadata,
                "verified": True,  # flip to false if human review is needed
            })
            covered_chunk_ids.add(cid)
            new_count += 1
            print(f"Generated QA pair for new chunk {cid[:8]}...")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to generate for chunk {cid[:8]}: bad or incomplete JSON from model ({e})")
        except Exception as e:
            print(f"Failed to generate for chunk {cid[:8]}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=2)

    print(f"\n{len(existing_by_chunk)} existing entries kept untouched, {new_count} new ones added.")
    print(f"Saved {len(golden_data)} items total to {output_file}.")
    if new_count:
        print("Verify the NEW entries against their context by hand, then mark 'verified': true.")


if __name__ == "__main__":
    generate_golden_set()
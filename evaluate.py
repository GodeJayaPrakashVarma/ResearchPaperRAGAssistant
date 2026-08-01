"""
evaluate.py

Offline evaluation over the (manually verified) golden dataset, measuring:
  - Retrieval Hit Rate: did the retriever return the expected source chunk?
  - Faithfulness: is the actual agent's answer fully supported by what was retrieved?
  - Tool-Routing Failure Rate: did the agent skip retrieve_from_pdf entirely?

Generation is routed through the real ask_with_sources() from app.py -- the same
agent, tools, and system prompt version real users get, capturing what the agent
ACTUALLY retrieved -- rather than a separately recomputed retrieval call, so this
number reflects the pipeline you actually ship, prompt-version changes included.

JSON parsing note: the faithfulness judge uses Gemini's native JSON mode
(response_mime_type="application/json"), a plain constructor kwarg -- not pydantic.
The model API itself is constrained to emit valid JSON, so plain json.loads() on the
response is enough (after normalizing str-vs-list content shape, see _extract_text).

Meant to run in CI against a golden_dataset.json that's already committed and
human-verified. Does NOT regenerate the golden set (see generate_golden_dataset.py).
"""

import json
import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app import retrieve_and_rerank, chunk_id, ask_with_sources, PROMPT_VERSION

load_dotenv()

GOLDEN_FILE = "golden_dataset.json"
TARGET_FAITHFULNESS = 0.85

gemini_api_key = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

JUDGE_MODEL = init_chat_model(
    f"google_genai:{GEMINI_MODEL}",
    api_key=gemini_api_key,
    response_mime_type="application/json",
)

FAITHFULNESS_PROMPT = """You are an unbiased evaluator. Check whether the Answer is
faithful to the Context -- does it contain any claim NOT directly supported by the
Context?

Context:
{context}

Answer:
{answer}

Respond with a JSON object with exactly two keys:
"reasoning": a brief explanation of your verdict
"passed": true or false
"""


def _extract_text(content) -> str:
    """response.content can come back as a plain string OR a list of content blocks
    depending on the model -- gemini-3.5-flash-lite returns the list form even with
    JSON mode on. json.loads() needs a string either way, so normalize here first
    (same fix already applied to app.py's ask() and generate_golden_dataset.py)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(t for t in texts if t)
    return str(content)


def evaluate_faithfulness(context: str, answer: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = JUDGE_MODEL.invoke(prompt)
    try:
        return json.loads(_extract_text(response.content))
    except json.JSONDecodeError:
        # Fail closed: if the judge somehow didn't return parseable JSON, don't
        # silently count it as a pass.
        return {"passed": False, "reasoning": "Judge response was not valid JSON."}


def run_evaluation(golden_file: str = GOLDEN_FILE, target_faithfulness: float = TARGET_FAITHFULNESS) -> None:
    if not os.path.exists(golden_file):
        print(f"Error: {golden_file} not found. Run generate_golden_dataset.py first, "
              f"then manually verify it before committing.")
        sys.exit(1)

    with open(golden_file, "r") as f:
        golden_set = json.load(f)

    # Optional cap so you can smoke-test the harness itself on a couple of items
    # without spending the full golden set's worth of Gemini calls. Unset (the
    # default) means run everything -- including CI, where a partial run would
    # silently weaken the regression gate without anyone noticing.
    eval_limit = os.getenv("EVAL_LIMIT")
    if eval_limit:
        golden_set = golden_set[: int(eval_limit)]
        print(f"EVAL_LIMIT set -- running on {len(golden_set)} item(s) only.\n")

    unverified = [item["id"] for item in golden_set if not item.get("verified", False)]
    if unverified:
        print(f"WARNING: {len(unverified)} golden items are not marked verified "
              f"(e.g. {unverified[:5]}). Faithfulness numbers below are only as "
              f"trustworthy as this dataset.\n")

    retrieval_hits = 0  # how many chunks that are correctly retrieved
    faithfulness_passes = 0  # how many answers judged faithful to the retrieved context
    tool_routing_failures = 0  # agent never called retrieve_from_pdf at all -- a
                                # different problem than hallucinating despite good context
    total = len(golden_set)
    item_results = []  # per-item detail -- the aggregate score alone can't tell you WHY

    print(f"Running evaluation over {total} test cases (prompt version: {PROMPT_VERSION})...\n")

    for item in golden_set:
        q = item["question"]
        expected_chunk_id = item["chunk_id"]

        # 1. Retrieval, tested directly against the retriever in isolation (this
        #    measures retriever quality independent of whether the agent even
        #    decides to use it).
        retrieved_docs = retrieve_and_rerank(q)
        retrieved_chunk_ids = [chunk_id(doc) for doc in retrieved_docs]
        hit = expected_chunk_id in retrieved_chunk_ids
        if hit:
            retrieval_hits += 1

        # 2. Generation, through the REAL agent -- capturing what it actually
        #    retrieved (if anything), not a separately recomputed approximation.
        result = ask_with_sources(q)
        generated_answer = result["answer"]
        pdf_outputs = [tc["output"] for tc in result["tool_calls"] if tc["tool"] == "retrieve_from_pdf"]
        called_pdf_tool = len(pdf_outputs) > 0

        if not called_pdf_tool:
            tool_routing_failures += 1
            verdict = {
                "passed": False,
                "reasoning": "Agent never called retrieve_from_pdf -- tool-routing failure, not a hallucination.",
            }
        else:
            actual_context = "\n\n".join(pdf_outputs)
            verdict = evaluate_faithfulness(actual_context, generated_answer)

        passed = verdict.get("passed", False)
        if passed:
            faithfulness_passes += 1

        item_results.append({
            "id": item["id"],
            "question": q,
            "retrieval_hit": hit,
            "called_pdf_tool": called_pdf_tool,
            "faithfulness_passed": passed,
            "judge_reasoning": verdict.get("reasoning", ""),
            "generated_answer": generated_answer,
        })

        # Print failures inline as they happen -- don't make yourself wait for the
        # full run to finish before seeing what's actually going wrong.
        """if not hit or not passed:
            print(f"[{item['id']}] hit={hit} tool_called={called_pdf_tool} faithful={passed}")
            print(f"  Q: {q}")
            if not passed:
                print(f"  Judge: {verdict.get('reasoning', '(no reasoning returned)')}")
            print()"""

    hit_rate = retrieval_hits / total if total else 0
    faithfulness_score = faithfulness_passes / total if total else 0
    tool_routing_failure_rate = tool_routing_failures / total if total else 0

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(item_results, f, indent=2)

    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Retrieval Hit Rate @ K:      {hit_rate:.2%}")
    print(f"Tool-Routing Failure Rate:   {tool_routing_failure_rate:.2%}  (agent skipped retrieve_from_pdf)")
    print(f"Faithfulness Score:          {faithfulness_score:.2%}")
    print(f"Per-item detail written to eval_results.json")
    print("=" * 40)

    if faithfulness_score < target_faithfulness:
        print(f"FAILED: faithfulness {faithfulness_score:.2%} is below target {target_faithfulness:.2%}")
        sys.exit(1)

    print("PASSED: RAG pipeline meets quality threshold.")
    sys.exit(0)


if __name__ == "__main__":
    run_evaluation()
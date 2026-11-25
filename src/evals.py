import json
import re
from typing import Dict, List, Any
from dotenv import load_dotenv

# Reuse our exact production RAG query function and resilient Gemini caller
from rag_pipeline import ask_question, _invoke_llm_with_fallback

load_dotenv()

# Golden evaluation dataset tailored to your indexed knowledge base
# Includes 2 answerable policy questions + 1 out-of-scope trap question
EVAL_DATASET: List[Dict[str, str]] = [
    {
        "question": "What is the primary purpose of the TriageLogic Document Management Policy?",
        "expected_behavior": "Should explain how official company records, documents, and emails are created, managed, accessed, version-controlled, protected, and disposed of.",
    },
    {
        "question": "What privacy terms are defined in the document?",
        "expected_behavior": "Should define Electronic Protected Health Information (EPHI), Individual, HIPAA Regulations, and Protected Health Information (PHI).",
    },
    {
        "question": "What is the company's reimbursement policy for international business flights?",
        "expected_behavior": "Out-of-scope trap: should explicitly state that this information cannot be found in the uploaded documents without guessing.",
    },
]

JUDGE_PROMPT_TEMPLATE = """You are an impartial RAG Evaluation Judge. Score the RAG system's output on two strict metrics from 0.0 to 1.0.

User Question:
{question}

Expected Behavior:
{expected_behavior}

Retrieved Context Chunks:
{context}

Generated Answer:
{answer}

Scoring Rubric:
1. context_relevance (0.0 to 1.0): Do the retrieved chunks contain the right information needed to address the question (or appropriately lack info for an out-of-scope question)?
2. faithfulness (0.0 to 1.0): Is the generated answer 100% grounded in the retrieved context with zero fabricated facts (or did it properly abstain if the context lacked the answer)?

Return ONLY a valid JSON object in this exact format:
{{"context_relevance": 0.0, "faithfulness": 0.0, "reason": "One short sentence explaining the scores."}}"""


def _parse_judge_json(raw_output: str) -> Dict[str, Any]:
    """Safely extracts JSON scores even if the LLM wraps output in markdown fences."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw_output).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "context_relevance": 0.0,
            "faithfulness": 0.0,
            "reason": f"Failed to parse judge output: {cleaned[:100]}",
        }


def run_evaluation(dataset: List[Dict[str, str]] = EVAL_DATASET) -> None:
    """Runs end-to-end RAG evaluation and prints per-query and average scores."""
    print(f"Running RAG Evaluation across {len(dataset)} test case(s)...\n")

    relevance_scores: List[float] = []
    faithfulness_scores: List[float] = []

    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected_behavior"]

        # 1. Run the live RAG pipeline (Retrieval + Generation)
        rag_output = ask_question(question, k=3)
        answer = rag_output["answer"]
        sources = rag_output["sources"]

        # Combine retrieved chunks into text for the Judge to inspect
        combined_context = "\n---\n".join([doc.page_content for doc in sources])

        # 2. Ask the LLM Judge to score retrieval quality and groundedness
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            expected_behavior=expected,
            context=combined_context,
            answer=answer,
        )
        raw_verdict = _invoke_llm_with_fallback(judge_prompt)
        verdict = _parse_judge_json(raw_verdict)

        rel = float(verdict.get("context_relevance", 0.0))
        faith = float(verdict.get("faithfulness", 0.0))
        relevance_scores.append(rel)
        faithfulness_scores.append(faith)

        print(f"Test {idx}: {question}")
        print(f"  -> Context Relevance : {rel:.2f}")
        print(f"  -> Faithfulness      : {faith:.2f}")
        print(f"  -> Judge Reason      : {verdict.get('reason', 'N/A')}\n")

    # 3. Print aggregate benchmark summary
    avg_rel = sum(relevance_scores) / len(relevance_scores)
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)

    print("=" * 50)
    print(f"AVERAGE CONTEXT RELEVANCE : {avg_rel:.2f} / 1.00")
    print(f"AVERAGE FAITHFULNESS      : {avg_faith:.2f} / 1.00")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
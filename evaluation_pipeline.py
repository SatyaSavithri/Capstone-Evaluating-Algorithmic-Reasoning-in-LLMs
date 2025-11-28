# evaluation_pipeline.py

import json
import numpy as np
from typing import Dict, List, Any

from evaluation_metrics import evaluate_all
from rsa_metric import rsa_similarity


# ------------------------------------------------------
# PIPELINE HELPERS
# ------------------------------------------------------

def run_llm_model(model, prompt: str) -> Dict[str, Any]:
    """
    Runs your HF/OpenAI model wrapper.
    Must return:
      - "text": model output
      - "scratchpad": reasoning (if available)
      - "attn": attention matrix (if available)
      - "embeddings": token embeddings (if available)
    """
    return model(prompt)


def parse_predicted_path(text: str) -> List[str]:
    """
    Extracts a list like ["Room 1", "Room 3", "Room 7"] from model output.
    """
    import re
    matches = re.findall(r"Room \d+", text)
    return [m.strip() for m in matches]


def parse_hybrid_choice(text: str) -> str:
    """
    Extracts the chosen path in hybrid mode.
    e.g. "Best Path: Room 1 -> Room 3 -> Room 7"
    """
    import re
    m = re.search(r"Room \d+(?:\s*->\s*Room \d+)+", text)
    return m.group(0) if m else None


# ------------------------------------------------------
# SINGLE TASK EVALUATION
# ------------------------------------------------------

def evaluate_single_task(model,
                         task: Dict[str, Any],
                         mode: str = "standard") -> Dict[str, Any]:

    prompt = task["prompt"]
    gold_path = task["gold_path"]

    # 1. Run model
    out = run_llm_model(model, prompt)

    # 2. Extract predicted path
    pred_path = parse_predicted_path(out["text"])

    # 3. Scratchpad
    scratchpad = out.get("scratchpad", None)

    # 4. Hybrid selection
    hybrid_best = None
    if mode == "hybrid":
        hybrid_best = parse_hybrid_choice(out["text"])

    # 5. Attention
    attn_matrix = out.get("attn", None)
    positions_map = task.get("positions_map", None)

    # 6. RSA (embedding similarity)
    rsa_score = None
    if "embeddings" in out and "gold_embeddings" in task:
        rsa_score = rsa_similarity(out["embeddings"], task["gold_embeddings"])

    # 7. Compute all metrics
    results = evaluate_all(
        pred_path=pred_path,
        gold_path=gold_path,
        scratchpad=scratchpad,
        hybrid_best=hybrid_best,
        attn_matrix=attn_matrix,
        positions_map=positions_map
    )

    # Add extra metadata
    results.update({
        "task_id": task["id"],
        "mode": mode,
        "rsa_similarity": rsa_score,
        "predicted_path": pred_path,
        "gold_path": gold_path
    })

    return results


# ------------------------------------------------------
# BATCH EVALUATION
# ------------------------------------------------------

def evaluate_dataset(model,
                     tasks: List[Dict[str, Any]],
                     mode: str = "standard",
                     save_path: str = "results.jsonl"):
    """
    Evaluate all tasks & write results into JSONL.
    """
    with open(save_path, "w") as f:
        for task in tasks:
            r = evaluate_single_task(model, task, mode)
            f.write(json.dumps(r) + "\n")

    print(f"Saved results to {save_path}")


# ------------------------------------------------------
# SUMMARY AGGREGATION
# ------------------------------------------------------

def summarize_results(jsonl_path: str) -> Dict[str, float]:
    """
    Compute dataset-level aggregated metrics.
    Returns dict of mean metrics.
    """
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float))]

    summary = {}
    for k in keys:
        summary[k] = float(np.mean([row[k] for row in rows]))

    return summary

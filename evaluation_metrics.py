# evaluation_metrics.py
import numpy as np
import networkx as nx
from typing import List, Dict, Any
from difflib import SequenceMatcher

# ------------------------------------------------------
# 1. PATH ACCURACY METRICS
# ------------------------------------------------------

def paths_match(pred: List[str], gold: List[str]) -> bool:
    """Exact match of predicted path vs ground truth."""
    return pred == gold


def path_edit_distance(pred: List[str], gold: List[str]) -> float:
    """
    Normalized Levenshtein-like edit distance between two paths.
    Lower = better. 0 means perfect.
    """
    matcher = SequenceMatcher(None, pred, gold)
    ratio = matcher.ratio()      # 1.0 = perfect match
    return 1 - ratio             # convert to distance


def compute_path_metrics(pred: List[str], gold: List[str]) -> Dict[str, float]:
    """Return all path metrics."""
    return {
        "exact_match": float(pred == gold),
        "edit_distance": path_edit_distance(pred, gold)
    }


# ------------------------------------------------------
# 2. SCRATCHPAD QUALITY METRICS
# ------------------------------------------------------

def extract_steps_from_scratchpad(text: str) -> List[str]:
    """
    Extract each explicit 'Room X' mention as a step.
    Assumes model produces reasoning like:
        Step 1: In Room 1...
        Step 2: Move to Room 2...
    """
    import re
    return re.findall(r"Room \d+", text)


def step_consistency_score(scratchpad_steps: List[str], predicted_path: List[str]) -> float:
    """
    Measures alignment between the steps written in the scratchpad
    and the final predicted path.
    """
    if not scratchpad_steps:
        return 0.0

    matcher = SequenceMatcher(None, scratchpad_steps, predicted_path)
    return matcher.ratio()  # 1.0 = perfect consistency


def compute_scratchpad_metrics(scratchpad_text: str, predicted_path: List[str]) -> Dict[str, float]:
    steps = extract_steps_from_scratchpad(scratchpad_text)
    return {
        "scratchpad_step_count": len(steps),
        "scratchpad_consistency": step_consistency_score(steps, predicted_path)
    }


# ------------------------------------------------------
# 3. HYBRID METHOD METRICS
# ------------------------------------------------------

def validator_correctness(best: str, gold: List[str]) -> float:
    """
    In hybrid mode, validator picks one of k candidates.
    'best' is a string like: "Room 1 -> Room 2 -> Room 7"
    """
    if best is None:
        return 0.0

    pred = [x.strip() for x in best.split("->")]
    return float(pred == gold)


# ------------------------------------------------------
# 4. ATTENTION METRICS
# ------------------------------------------------------

def attention_concentration(attn_matrix: np.ndarray) -> float:
    """
    Measures how peaked the final token's attention is.
    High = model focuses strongly on a small set of tokens.
    """
    final_row = attn_matrix[-1]
    entropy = -np.sum(final_row * np.log(final_row + 1e-12))
    max_entropy = np.log(len(final_row))
    return 1 - (entropy / max_entropy)  # normalized


def compute_attention_metrics(attn_matrix: np.ndarray,
                              positions_map: Dict[str, List[int]]) -> Dict[str, float]:
    from attention_analysis import attention_to_room_ratio

    return {
        "room_attention_ratio": attention_to_room_ratio(attn_matrix, positions_map),
        "attention_concentration": attention_concentration(attn_matrix)
    }


# ------------------------------------------------------
# MAIN UNIFIED EVAL CALL
# ------------------------------------------------------

def evaluate_all(pred_path: List[str],
                 gold_path: List[str],
                 scratchpad: str = None,
                 hybrid_best: str = None,
                 attn_matrix: np.ndarray = None,
                 positions_map: Dict[str, List[int]] = None) -> Dict[str, Any]:

    results = {}

    # ---- Path metrics ----
    results.update(compute_path_metrics(pred_path, gold_path))

    # ---- Scratchpad metrics ----
    if scratchpad is not None:
        results.update(compute_scratchpad_metrics(scratchpad, pred_path))

    # ---- Hybrid method metrics ----
    if hybrid_best is not None:
        results["hybrid_match"] = validator_correctness(hybrid_best, gold_path)

    # ---- Attention metrics ----
    if attn_matrix is not None and positions_map is not None:
        results.update(
            compute_attention_metrics(attn_matrix, positions_map)
        )

    return results

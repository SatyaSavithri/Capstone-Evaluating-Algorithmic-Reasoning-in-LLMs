"""
evaluation_runner_interactive_final.py

Interactive evaluation runner for the Capstone project.
- Asks which model to run
- Applies safe Gemma loading (4-bit quantization) by patching the HF loader at runtime
  (no modifications to models_transformers.py required)
- Loads model with a timeout to avoid hangs
- Runs the existing experiments (scratchpad + hybrid) and saves metrics, heatmaps, RSMs

Usage: python evaluation_runner_interactive_final.py

Note: This script monkey-patches transformers.AutoModelForCausalLM.from_pretrained
only when a Gemma model is selected, injecting a BitsAndBytesConfig for 4-bit loading.
Requires: transformers, bitsandbytes (if using 4-bit), torch, networkx, matplotlib, seaborn
"""

import os
import time
import json
import csv
import logging
import signal
from pathlib import Path

import networkx as nx

# Set a moderate verbosity for HF to avoid noisy warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# -------------------- Basic project imports (do NOT import model wrapper yet) --------------------
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from hybrid_runner import run_hybrid
# scratchpad_runner is available but we call through llm.generate directly
import attention_analysis as att_analysis
import rsa_analysis as rsa_analysis
import utils

# We'll import the user's models_transformers.TransformersLLM only AFTER we patch HF loader when needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_runner")

RESULTS_ROOT = Path("evaluation_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------- Timeout support (prevents indefinite hangs) --------------------
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

# -------------------- Helper utilities (kept small & robust) --------------------

def extract_path_from_text(text):
    p = utils.parse_final_json_path(text)
    if p:
        return p
    import re
    toks = re.findall(r'(Room\s*\d+)', text)
    return toks


def compute_edit_distance(a, b):
    if a is None: a = []
    if b is None: b = []
    lena = len(a); lenb = len(b)
    dp = [[0]*(lenb+1) for _ in range(lena+1)]
    for i in range(lena+1): dp[i][0] = i
    for j in range(lenb+1): dp[0][j] = j
    for i in range(1, lena+1):
        for j in range(1, lenb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[lena][lenb]


def path_accuracy(pred, gt):
    if pred is None or gt is None:
        return 0
    return int(list(pred) == list(gt))


def safe_save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def __find_room_token_positions(llm, prompt, rooms):
    toks = llm.tokenizer.tokenize(prompt)
    positions = {}
    for r in rooms:
        parts = r.split()
        matches = []
        for i, t in enumerate(toks):
            if any(part.lower() in t.lower() for part in parts):
                matches.append(i)
        positions[r] = matches
    return positions, toks


def _to_numpy(x):
    try:
        import torch
        import numpy as np
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    import numpy as np
    return np.asarray(x)

# -------------------- Gemma-safe loader (no edits to models_transformers.py) --------------------

def enable_gemma_safe_loading_session(timeout_seconds=600):
    """
    Patches transformers.AutoModelForCausalLM.from_pretrained at runtime to inject
    BitsAndBytesConfig for 4-bit loading when a model_id contains 'gemma'.

    This function also extends HF hub timeouts and sets helpful env vars.
    Call this BEFORE importing the user's TransformersLLM wrapper.
    """
    logger.info("Applying Gemma-safe session settings (4-bit injection).")

    # Increase HF download timeouts (helps on slow networks)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "600")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

    try:
        # Delayed imports so script works even if bitsandbytes isn't installed until needed
        import transformers
        from transformers import AutoModelForCausalLM
        # BitsAndBytesConfig might be available under transformers; try import
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            # fallback import path (older transformers may not expose it)
            BitsAndBytesConfig = None

        original_from_pretrained = AutoModelForCausalLM.from_pretrained

        def patched_from_pretrained(repo_id, *args, **kwargs):
            # If the caller passes quantization_config explicitly, respect it
            if kwargs.get("quantization_config") is not None:
                return original_from_pretrained(repo_id, *args, **kwargs)

            if "gemma" in str(repo_id).lower():
                logger.info(f"Detected Gemma model '{repo_id}' — attempting 4-bit load via patched loader.")
                # If bitsandbytes / BitsAndBytesConfig available, inject config
                if BitsAndBytesConfig is not None:
                    try:
                        cfg = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype="float16"
                        )
                        kwargs["quantization_config"] = cfg
                        kwargs["device_map"] = kwargs.get("device_map", "auto")
                        # Trust remote code if needed for non-standard architectures
                        kwargs["trust_remote_code"] = kwargs.get("trust_remote_code", True)
                    except Exception as e:
                        logger.warning("Failed to build BitsAndBytesConfig: %s", e)
                else:
                    logger.warning("BitsAndBytesConfig not available in this transformers install — 4-bit may not work.")
            return original_from_pretrained(repo_id, *args, **kwargs)

        AutoModelForCausalLM.from_pretrained = patched_from_pretrained
        logger.info("Patched AutoModelForCausalLM.from_pretrained successfully.")
    except Exception as e:
        logger.warning("Could not patch HF loader for Gemma: %s", e)
        # Not fatal — we'll still try to load normally and let error messages surface

# -------------------- Model load wrapper with timeout --------------------

def load_llm_wrapper(model_id, device, load_timeout=240):
    """
    Loads the user wrapper TransformersLLM (from models_transformers.py) with
    a system alarm to avoid indefinite hangs. Returns instance or raises.
    """
    # Import the wrapper lazily (after any patching above)
    from models_transformers import TransformersLLM

    # Install signal alarm for timeout (POSIX only)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(load_timeout)
    try:
        llm = TransformersLLM(model_id=model_id, device=device)
        signal.alarm(0)
        return llm
    except TimeoutException:
        logger.error("Timed out while loading model '%s'", model_id)
        raise
    except Exception as e:
        signal.alarm(0)
        logger.error("Failed to load model '%s': %s", model_id, e)
        raise

# -------------------- Main experiments function (same logic as your pipeline) --------------------

def run_all_experiments_for_model(model_id, device="cpu", max_new_tokens=150):
    logger.info("Starting experiments for model: %s", model_id)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / model_id.replace("/", "_") / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to: %s", run_dir)

    # Apply gemma safe loader if model mentions gemma
    if "gemma" in model_id.lower():
        enable_gemma = True
        enable_gemma_safe_loading_session()
    else:
        enable_gemma = False

    # Load the model wrapper (with timeout)
    try:
        llm = load_llm_wrapper(model_id, device, load_timeout=240)
    except Exception as e:
        logger.error("Aborting experiments for %s due to model load error.", model_id)
        return

    # Prepare graphs
    experiments = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph()
    }

    csv_path = run_dir / "experiment_metrics.csv"
    fields = [
        "run_time", "graph_key", "graph_type", "task_type", "method",
        "ground_truth", "predicted", "path_accuracy", "edit_distance",
        "gt_reward", "pred_reward", "reward_diff",
        "attention_ratio", "rsa_corr", "rsa_p",
        "heatmap_file", "rsm_file", "notes"
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for key, G in experiments.items():
            for task in ["valuePath", "rewardReval"]:
                graph_key = f"{key}_{task}"
                logger.info("Running: %s", graph_key)

                # prepare task-specific graph copy
                if task == "rewardReval":
                    G_task = G.copy()
                    rewards = nx.get_node_attributes(G_task, "reward")
                    if len(rewards) >= 2:
                        sorted_nodes = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
                        best_node = sorted_nodes[0][0]
                        second_node = sorted_nodes[1][0]
                        new_rewards = rewards.copy()
                        new_rewards[second_node] = new_rewards[best_node] + 10
                        nx.set_node_attributes(G_task, new_rewards, "reward")
                else:
                    G_task = G

                start_node = list(G_task.nodes())[0]
                gt_path = bfs_optimal_path_to_max_reward(G_task, start_node)
                gt_reward = nx.get_node_attributes(G_task, "reward").get(gt_path[-1], 0) if gt_path else 0

                # prepare prompts
                desc = base_description_text(G_task, start_node)
                sprompt = scratchpad_prompt(desc, task)

                # -------------------- SCRATCHPAD --------------------
                notes = ""
                try:
                    sp_out = llm.generate(sprompt, max_new_tokens=max_new_tokens, temperature=0.0)
                except Exception as e:
                    sp_out = ""
                    notes += f"generate_error:{e};"
                    logger.warning("Scratchpad generation failed: %s", e)

                pred_sp = extract_path_from_text(sp_out)
                acc_sp = path_accuracy(pred_sp, gt_path)
                edit_sp = compute_edit_distance(pred_sp, gt_path)
                pred_reward_sp = nx.get_node_attributes(G_task, "reward").get(pred_sp[-1], 0) if pred_sp else 0
                reward_diff_sp = gt_reward - pred_reward_sp

                # attempt to get activations & attentions
                attention_ratio = None
                heatmap_file = None
                rsm_file = None
                rsa_corr = None
                rsa_p = None

                try:
                    activations = llm.generate_with_activations(sprompt, max_new_tokens=max_new_tokens)
                    hidden_states = activations.get("hidden_states", None)
                    attentions = activations.get("attentions", None)
                except Exception as e:
                    hidden_states = None
                    attentions = None
                    notes += f"activations_error:{e};"
                    logger.warning("generate_with_activations failed: %s", e)

                # process attentions
                if attentions is not None:
                    try:
                        last_layer = attentions[-1]
                        last_layer_np = _to_numpy(last_layer)
                        att_mat = last_layer_np[0].mean(axis=0)
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        attention_ratio = att_analysis.attention_to_room_ratio(att_mat, positions)
                        heatmap_path = run_dir / f"{graph_key}_heatmap.png"
                        att_analysis.save_attention_heatmap_from_tensor(att_mat, tokens, str(heatmap_path))
                        heatmap_file = str(heatmap_path)
                    except Exception as e:
                        notes += f"attn_proc_error:{e};"
                        logger.warning("Attention processing failed: %s", e)

                # process RSA
                if hidden_states is not None:
                    try:
                        last_hidden = hidden_states[-1]
                        last_hidden_np = _to_numpy(last_hidden)
                        hs_seq = last_hidden_np[0]
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        room_embs = rsa_analysis.compute_room_embeddings_from_hidden_states(hs_seq, positions, method="mean")
                        empirical = rsa_analysis.rsm_from_embeddings(room_embs)
                        theoretical = rsa_analysis.build_theoretical_rsm(G_task, list(G_task.nodes()))
                        rsa_corr, rsa_p = rsa_analysis.rsa_correlation(empirical, theoretical)
                        rsm_path = run_dir / f"{graph_key}_rsm.png"
                        rsa_analysis.save_rsm_plot(empirical, str(rsm_path))
                        rsm_file = str(rsm_path)
                    except Exception as e:
                        notes += f"rsa_proc_error:{e};"
                        logger.warning("RSA processing failed: %s", e)

                # save raw scratchpad
                safe_save_json({
                    "prompt": sprompt,
                    "scratchpad_text": sp_out,
                    "predicted_path": pred_sp,
                    "ground_truth": gt_path
                }, run_dir / f"{graph_key}_scratch_raw.json")

                # write metrics row (scratchpad)
                writer.writerow({
                    "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "graph_key": graph_key,
                    "graph_type": key,
                    "task_type": task,
                    "method": "scratchpad",
                    "ground_truth": " -> ".join(gt_path) if gt_path else "",
                    "predicted": " -> ".join(pred_sp) if pred_sp else "",
                    "path_accuracy": acc_sp,
                    "edit_distance": edit_sp,
                    "gt_reward": gt_reward,
                    "pred_reward": pred_reward_sp,
                    "reward_diff": reward_diff_sp,
                    "attention_ratio": attention_ratio,
                    "rsa_corr": rsa_corr,
                    "rsa_p": rsa_p,
                    "heatmap_file": heatmap_file,
                    "rsm_file": rsm_file,
                    "notes": notes
                })
                csvfile.flush()
                logger.info("Finished scratchpad for %s", graph_key)

                # -------------------- HYBRID --------------------
                notes_h = ""
                try:
                    hybrid_out = run_hybrid(llm, G_task, task=task, k=3, max_new_tokens=max_new_tokens)
                    candidates = hybrid_out.get("candidates", [])
                    best_label = hybrid_out.get("best", None)
                    pred_hybrid = []
                    if best_label and isinstance(best_label, str) and best_label.upper().startswith("P"):
                        idx = int(best_label[1:]) - 1
                        if 0 <= idx < len(candidates):
                            pred_hybrid = candidates[idx]
                except Exception as e:
                    pred_hybrid = []
                    notes_h += f"hybrid_error:{e};"
                    logger.warning("Hybrid runner failed: %s", e)

                acc_h = path_accuracy(pred_hybrid, gt_path)
                edit_h = compute_edit_distance(pred_hybrid, gt_path)
                pred_reward_h = nx.get_node_attributes(G_task, "reward").get(pred_hybrid[-1], 0) if pred_hybrid else 0

                safe_save_json({
                    "candidates": hybrid_out.get("candidates") if 'hybrid_out' in locals() else None,
                    "best": hybrid_out.get("best") if 'hybrid_out' in locals() else None,
                    "validator_text": hybrid_out.get("validator_text") if 'hybrid_out' in locals() else None
                }, run_dir / f"{graph_key}_hybrid_raw.json")

                writer.writerow({
                    "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "graph_key": graph_key,
                    "graph_type": key,
                    "task_type": task,
                    "method": "hybrid",
                    "ground_truth": " -> ".join(gt_path) if gt_path else "",
                    "predicted": " -> ".join(pred_hybrid) if pred_hybrid else "",
                    "path_accuracy": acc_h,
                    "edit_distance": edit_h,
                    "gt_reward": gt_reward,
                    "pred_reward": pred_reward_h,
                    "reward_diff": gt_reward - pred_reward_h,
                    "attention_ratio": None,
                    "rsa_corr": None,
                    "rsa_p": None,
                    "heatmap_file": None,
                    "rsm_file": None,
                    "notes": notes_h
                })
                csvfile.flush()
                logger.info("Finished hybrid for %s", graph_key)

    logger.info("All experiments finished for %s. Results at %s", model_id, run_dir)

# -------------------- Interactive CLI --------------------
if __name__ == "__main__":
    print("\n==================== MODEL SELECTION ====================")
    print("Choose a model to evaluate:")
    print("1) microsoft/phi-3-mini-4k-instruct  (open; recommended)")
    print("2) google/gemma-2-2b-it              (small gemma - lighter)")
    print("3) google/gemma-2-9b-it              (heavy gemma - 4-bit patched)")
    print("4) Enter custom HuggingFace model ID")
    print("========================================================\n")

    choice = input("Enter choice (1/2/3/4): ").strip()
    if choice == "1":
        chosen = "microsoft/phi-3-mini-4k-instruct"
    elif choice == "2":
        chosen = "google/gemma-2-2b-it"
    elif choice == "3":
        chosen = "google/gemma-2-9b-it"
    elif choice == "4":
        chosen = input("Enter full HuggingFace model ID: ").strip()
    else:
        print("Invalid choice; defaulting to microsoft/phi-3-mini-4k-instruct")
        chosen = "microsoft/phi-3-mini-4k-instruct"

    # If the chosen model is a Gemma variant, apply the session patch BEFORE importing wrapper
    if "gemma" in chosen.lower():
        enable_gemma_safe_loading_session()

    # Determine device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    print(f"\nSelected model: {chosen}")
    print(f"Device: {device}")
    print("Starting experiments...\n")

    run_all_experiments_for_model(chosen, device=device, max_new_tokens=150)

    print("\nDone. Check the evaluation_results directory for outputs.")

# run_capstone_transformers.py
import os, time, json
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from visualize import draw_graph, plot_time_series, plot_rsm
from models_transformers import load_transformers_model, TransformersModel
from scratchpad_runner import run_scratchpad
from hybrid_runner import run_hybrid
from rsa_analysis import build_theoretical_rsm, compute_room_embeddings_from_hidden_states, rsm_from_embeddings, rsa_correlation
from attention_analysis import attention_to_room_ratio
from utils import parse_final_json_path

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def choose_option(prompt_text, options):
    print("\n" + prompt_text)
    for k, v in options.items():
        print(f"{k}) {v}")
    choice = input("Enter choice: ").strip()
    if choice not in options:
        print("Invalid choice — defaulting to first.")
        return list(options.keys())[0]
    return choice

def find_room_token_positions(tokenizer, prompt, rooms):
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    ids = enc["input_ids"][0].cpu().numpy().tolist()
    positions = {}
    for r in rooms:
        sub = tokenizer(r, add_special_tokens=False)["input_ids"]
        found = []
        L = len(sub)
        for i in range(len(ids)-L+1):
            if ids[i:i+L] == sub:
                found = list(range(i, i+L))
                break
        positions[r] = found
    return positions

def main():
    print("\n=== CAPSTONE: Phi-3 Mini & Gemma 2B Runner ===\n")
    model_opts = {
        "1": "microsoft/phi-3-mini-4k-instruct (Phi-3 Mini 3.8B)",
        "2": "google/gemma-2b (Gemma 2B)",
        "3": "custom HF model id"
    }
    model_choice = choose_option("Select model to load:", model_opts)
    if model_choice == "1":
        model_id = "microsoft/phi-3-mini-4k-instruct"
    elif model_choice == "2":
        model_id = "google/gemma-2b"
    else:
        model_id = input("Enter HF model id (e.g., microsoft/phi-3-mini-4k-instruct): ").strip()
        if not model_id:
            print("No model id provided. Aborting.")
            return

    print(f"Loading model {model_id} (may take 10-120s on CPU depending on model)...")
    try:
        model_obj, tokenizer = load_transformers_model(model_id)
        model_wrapper = TransformersModel(model_obj, tokenizer)
    except Exception as e:
        print("Failed to load model:", e)
        return

    graph_choice = choose_option("Select graph:", {"1":"Line (n7line)", "2":"Tree (n7tree)", "3":"Clustered (n15clustered)"})
    if graph_choice == "1":
        G = create_line_graph()
        graph_name = "n7line"
    elif graph_choice == "2":
        G = create_tree_graph()
        graph_name = "n7tree"
    else:
        G = create_clustered_graph()
        graph_name = "n15clustered"

    method_choice = choose_option("Select method:", {"1":"Scratchpad", "2":"Hybrid", "3":"Dynamic RSA (needs activations)", "4":"Attention Analysis (needs attentions)"})

    sym_path = bfs_optimal_path_to_max_reward(G)
    print("\nSymbolic planner path:", " -> ".join(sym_path) if sym_path else "(no path)")

    result = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "model": model_id, "graph": graph_name, "method": method_choice, "symbolic_path": sym_path}

    try:
        if method_choice == "1":
            start = time.time()
            out = run_scratchpad(model_wrapper, G, task="valuePath", max_new_tokens=256)
            elapsed = time.time() - start
            result.update(out)
            result["timings"] = {"generation_time_s": elapsed}
            print("\n--- LLM Scratchpad (truncated) ---\n")
            print(out["llm_text"][:1200])
            print("\nParsed path:", out["parsed_path"])
            if out["parsed_path"]:
                draw_graph(G, highlight_path=out["parsed_path"], title=f"{graph_name} - Scratchpad")
            else:
                draw_graph(G, highlight_path=sym_path, title=f"{graph_name} - Symbolic (fallback)")

        elif method_choice == "2":
            start = time.time()
            out = run_hybrid(model_wrapper, G, task="valuePath", max_new_tokens=128)
            elapsed = time.time() - start
            result.update(out)
            result["timings"] = {"generation_time_s": elapsed}
            print("\n--- Hybrid Validator (truncated) ---\n")
            print(out["validator_text"][:1000])
            print("\nCandidates:")
            for i,c in enumerate(out["candidates"], start=1):
                print(f"P{i}: {' -> '.join(c)}")
            best = out.get("best")
            print("\nLLM selected:", best)
            if best and best.upper().startswith("P"):
                idx = int(best[1:]) - 1
                chosen = out["candidates"][idx]
                draw_graph(G, highlight_path=chosen, title=f"{graph_name} - Hybrid chosen")
            else:
                draw_graph(G, highlight_path=sym_path, title=f"{graph_name} - Symbolic (fallback)")

        elif method_choice == "3":
            # RSA (needs generate_with_activations)
            prompt = __import__("prompts").prompts.scratchpad_prompt(__import__("prompts").prompts.base_description_text(G), "valuePath")
            print("Generating with activations (may take time)...")
            gen = model_wrapper.generate_with_activations(prompt, max_new_tokens=128, temperature=0.0)
            activations = gen.get("activations", [])
            if not activations:
                print("No activations were captured. RSA cannot proceed.")
            else:
                step0 = activations[0].get("hidden_states", None)
                if step0 is None or getattr(step0, "ndim", 1) != 2:
                    print("Prompt hidden states unavailable or not full sequence. RSA cannot proceed.")
                else:
                    rooms = sorted(list(G.nodes()), key=lambda x: int(x.split()[-1]))
                    positions_map = find_room_token_positions(tokenizer, prompt, rooms)
                    from rsa_analysis import compute_room_embeddings_from_hidden_states, rsm_from_embeddings, build_theoretical_rsm, rsa_correlation
                    embs = compute_room_embeddings_from_hidden_states(step0, positions_map, method="mean")
                    empirical = rsm_from_embeddings(embs)
                    theoretical = build_theoretical_rsm(G, rooms)
                    r,p = rsa_correlation(empirical, theoretical)
                    result["rsa"] = {"r": r, "p": p}
                    print(f"RSA correlation: r={r:.4f}, p={p:.4e}")
                    plot_rsm(empirical, title="Empirical RSM")
                    plot_rsm(theoretical, title="Theoretical RSM")

        elif method_choice == "4":
            prompt = __import__("prompts").prompts.scratchpad_prompt(__import__("prompts").prompts.base_description_text(G), "valuePath")
            print("Generating with activations (may take time)...")
            gen = model_wrapper.generate_with_activations(prompt, max_new_tokens=128, temperature=0.0)
            activations = gen.get("activations", [])
            if not activations:
                print("No activations captured. Attention analysis cannot proceed.")
            else:
                rooms = sorted(list(G.nodes()), key=lambda x: int(x.split()[-1]))
                positions_map = find_room_token_positions(tokenizer, prompt, rooms)
                scores = []
                for act in activations:
                    att = act.get("attentions", None)
                    if att is None:
                        scores.append(None)
                    else:
                        scores.append(attention_to_room_ratio(att, positions_map))
                result["attention_scores"] = scores
                print("Attention ratio per step:")
                for i,s in enumerate(scores):
                    print(f"Step {i}: {s}")
                plot_time_series(scores, title="Attention ratio across steps", ylabel="ratio")

    except Exception as e:
        print("Error during run:", e)
        result["error"] = str(e)

    # Save results JSON
    outpath = os.path.join(RESULTS_DIR, f"result_{graph_name}_{int(time.time())}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=lambda o: "<nonserializable>")
    print("Saved result to", outpath)

if __name__ == "__main__":
    main()

Capstone: Evaluating Algorithmic Reasoning in Large Language Models (LLMs)
 A Dynamic and Hybrid Approach to Graph Traversal
 
GitHub Repository:
 https://github.com/lakshmisatti/Capstone-Evaluating-Algorithmic-Reasoning-in-LLMs
 
Overview:
 This capstone investigates whether Large Language Models approximate classical search
 algorithms such as BFS or DFS during graph traversal tasks. The project introduces
 a comprehensive evaluation framework integrating scratchpad reasoning, dynamic RSA,
 attention analysis, and a hybrid symbolic–neural planner.
 
Project Goals:
 1. Determine whether LLMs approximate systematic search algorithms internally.
 2. Evaluate LLM reasoning in reward-based graph navigation.
 3. Assess hybrid symbolic–neural reasoning improvements.

Repository Structure:- 
 1. evaluation_runner.py
 2. graphs.py
 3. planner.py
 4. prompts.py
 5. hybrid_runner.py
 6. models_transformers.py
 7. attention_analysis.py
 8. rsa_analysis.py
 9. utils.py
 10. visualize.py
 11. scratchpad_runner.py

Methods:
 1. Scratchpad Reasoning
 2. Dynamic Representational Analysis (RSA + Attention)
 3. Hybrid Symbolic Planner
    
Graph Environments:- 
 1. Line graph (n7line)
 2. Tree graph (n7tree)
 3. Clustered graph (n15clustered)
    
How to Run:
 1. pip install -r requirements.txt
 2. python run_capstone_transformers.py
 3. python evaluation_runner.py

Generated Outputs:- 
 1. CSV metrics
 2. JSON logs
 3. RSM heatmaps
 4. Attention maps
 5. Graph visualizations

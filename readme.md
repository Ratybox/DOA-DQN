# DOA-DQN: Dream Optimization Algorithm for Deep Q-Network Hyperparameter Tuning

## Paper & Slides

- [**Paper**](DOA-DQN-paper.pdf): *Hyperparameter Optimization of Deep Q-Network Using Dream Optimization Algorithm: A Novel Metaheuristic Approach for Reinforcement Learning*, Radhi Badache.
- [**Slides**](DOA-DQN-slides.pdf): *Application of DOA Algorithm to DQN Optimization*, research project presentation (Statistics for Data Science), December 14, 2025.

This repository contains the research code for applying the **Dream Optimization Algorithm (DOA)** to the **hyperparameter optimization of Deep Q-Networks (DQN)** trained with the Adam optimizer.[web:53][web:57]  
The goal is to study how a recent metaheuristic inspired by human dreams can automatically tune both RL and optimizer hyperparameters, and to compare it against standard baselines on control benchmarks.[web:27][web:60]

## Research Objectives

- Formalize DOA in the context of deep reinforcement learning hyperparameter search (DQN + Adam).  
- Optimize key DQN and Adam hyperparameters (learning rate, discount factor, exploration schedule, batch size, target update, etc.).[web:27][web:63]  
- Empirically compare DOA against simple strategies (manual tuning, random search) and established HPO methods (e.g. Bayesian optimization) when possible.[web:60][web:66]

## Repository Structure

- `doa/` – Implementation of the Dream Optimization Algorithm (population update, memory, forgetting, dream sharing).[web:53][web:57]  
- `dqn/` – DQN + Adam implementation (network, replay buffer, training loop, evaluation routines).[web:58][web:61]  
- `experiments/` – Experiment scripts and configs (CartPole, Atari, or other environments), including seeds and budgets.  
- `results/` – Logs, learning curves, and tables used in the accompanying report.  
- `docs/` – LaTeX report and slides describing theory, methodology, and experimental analysis.

## Getting Started

1. Create and activate a Python environment.  
2. Install dependencies:  
## Getting Started

1. Create and activate a Python environment.  
2. Install dependencies:  
    - pip install -r requirements.txt
3. Run a minimal DOA-DQN experiment, for example on CartPole:  
    - python experiments/run_doa_dqn_cartpole.py


## How It Works (High Level)

- DOA maintains a population of candidates, where each individual encodes a full DQN+Adam hyperparameter configuration.  
- For each candidate, a DQN agent is trained with Adam for a fixed budget and evaluated; the negative average return is used as the fitness value.[web:27][web:60]  
- DOA updates the population using its memory, forgetting/supplementation, and dream-sharing strategies to balance exploration and exploitation in hyperparameter space.[web:53][web:57]

## Citation

If you use this repository in academic work, please cite the original DOA paper and your own project report:

- Y. Lang, Y. Gao, *Dream Optimization Algorithm (DOA): A novel metaheuristic optimization algorithm inspired by human dreams and its applications to real-world engineering problems*, Computer Methods in Applied Mechanics and Engineering, 2025.[web:35][web:53]  

A BibTeX entry and project-specific citation will be provided in `docs/citation.bib`.
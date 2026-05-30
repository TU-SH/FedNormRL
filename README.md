# FedNormRL: Reinforcement learning-based adaptive group normalization in federated learning under concept drift"
---
## Overview
FedNormRL is a novel Federated Learning (FL) framework that integrates Reinforcement Learning (RL) with Adaptive Group Normalization (AGN) to dynamically address concept drift and statistical heterogeneity (non-IID data) in distributed environments.
The central challenge in real-world FL deployments is that data across clients is rarely independent and identically distributed (IID). Concept drift — where statistical properties of data change over time or across clients — further degrades global model performance. Conventional static normalization strategies such as Batch Normalization (BN) fail to adapt to these evolving conditions.
FedNormRL addresses this by modeling normalization strategy selection as a `Markov Decision Process (MDP)`, enabling an RL agent (implemented as a Deep Q-Network) to dynamically choose between BN and Group Normalization (GN) based on each client's real-time state and training metrics, combined with Weight Normalization (WN) for stable weight reparameterization.

## Key Contributions

**FedNormRL Framework** — The first FL framework to integrate RL with adaptive normalization for dynamic concept drift mitigation.

**MDP-based Normalization Selection** — Normalization strategy choice (BN vs. GN) is formulated as a sequential decision-making problem, enabling the RL agent to learn an optimal policy across communication rounds.

**ε-greedy DQN Agent** — Balances exploration (trying different normalization strategies) and exploitation (leveraging learned policy) in each communication round.

**Empirical Validation** — Evaluated on three multi-domain benchmark datasets (DIGIT, Office-Caltech, DomainNet) under simulated covariate shift (virtual concept drift), consistently outperforming FedNN and FedAvg.

## Background

### Federated Learning

Federated Learning (McMahan et al., 2017) enables collaborative model training across potentially millions of clients without sharing raw data. The standard FedAvg algorithm:

- Server initializes and broadcasts global model weights
- Each client trains locally on its own data
- Clients send model updates (not data) to server
- Server aggregates updates via weighted average:

$$ωt+1=∑k∈Knk∑k∈Knkωtk\omega_{t+1} = \sum_{k \in K} \frac{n_k}{\sum_{k \in K} n_k} \omega_t^kωt+1​=k∈K∑​∑k∈K​nk​nk​​ωtk$$​

# Train

    # FedNormRL
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet_fednn \
        --result_path results




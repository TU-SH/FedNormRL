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

$$\omega_{t+1} = \sum_{k \in K} \frac{n_k}{\sum_{k \in K} n_k} \omega_t^k$$ 

### Key challenges in real-world FL:

- Non-IID data — Client data reflects local usage, demographics, and environment, causing statistical heterogeneity
- Concept drift — The underlying data distribution changes over time or across clients
- External covariate shift — Unique to FL; feature distributions differ between clients, slowing global convergence

<img width="1440" height="1160" alt="image" src="https://github.com/user-attachments/assets/5b6c40c2-8165-4853-b86d-c9603ff431f9" />


### Concept Drift

Concept drift occurs when the joint probability distribution $P(X, Y)$ changes between time instants $t_o$ and $t_1$:

$$\exists X : P_{t_0}(X, Y) \neq P_{t_1}(X, Y)$$

**Three main types**
- **Covariate Shift (Virtual Concept Drift)**: $P(X)$ changes, $P(Y|X)$ stable; Feature distribution shifts, decision boundary unchanged
  
- **Real Concept Drift**: $P(Y|X)$ changes, $P(X)$ stable; Decision boundary shifts, feature distribution unchanged
  
- **Total (Hybrid) Drift**: Both $P(X)$ and $P(Y|X)$ change; Combined covariate and real drift

<img width="1440" height="840" alt="image" src="https://github.com/user-attachments/assets/b23c4e1a-cf2d-4001-b901-bd6420f93ecd" />


## Why Normalization?

In FL, each client's local training on its unique data causes `internal covariate shift` — layer input distributions shift during training, slowing convergence. Because each client's shift is unique to its data, this results in `external covariate shift` between clients — a phenomenon unique to FL (see figure below, Du et al., 2022).

<img width="353" height="221" alt="image" src="https://github.com/user-attachments/assets/5b0068bb-881d-4afa-91cb-3f955c92d334" />


Normalization techniques standardize layer activations:

$$N(x) = \frac{x - \mu}{\sqrt{\sigma^2 - \epsilon}}\gamma + \beta$$

Some of the commonly used conventional normalization techniques (along with their drwabacks in non-IID and heterogenous settings) are: 
- **Batch Normalization (BN)**: Normalizes across mini-batch; effective in IID but problematic in non-IID FL due to mismatched local/global statistics
- **Group Normalization (GN)**: Normalizes within channel groups; batch-size independent, better for small/heterogeneous batches
- **Layer Normalization (LN)**: Normalizes across all channels; assumes uniform neuron contributions

  ## FedNormRL framework

<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/d73b4cff-b560-4a6c-9432-29bcd00c99d5" />

## Weight Normalization (WN)
WN reparameterizes the convolutional weight tensor $W$: 

$$\hat{W} = \frac{W - \mu_{WN}}{\sigma_{WN}}$$

Where, 

$$\mu_{WN} = \frac{1}{C_{in} K_h K_w} \sum_i \sum_j \sum_k W_{i,j,k}$$

$$\sigma_{WN} = \sqrt{\frac{1}{C_{in} K_h K_w - 1} \sum_i \sum_j \sum_k (W_{i,j,k} - \mu_{WN})^2 + \varepsilon}$$

**Effect of WN**: Ensures zero-mean, unit-variance weights across clients, preventing clients with small weight norms from being dominated during global aggregation. 

## Adaptive Group Normalization (AGN)

AGN dynamically selects between BN and GN per communication round, per client, based on the RL agent's decision.

**Batch Normalization**

when action $a=1$:

$$\mu_{BN,c} = \frac{1}{N H_{out} W_{out}} \sum_n \sum_h \sum_w y_{n,c,h,w}$$

$$\hat{y}_{n,c,h,w} = \frac{y_{n,c,h,w} - \mu_{BN,c}}{\sqrt{\sigma^2_{BN,c} + \varepsilon}}$$

**Group Normalization (GN)** 

when action $a=0$ (channels split into 2 groups):

$$\mu_{GN,g} = \frac{1}{N H_{out} W_{out} (C_{out}/2)} \sum_n \sum_{c' \in g} \sum_h \sum_w y_{n,c',h,w}$$


$$\hat{y}_{n,c,h,w} = \frac{y_{n,c,h,w} - \mu_{GN,g}}{\sqrt{\sigma^2_{GN,g} + \varepsilon}}$$

Both followed by learnable scale and shift:

$$y'_{n,c,h,w} = \gamma_c \cdot \hat{y}_{n,c,h,w} + \beta_c$$

## Reinforcement Learning Agent (DQN)

The RL agent is a Deep Q-Network with architecture: 3 → 128 → 64 → 2 (ReLU activations):

- **Input (3)**: State vector = $[\mu_y,\ \sigma^2_y,\ \text{validation\loss\per\round}]$
- **Hidden layers**: 128 units → 64 units (ReLU)
- **Output (2)**: Q-values for actions {GN (0), BN (1)} 









# Train

    # FedNormRL
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet_fednn \
        --result_path results




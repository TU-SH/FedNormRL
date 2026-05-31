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

- **Input (3)**: State vector = $[\mu_y,\ \sigma^2_y,\ \text{validation-loss-per-round}]$
- **Hidden layers**: 128 units → 64 units (ReLU)
- **Output (2)**: Q-values for actions {GN (0), BN (1)}

### Action selection via ε-greedy policy

$$a = \begin{cases} \text{random action} \in \{0, 1\} & \text{with probability } \epsilon \\ \arg\max_a\, Q(\text{state}, a) & \text{with probability } 1 - \epsilon \end{cases}$$

### Q-value update (Bellman equation)

$$Q(\text{state}, a) \leftarrow Q(\text{state}, a) + \alpha \left[ r + \gamma_{RL} \cdot \max_{a'} Q(\text{state}', a') - Q(\text{state}, a) \right]$$

### Reward Signal 

$$r = -(\text{validation-loss}_{after} - \text{validation-loss}_{before})$$

A positive reward reinforces the chosen normalization strategy when validation loss decreases.

### DQN Training 

Backpropagation minimizes the squared TD error: 

$$L = \left[ r + \gamma_{RL} \cdot \max_{a'} Q(\text{state}', a') - Q(\text{state}, a) \right]^2$$

### MDP Formulation
|MDP Component | FedNormRL Implementation |
|--------------|--------------------------|
|State         | [feature map mean, feature map variance, validation loss per round]|
|Action        |  a=0 → Group Normalization; a=1 → Batch Normalization                        |
|Reward        |  Negative change in validation loss (positive if loss decreases)                       |
|Policy        |  ε-greedy with decaying ε (exploration → exploitation over rounds)                   |
|Value Function|  DQN Q-values updated via Bellman equation with Adam optimizer                       |


## Experimental Setup 

### Datasets
Three multi-domain benchmark datasets simulating non-IID feature shift (covariate shift / virtual concept drift):

|Dataset     | Domains/Subsets       |Task     |Drift Severity        | 
|------------|-----------------------|-----------------|--------------|
| DIGIT      |MNIST, MNIST-M, SVHN, SynthDigits, USPS|Digit classification (0–9)        |Mild-Moderate            |
|Office-Caltech         |Amazon, DSLR, Webcam, Caltech |Object classification (10 classes)|Moderate-Severe              |
|DomainNet| Real, Clipart, Painting, Sketch, Quickdraw, Infograph|Object classification (345 classes)|Severe          |  


- Train / Validation split: 90% / 10% (random sampling)
- Each domain assigned to a separate client, creating non-IID feature shift

### Model Architecture

CNN (modified LeNet-5) + AGN + DQN:

<img width="1440" height="1480" alt="image" src="https://github.com/user-attachments/assets/ad74dff8-a3be-4f83-9489-4eaef3294cee" />

### DQN Architecture

<img width="1440" height="848" alt="image" src="https://github.com/user-attachments/assets/3086c245-a634-4529-b914-926a5963627a" />

### Hyperparameters

|Parameter    | Value   |
|-------------|---------|
| Communication round            | 100     |
| Client participation probability           | 0.4         |
| Local SGD learning rate            | 0.01         |
| Batch size            |50         |
| DQN learning rate (Adam)            |0.001         |
| Discount factor $\gamma_{RL}$            |0.9         |
| Initial exploration rate $\epsilon$          |0.1         |
| $\epsilon$ decay per round            |0.999         |
| GN groups          |2         |

### Baseline
**FedNN** (Kang et al., 2024 — Pattern Recognition): FL framework using WN + AGN with `Gumbel-Softmax` trick for normalization selection. Unlike FedNormRL, FedNN's Gumbel-Softmax selection is fixed per client once chosen and does not learn sequentially across communication rounds.

**FedAvg** (McMahan et al., 2017): Standard federated averaging with no normalization — used for additional comparative analysis on DIGIT dataset. 

## Results 

Legend
- All - all clients participation
- Sel - selected clients participation

### Accuracy (%)

#### DIGIT Dataset

**Training Phase**
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  15.04       | 36.59             | 15.04          |     34.37          |
| Round 50           |  91.81       | 99.10              | 90.94          |    98.86           |
| Round 100          |  99.2        | 100                | 99.25         |     100          |


**Testing Phase**
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  11.51     | 25.63             | 11.56          |     24.55          |
| Round 50           |  75.06       | 81.48              | 74.37          |    81.24           |
| Round 100          |  77.79       | 81.62              | 77.46         |     81.53        |


<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/9c5d90f5-2bd9-4a83-8b1c-fb4ff60c2457" />
<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/16113f8d-2d8e-40ec-9a46-51f0cb7a4e5b" />



#### Office-Caltech Dataset

**Training Phase**
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  23.36       | 28.35             | 23.36          |     28.35          |
| Round 50           |  93.04       | 92.44             | 93.04          |    92.44          |
| Round 100          |  95.70       | 97.33              | 96.35         |     97.33         |

**Testing Phase**
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  22.24     | 27.36            | 22.24         |     27.36         |
| Round 50           |   56.69      | 64.37              | 56.69          |  64.37           |
| Round 100          |  55.91      | 65.35             | 55.91       |     65.35      |

<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/b1de0534-d381-4a7c-9697-5ac8668a4120" />

<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/3167de63-1582-4d8f-9c87-d48becbe4243" />

#### DomainNet Dataset
**Training Phase**
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  22.32       | 31.04            | 22.32          |     31.04          |
| Round 50           |  84.77       | 98.74             | 84.77          |    98.74          |
| Round 100          |  92.40       | 99.53              | 92.40         |     99.53         |

**Testing Phase**
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  21.78     | 30.87           | 21.78        |     30.87         |
| Round 50           |  47.57      | 59.88              | 47.57          |  59.88           |
| Round 100          |  47.68      | 61.16             | 47.68      |     61.16      |


<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/af1c2132-92b5-4e07-85f2-19243c1df945" />
<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/dbc92138-89af-4e45-9b01-5f040b62a340" />

### Validation Loss 

#### DIGIT Dataset 
|Communication Round | FedNN (all) | FedNormRL (all) | FedNN (sel) | FedNormRL (sel)| 
|--------------------|--------------|------------------|-----------|---------------|
| Round 1            |  2.289       | 2.0551            | 2.2769          |     1.8826          |
| Round 50           |  0.2846      | 0.0538            | 0.3026         |    0.0555        |
| Round 100          |  0.0588     | 0.0065            | 0.0596       |     0.0067        |

<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/4c68d7c9-3f9e-4dac-ba25-0cf6fe4fb2bf" />
<img width="401" height="335" alt="image" src="https://github.com/user-attachments/assets/4001fed8-5089-4d81-a525-39212adf6e7d" />


#### Office-Caltech Dataset 

#### DomainNet Dataset 










# Train

    # FedNormRL
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet_fednn \
        --result_path results




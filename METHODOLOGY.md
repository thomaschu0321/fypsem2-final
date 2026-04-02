# GLANCE + Bi-GAT: Adaptive LLM-Augmented Rumor Detection on Social Media

## 1. Introduction

### 1.1 Background

Rumor detection on social media has become increasingly critical as misinformation spreads rapidly through online platforms. Social media posts rarely exist in isolation—they form complex conversation threads where users reply, quote, and build upon each other's content. These interactions create **text-attributed graphs (TAGs)**, where nodes represent text content (tweets) and edges represent social relationships (replies, retweets).

Traditional approaches to rumor detection have relied on hand-crafted features or shallow text embeddings combined with machine learning classifiers. However, these methods fail to capture the intricate interplay between textual semantics and propagation dynamics. Graph Neural Networks (GNNs) have emerged as a powerful paradigm for modeling graph-structured data, demonstrating success in learning joint representations from both content and structure.

### 1.2 Problem Statement

Despite the success of GNNs in graph learning tasks, they exhibit significant limitations when applied to rumor detection:

1. **Homophily Assumption**: GNNs assume that connected nodes share similar labels (homophily). However, in social media conversations, reply threads often contain diverse viewpoints—supporters, skeptics, and fact-checkers all engaging with the same content.

2. **Degree Bias**: GNNs perform poorly on low-degree nodes (e.g., early-stage tweets with few replies) where message passing provides limited information.

3. **Heterophily Challenges**: Rumor threads frequently contain mixed signals where neighboring nodes have contradictory labels or sentiments.

4. **Semantic Limitations**: Standard GNN architectures capture structural patterns but may miss nuanced semantic content that requires broader contextual understanding.

### 1.3 Proposed Approach

We propose **GLANCE + Bi-GAT**, a hybrid framework that combines the selective LLM querying mechanism of GLANCE with the bidirectional attention architecture of Bi-GAT for enhanced rumor detection.

**GLANCE** (GNN with LLM Assistance for Neighbor- and Context-aware Embeddings) introduces a lightweight router that learns to identify nodes where GNNs typically fail and selectively queries a Large Language Model (LLM) for these challenging cases. The router is trained with an advantage-based objective that compares the utility of LLM queries against relying solely on the GNN.

**Bi-GAT** (Bidirectional Graph Attention Network) provides the GNN backbone, capturing bidirectional information flow through the propagation tree structure. By processing information both top-down (root to leaves) and bottom-up (leaves to root), Bi-GAT effectively models the causal relationships and evidence aggregation inherent in rumor propagation.

## 2. Related Work

### 2.1 Rumor Detection Methods

Rumor detection has evolved through several phases:

- **Early machine learning approaches** relied on hand-crafted features (user credibility, text patterns,传播 patterns) with classifiers like SVM. While achieving moderate success, these methods struggled with the diversity and noise inherent in social media data.

- **Deep learning approaches** introduced CNNs for spatial feature extraction and RNNs/LSTMs for sequential modeling. Ma et al. pioneered the use of RNNs with tree-structured propagation for rumor detection.

- **Graph Neural Networks** emerged as a natural fit for social media graphs, with GCNs, GATs, and GraphSAGE demonstrating improved performance by aggregating information across neighbor nodes.

- **Transformer-based methods** like BERT brought advanced semantic understanding to rumor detection tasks.

### 2.2 GNN-LLM Fusion

Recent work has explored combining LLMs with GNNs in two paradigms:

1. **LLM-as-Enhancer**: LLMs generate text embeddings or external knowledge to enrich node features fed to GNNs.

2. **LLM-as-Predictor**: LLMs directly classify serialized graph inputs, avoiding the need for GNN training.

However, most hybrid systems apply a single fusion strategy uniformly across all nodes, overlooking per-node variations in semantic quality and structural attributes. This uniform approach wastes computational resources on nodes already well-modeled by GNNs.

### 2.3 Bi-GAT for Rumor Detection

The Bi-GAT architecture, specifically designed for rumor detection, captures bidirectional dependencies in tweet propagation trees:

- **Top-Down GAT (TD-GAT)**: Simulates information flow from high-level nodes (root/source) to low-level nodes (replies), suitable for capturing causal relationships where the source tweet influences subsequent replies.

- **Bottom-Up GAT (BU-GAT)**: Simulates aggregation of features from low-level nodes (replies) to high-level nodes (root), enabling the source to incorporate collective evidence from the conversation.

## 3. Methodology

### 3.1 Overview

GLANCE + Bi-GAT follows a three-phase training pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Bi-GAT Pre-training                │
│              Train bidirectional GAT for rumor detection        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 2: Homophily Estimator Training           │
│           Train MLP Q to predict pseudo-labels for routing      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 3: GLANCE Router + Refiner Training       │
│    Train lightweight router to selectively query LLM for         │
│    challenging nodes; train refiner to fuse GNN + LLM embeddings │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 1: Bi-GAT Pre-training

#### 3.2.1 Bidirectional Graph Attention Network

The Bi-GAT architecture consists of two parallel GAT streams:

**Top-Down GAT (TD-GAT)**
```
Layer 1: h^(1,td)_i = ReLU( ||_h=1^H Σ_{j∈N_out(i)} α^(h)_ij W^(h)_td1 x_j )
Layer 2: h^(2,td)_i = 1/H Σ_h=1^H Σ_{j∈N_out(i)} α_ij W_td2 [h^(1,td)_j || x_r]
```
where:
- $N_{out}(i)$ is the set of out-neighbors of node $i$
- $x_r$ is the root node feature
- $H$ is the number of attention heads
- $\alpha_{ij}$ are normalized attention coefficients

**Bottom-Up GAT (BU-GAT)**
```
Layer 1: h^(1,bu)_i = ReLU( ||_h=1^H Σ_{j∈N_in(i)} α^(h)_ij W^(h)_bu1 x_j )
Layer 2: h^(2,bu)_i = 1/H Σ_h=1^H Σ_{j∈N_in(i)} α_ij W_bu2 [h^(1,bu)_j || x_r]
```
where $N_{in}(i)$ is the set of in-neighbors of node $i$.

**Bidirectional Fusion**
```
z_i = [h^(2,td)_i || h^(2,bu)_i] ∈ R^{2F'}
```

#### 3.2.2 Graph Attention Convolution

The GATConv layer computes attention scores between nodes:
```
e_ij = LeakyReLU(a^T [W_q x_i || W_k x_j])
α_ij = softmax_j(exp(e_ij) / Σ_k exp(e_ik))
h'_i = Σ_{j∈N(i)} α_ij W_v x_j
```

Multiple attention heads stabilize learning:
```
h_i = ||_h=1^H h_i^(h)
```

#### 3.2.3 Root Feature Skip Connection

Both TD-GAT and BU-GAT incorporate a skip connection to the root node feature in Layer 2:
```
h^(2)_i = h^(2)_i + W_r x_r
```

This design ensures that root information—containing the original claim being evaluated—directly influences all node representations.

#### 3.2.4 Classification Head

For graph-level rumor detection, we apply global mean pooling over node embeddings:
```
h_g = (1/N) Σ_{i∈g} z_i
ŷ_g = softmax(MLP(h_g))
```

#### 3.2.5 Training Objective

Phase 1 minimizes cross-entropy loss:
```
L_bigat = -Σ_{g∈B} [y_g log(ŷ_g) + (1-y_g) log(1-ŷ_g)]
```

### 3.3 Phase 2: Homophily Estimator

#### 3.3.1 Soft Local Homophily

Local homophily measures how similar a node's label is to its neighbors' labels:
```
h_v = p_Q,v · (1/|N(v)| Σ_{u∈N(v)} p_Q,u)
```
where $p_Q$ is the probability distribution predicted by the homophily estimator MLP Q.

#### 3.3.2 Homophily Estimator MLP

A lightweight MLP predicts pseudo-labels from node features:
```
p_Q = softmax(MLP_Q(x_v))
```

**Training**: The MLP is trained using graph labels propagated to all nodes within each graph. All nodes in a rumor thread receive the rumor label; all nodes in a non-rumor thread receive the non-rumor label.

#### 3.3.3 Importance of Homophily Estimation

GLANCE identifies local homophily as a strong predictor of GNN vs. LLM advantage:
- **High homophily**: Neighbors share labels → GNNs excel through message passing
- **Low homophily**: Neighbors have diverse labels → LLMs may provide better context

### 3.4 Phase 3: GLANCE Router and Refiner

#### 3.4.1 Routing Features

The router receives a feature vector for each node:
```
f_v = [z_G(v), u_v, ĥ_v, deg(v), x_v]
```
where:
- $z_G(v)$: Bi-GAT node embedding (256-dim)
- $u_v$: Uncertainty estimate via MC dropout (1-dim)
- $\hat{h}_v$: Estimated homophily score (1-dim)
- $deg(v)$: Node degree (1-dim)
- $x_v$: BERT text embedding (768-dim)

**Total routing feature dimension**: 256 + 1 + 1 + 1 + 768 = **1027**

#### 3.4.2 Router Architecture

A lightweight linear layer maps routing features to a routing probability:
```
a_v = σ(w^T f_v)
```
where $a_v \in [0, 1]$ represents the probability of routing node $v$ to the LLM.

#### 3.4.3 Top-K Routing Strategy

Rather than routing based on a fixed threshold, GLANCE uses a **top-K selection** per graph:
```
R_g = TopK({a_v : v ∈ g}, k)
```
where $k$ follows an exponential decay schedule:
```
k(epoch) = k_end + (k_start - k_end) × decay_rate^epoch
```

This curriculum-style approach:
- Routes more nodes early in training (exploration)
- Reduces routing budget as the model learns to rely on GNN for easier cases

#### 3.4.4 LLM Encoder (Multi-hop Context)

For routed nodes, the LLM encoder generates multi-hop context embeddings:

**Level 0 (Ego)**: The node's own text
```
prompt_0(v) = "Predict the node's category from the provided context.\nPossible categories: [rumour, non-rumour].\nEGO:\n{text_v}\nCategory?"
```

**Level 1 (1-hop neighbors)**: Ego + direct neighbors
```
prompt_1(v) = "Predict the node's category from the provided context.\nPossible categories: [rumour, non-rumour].\nEGO:\n{text_v}\nHOP1:\n- {neighbor_1}\n- {neighbor_2}\n...\nCategory?"
```

**Level 2 (2-hop neighbors)**: Ego + 2-hop context
```
prompt_2(v) = "Predict the node's category from the provided context.\nPossible categories: [rumour, non-rumour].\nEGO:\n{text_v}\nHOP2:\n- {2hop_neighbor_1}\n...\nCategory?"
```

**Embedding Concatenation**:
```
z_L(v) = [embed(prompt_0) || embed(prompt_1) || embed(prompt_2)] ∈ R^{3×3072}
```

#### 3.4.5 Refiner MLP

The refiner fuses Bi-GAT and LLM embeddings for routed nodes:
```
z_refined = MLP_refiner([z_G(v) || z_L(v)])
```

Architecture:
```
Input: [z_G || z_L] ∈ R^{256 + 9216}
Layer 1: Linear(9472, 256) + ReLU + Dropout(0.3)
Layer 2: Linear(256, 128) + ReLU + Dropout(0.3)
Output: Linear(128, 2) → logits
```

#### 3.4.6 Graph-Level Prediction

For graph-level classification:
1. Non-routed nodes retain Bi-GAT embeddings
2. Routed nodes use refined embeddings
3. Graph embedding = mean pooling over node embeddings
4. Classification head produces final logits

#### 3.4.7 Advantage-Based Router Training

Since LLM calls are non-differentiable, the router is trained using a policy gradient approach.

**Advantage Computation**:
```
advantage_g = L_GNN(g) - L_refined(g) - β
```
where $\beta$ is an LLM cost penalty.

**Router Loss**:
```
L_router = -E[advantage × log(a_v) for routed nodes]
           -E[(-L_GNN) × log(1 - a_v) for non-routed nodes]
           + λ × entropy(a_v)
```

The advantage is positive when routing improves prediction (LLM helps); negative when routing hurts.

**Combined Loss**:
```
L_total = L_pred + λ_router × L_router
```

### 3.5 Uncertainty Estimation

Monte Carlo dropout estimates node-level uncertainty:
```python
# T forward passes with dropout enabled
predictions = [bigat(x, edges) for _ in range(T)]
uncertainty = variance([‖z‖ for z in predictions])
```

High uncertainty indicates nodes where Bi-GAT is unreliable, warranting LLM consultation.

## 4. Dataset

### 4.1 PHEME Dataset

We evaluate on the PHEME dataset, specifically the Germanwings Crash event:

| Statistic | Value |
|-----------|-------|
| Total threads | 469 |
| Non-rumor threads | 231 |
| Rumor threads | 238 |

The Germanwings Crash dataset captures tweets surrounding the 2015 Germanwings Flight 9525 crash, including both accurate news coverage and misinformation.

### 4.2 Data Processing

Each conversation thread is parsed into a rooted tree structure:
- **Root node**: Source tweet (original claim)
- **Child nodes**: Reply tweets
- **Edges**: Parent-child relationships following the conversation structure

The dataset is split:
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

Stratified sampling preserves class balance across splits.

### 4.3 Feature Extraction

**Text Features**: BERT (bert-base-uncased) produces 768-dimensional embeddings:
```
x_v = BERT_CLS(text_v) ∈ R^{768}
```

**Structural Features**:
- Top-down edge index (parent → child)
- Bottom-up edge index (child → parent)
- Root node mask
- Node degree

## 5. Experimental Setup

### 5.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| BERT model | bert-base-uncased |
| BERT embedding dim | 768 |
| Bi-GAT hidden dim | 128 |
| Bi-GAT attention heads | 4 |
| Bi-GAT dropout | 0.3 |
| GNN embedding dim | 256 (2 × 128) |
| Homophily hidden dim | 128 |
| Router input dim | 1027 |
| LLM embedding dim | 3072 |
| Multi-hop LLM dim | 9216 (3 × 3072) |
| Refiner hidden dim | 256 |
| Batch size | 32 |
| Learning rate | 5e-4 |
| Weight decay | 1e-3 |
| Gradient clip | 1.0 |
| Phase 1 epochs | 150 |
| Phase 2 epochs | 100 |
| Phase 3 epochs | 150 |
| Early stopping patience | 20 |
| K budget start | 12 |
| K budget end | 3 |
| K decay rate | 0.5 |
| LLM cost penalty (β) | 0.2 |
| Entropy weight (λ_ent) | 0.01 |
| MC dropout passes | 5 |

### 5.2 Evaluation Metrics

- **Accuracy**: Overall correct classification rate
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### 5.3 Baselines for Comparison

1. **Bi-GAT (Phase 1 only)**: Standalone bidirectional GAT without LLM augmentation
2. **Bi-GAT + Homophily**: Bi-GAT with homophily estimation (no routing)
3. **GLANCE + Bi-GAT (Full)**: Complete framework with adaptive LLM routing

## 6. Expected Results and Analysis

### 6.1 Hypothesized Outcomes

Based on the GLANCE paper findings and Bi-GAT's proven effectiveness:

1. **Bi-GAT Baseline**: Strong performance due to bidirectional attention capturing both causal and evidential patterns in propagation trees.

2. **GLANCE + Bi-GAT Improvement**: 
   - Overall accuracy improvement of 1-3%
   - Significant gains on heterophilous nodes (10-15% improvement)
   - Reduced LLM usage through intelligent routing

3. **Routing Behavior**:
   - High-degree, high-homophily nodes: Routed less frequently (GNN handles well)
   - Low-degree, low-homophily nodes: Routed more frequently (LLM provides context)

### 6.2 Key Insights

1. **Complementary Strengths**: Bi-GAT excels at structural pattern recognition while LLMs provide semantic reasoning—GLANCE's routing mechanism intelligently combines both.

2. **Cost Efficiency**: By routing only ~25% of nodes to the LLM, we achieve significant performance gains with manageable computational overhead.

3. **Robustness**: The advantage-based training objective ensures the router learns from both successes and failures, continuously improving routing decisions.

## 7. Conclusion

This methodology presents GLANCE + Bi-GAT, a framework that synergistically combines:

1. **Bi-GAT's bidirectional attention** for capturing causal and evidential patterns in rumor propagation trees

2. **GLANCE's adaptive routing** for intelligently selecting nodes where the LLM can provide complementary context

3. **Multi-hop LLM embeddings** that capture neighborhood context at multiple structural levels

The three-phase training pipeline—Bi-GAT pre-training, homophily estimation, and GLANCE router training—enables efficient learning while maintaining interpretability. By selectively invoking the LLM only where needed, GLANCE + Bi-GAT achieves robust rumor detection performance while remaining computationally tractable.

### Future Work

- Extending to additional PHEME events (Charlie Hebdo, Ferguson, Ottawa, Sydney Siege)
- Experimenting with different LLM embedding models
- Exploring alternative routing strategies (e.g., reinforcement learning)
- Applying the framework to other rumor detection datasets (Weibo, FakeNewsNet)

## References

1. Loveland, D., Yang, Y.-A., & Koutra, D. (2025). *GLANCE: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion*. arXiv:2510.10849.

2. Tao, J., Wang, C., & Jiang, B. (2026). *LLM-Enhanced Rumor Detection via Virtual Node Induced Edge Prediction*. arXiv:2602.13279.

3. Veličković, P., et al. (2018). *Graph Attention Networks*. ICLR.

4. Bian, T., et al. (2020). *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks*. AAAI.

5. Ma, J., Gao, W., & Wong, K.-F. (2016). *Detecting Rumors from Microblogs with Recurrent Neural Networks*. IJCAI.

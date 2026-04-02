# 3. Methodology

## 3.1 Overview and Problem Formulation

Rumor detection on social media presents unique challenges that stem from the complex interplay between textual content and propagation structure. Social media posts rarely exist in isolation—they form conversation threads where users reply to, quote, or build upon each other's content. These interactions create **text-attributed graphs (TAGs)**, where each node represents a post with associated text, and edges represent social relationships such as reply or retweet actions.

Formally, we define a rumor detection dataset as a collection of propagation trees, where each tree $T = (V, E, T, y)$ consists of:

- $V = \{v_0, v_1, ..., v_n\}$: The set of nodes, where $v_0$ denotes the source post (root node) and each $v_i \in V \setminus \{v_0\}$ represents a reply post
- $E \subseteq V \times V$: The set of directed edges following the direction of information propagation (parent → child)
- $T = \{t_v\}_{v \in V}$: The text content associated with each node $v$
- $y \in \{0, 1\}$: The binary label, where $y = 1$ indicates a rumor and $y = 0$ indicates a non-rumor

The task is to learn a model $\psi: T \rightarrow y$ that predicts whether the source post is a rumor based on both the propagation structure and the textual content of all posts in the thread.

### 3.1.1 Challenges with Standard GNNs

Graph Neural Networks (GNNs) have demonstrated strong performance in various graph learning tasks by aggregating information from neighboring nodes. The standard message-passing framework computes node representations as:

$$h_v^{(\ell)} = \text{UPDATE}^{(\ell)}\left(h_v^{(\ell-1)}, \text{AGGREGATE}^{(\ell)}\left(\{h_u^{(\ell-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

where $\mathcal{N}(v)$ denotes the neighbors of node $v$. Despite their success, GNNs face significant limitations when applied to rumor detection:

1. **Homophily Assumption**: GNNs assume that connected nodes share similar labels (the homophily assumption). However, rumor threads often contain diverse viewpoints—supporters, skeptics, fact-checkers, and casual observers all engage with the same content. This **heterophily** violates the fundamental assumption underlying message passing.

2. **Degree Bias**: GNNs perform poorly on low-degree nodes (e.g., early-stage tweets with few replies) where message passing provides limited contextual information.

3. **Structural Limitations**: Standard GNN architectures capture local structural patterns but may miss broader contextual relationships that span multiple hops in the propagation tree.

### 3.1.2 The GLANCE + Bi-GAT Framework

We propose **GLANCE + Bi-GAT**, a hybrid framework that combines the selective LLM querying mechanism of GLANCE with the bidirectional attention architecture of Bi-GAT for enhanced rumor detection. The key insight is that GNNs and LLMs excel at different types of nodes:

- **GNNs** (specifically Bi-GAT) excel on nodes with high homophily and sufficient connectivity, where message passing effectively aggregates relevant neighborhood information.

- **LLMs** excel on nodes where the GNN struggles—heterophilous nodes with diverse neighbor labels, low-degree nodes with limited structural context, and nodes requiring semantic reasoning beyond structural patterns.

Rather than applying LLMs uniformly (which wastes computational resources) or using fixed heuristics (which are brittle and dataset-dependent), GLANCE employs a **learned router** that adaptively decides which nodes benefit from LLM consultation.

The framework consists of three main components:

1. **Bi-GAT Backbone**: A bidirectional graph attention network that captures both causal (top-down) and evidential (bottom-up) information flow through the propagation tree.

2. **Homophily Estimator**: An MLP that predicts pseudo-labels from node features, enabling the estimation of local homophily—a strong signal for identifying nodes where GNNs typically fail.

3. **GLANCE Router + Refiner**: A lightweight router that decides which nodes to query the LLM, and a refiner MLP that fuses Bi-GAT and LLM embeddings for refined predictions.

## 3.2 Bi-GAT: Bidirectional Graph Attention Network

### 3.2.1 Motivation for Bidirectional Processing

Rumor propagation follows a distinctive tree structure where information flows in two complementary directions:

1. **Top-Down Flow**: The source post (root) influences subsequent replies. The root contains the original claim being evaluated, and this information propagates downward through the tree. Understanding how replies relate to the source requires modeling this causal direction.

2. **Bottom-Up Flow**: Replies provide evidence about the source. Supportive comments, corrective responses, and expressions of doubt all aggregate information that informs the veracity of the root claim. Capturing this collective evidence requires aggregating from leaves toward the root.

Standard GNNs (GCN, GAT) process edges bidirectionally but without distinguishing these semantic roles. Bi-GAT explicitly models both directions as separate processing streams, enabling the network to learn direction-specific attention patterns.

### 3.2.2 Graph Attention Convolution

The fundamental operation in Bi-GAT is the **Graph Attention Convolution (GATConv)**, which computes node representations by attending over neighboring nodes with learned attention coefficients.

Given a graph with node features $\mathbf{x}_i \in \mathbb{R}^F$, the GATConv layer computes attention scores between each pair of connected nodes:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}_q \mathbf{x}_i \| \mathbf{W}_k \mathbf{x}_j]\right)$$

where:
- $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{F' \times F}$ are learnable weight matrices
- $\mathbf{a} \in \mathbb{R}^{2F'}$ is the attention parameter vector
- $[\cdot \| \cdot]$ denotes concatenation

The attention coefficients are then normalized across all neighbors using softmax:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

Finally, the node representation is updated by weighted aggregation of neighbor features:

$$\mathbf{h}'_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_v \mathbf{x}_j$$

where $\mathbf{W}_v \in \mathbb{R}^{F' \times F}$ is the value projection matrix.

**Multi-Head Attention**: To stabilize learning and capture diverse relationship types, Bi-GAT employs multiple attention heads. Each head $h = 1, ..., H$ has its own parameters $\mathbf{W}_v^{(h)}$, $\mathbf{a}^{(h)}$, and produces an independent representation. The outputs are concatenated:

$$\mathbf{h}_i = \|_{h=1}^{H} \mathbf{h}_i^{(h)} \in \mathbb{R}^{H \cdot F'}$$

### 3.2.3 Top-Down GAT (TD-GAT)

The Top-Down stream simulates information transmission from high-level nodes (root/source) to low-level nodes (replies). This captures **causal relationships** where the source claim influences downstream engagement.

**Layer 1**: Multi-head attention over out-neighbors (nodes that receive information from the current node):

$${\mathbf{h}_i^{(1,td)}}^{(h)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}_{out}(i)} \alpha_{ij}^{(h)} \mathbf{W}_{td1}^{(h)} \mathbf{x}_j\right)$$

where $\mathcal{N}_{out}(i)$ is the set of out-neighbors of node $i$.

**Layer 2**: The second layer incorporates a **root skip connection**, concatenating the neighbor-aggregated representation with the root node feature:

$${\mathbf{h}_i^{(2,td)}}^{(h)} = \frac{1}{H} \sum_{h=1}^{H} \sum_{j \in \mathcal{N}_{out}(i)} \alpha_{ij} \mathbf{W}_{td2}\left[{\mathbf{h}_j^{(1,td)}} \| \mathbf{x}_r\right]$$

where $\mathbf{x}_r$ is the feature vector of the root node $v_0$. This skip connection ensures that information from the source post directly influences all node representations, preventing information dilution through multiple aggregation steps.

The output of TD-GAT is a single vector per node:

$$h_i^{td} = {\mathbf{h}_i^{(2,td)}} \in \mathbb{R}^{F'}$$

### 3.2.4 Bottom-Up GAT (BU-GAT)

The Bottom-Up stream simulates aggregation of features from low-level nodes (replies) to high-level nodes (root and intermediate nodes). This captures **evidential relationships** where replies provide supportive or corrective evidence.

**Layer 1**: Multi-head attention over in-neighbors (nodes that send information to the current node):

$${\mathbf{h}_i^{(1,bu)}}^{(h)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}_{in}(i)} \alpha_{ij}^{(h)} \mathbf{W}_{bu1}^{(h)} \mathbf{x}_j\right)$$

where $\mathcal{N}_{in}(i)$ is the set of in-neighbors of node $i$.

**Layer 2**: Similar to TD-GAT, the second layer incorporates a root skip connection:

$${\mathbf{h}_i^{(2,bu)}}^{(h)} = \frac{1}{H} \sum_{h=1}^{H} \sum_{j \in \mathcal{N}_{in}(i)} \alpha_{ij} \mathbf{W}_{bu2}\left[{\mathbf{h}_j^{(1,bu)}} \| \mathbf{x}_r\right]$$

The output of BU-GAT is:

$$h_i^{bu} = {\mathbf{h}_i^{(2,bu)}} \in \mathbb{R}^{F'}$$

### 3.2.5 Bidirectional Fusion

The final Bi-GAT embedding for each node is the concatenation of its top-down and bottom-up representations:

$$z_i^{BiGAT} = [h_i^{td} \| h_i^{bu}] \in \mathbb{R}^{2F'}$$

This fusion preserves complementary information from both processing streams. The top-down stream captures how the source influences each node, while the bottom-up stream captures how each node contributes to understanding the source.

### 3.2.6 Graph-Level Classification

For rumor detection, we need to classify entire propagation trees rather than individual nodes. Bi-GAT applies **global mean pooling** over node embeddings to obtain a graph-level representation:

$$\mathbf{h}_g = \frac{1}{|V_g|} \sum_{i \in V_g} z_i^{BiGAT}$$

The graph-level prediction is then:

$$\hat{y}_g = \text{softmax}\left(\text{MLP}_{class}(\mathbf{h}_g)\right)$$

### 3.2.7 Phase 1: Pre-training Objective

In Phase 1, the Bi-GAT backbone is pre-trained for graph-level rumor detection using standard cross-entropy loss:

$$\mathcal{L}_{BiGAT} = -\sum_{g \in \mathcal{B}} \left[ y_g \log(\hat{y}_g) + (1 - y_g) \log(1 - \hat{y}_g) \right]$$

where $\mathcal{B}$ is a mini-batch of propagation trees and $y_g$ is the ground-truth label.

The Bi-GAT backbone is trained with:
- AdamW optimizer with learning rate $\eta = 5 \times 10^{-4}$
- Weight decay $\lambda = 10^{-3}$
- Gradient clipping at 1.0
- Early stopping with patience of 20 epochs

## 3.3 Homophily Estimator

### 3.3.1 The Role of Local Homophily

A key insight from the GLANCE paper is that **local homophily** is a strong predictor of when GNNs vs. LLMs will perform better. Local homophily measures the label similarity between a node and its neighbors:

$$h_v = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbb{1}[y_u = y_v]$$

Nodes with **high homophily** tend to be surrounded by nodes with the same label, making them well-suited for GNN message passing. Nodes with **low homophily** (heterophilous nodes) have diverse neighbor labels, making GNN aggregation less reliable but potentially better suited for LLM reasoning.

However, true homophily requires access to ground-truth labels, which are unavailable during inference. We therefore train a proxy estimator.

### 3.3.2 MLP Q Architecture

The **Homophily Estimator** is a lightweight MLP $Q$ that predicts node labels from features alone:

$$\hat{y}_v = \arg\max Q(\mathbf{x}_v)$$

The architecture consists of:
- Input: 768-dimensional BERT embedding
- Hidden layer: 128 units with ReLU activation
- Dropout: 0.3
- Output: 2-class logits (rumor/non-rumor)

### 3.3.3 Training with Pseudo-Labels

Since we don't have node-level labels, we propagate the graph-level label to all nodes within each tree:

$$\hat{y}_v^{(pseudo)} = y_g \quad \forall v \in V_g$$

This is a reasonable proxy because all posts in a rumor thread are related to the same source claim. While individual replies may have varying stances, they are all contextually linked to the source being evaluated.

### 3.3.4 Soft Local Homophily Estimation

Beyond hard pseudo-labels, we compute a **soft local homophily** estimate that captures the probability distribution over labels:

$$p_{Q,v} = \text{softmax}(Q(\mathbf{x}_v)) \in \mathbb{R}^C$$

where $C = 2$ is the number of classes. The soft local homophily is then computed as:

$$\hat{h}_v = p_{Q,v} \cdot \left(\frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} p_{Q,u}\right)$$

This measures the alignment between a node's predicted label distribution and its neighbors' predicted distributions. A node surrounded by neighbors with similar predicted labels will have high $\hat{h}_v$, while heterophilous nodes will have low values.

### 3.3.5 Phase 2: Pre-training Objective

The Homophily Estimator is trained with cross-entropy loss on node-level pseudo-labels:

$$\mathcal{L}_{Q} = -\sum_{v \in \mathcal{V}} \left[ \hat{y}_v^{(pseudo)} \log(p_{Q,v}) + (1 - \hat{y}_v^{(pseudo)}) \log(1 - p_{Q,v}) \right]$$

After Phase 2 training, the Homophily Estimator is frozen and used as a fixed component in Phase 3.

## 3.4 GLANCE: Adaptive LLM-Augmented Rumor Detection

### 3.4.1 Overview of the GLANCE Framework

GLANCE (GNN with LLM Assistance for Neighbor- and Context-aware Embeddings) is designed to leverage the complementary strengths of Bi-GAT and LLMs. The key insight is that different nodes require different processing strategies:

- Nodes where Bi-GAT performs well (high homophily, sufficient connectivity) should rely on Bi-GAT embeddings
- Nodes where Bi-GAT struggles (heterophilous, low-degree, semantically complex) should additionally consult the LLM

GLANCE consists of three steps:

1. **Generate routing features** for each node using Bi-GAT embeddings, uncertainty estimates, homophily scores, degree, and original features

2. **Route challenging nodes** to the LLM based on the router's decision

3. **Refine predictions** by fusing Bi-GAT and LLM embeddings for routed nodes

### 3.4.2 Step 1: Generating Routing Features

The router receives a feature vector $\mathbf{f}_v \in \mathbb{R}^{1027}$ for each node, consisting of five signals:

**1. Bi-GAT Embedding** $\mathbf{z}_G(v) \in \mathbb{R}^{256}$: The 256-dimensional bidirectional GAT embedding capturing structural patterns learned by the pre-trained Bi-GAT.

**2. Uncertainty Estimate** $u_v \in \mathbb{R}^1$: Estimated via **Monte Carlo (MC) dropout**. We perform multiple forward passes with dropout enabled and compute the variance of node embedding norms:

```python
# T forward passes with dropout enabled
predictions = [BiGAT(x, edges) for _ in range(T)]
uncertainty = variance([‖z‖ for z in predictions])
```

High uncertainty indicates nodes where the Bi-GAT is unreliable, warranting LLM consultation.

**3. Estimated Homophily** $\hat{h}_v \in \mathbb{R}^1$: The soft local homophily score from the Homophily Estimator. Low homophily indicates heterophilous nodes where LLMs may excel.

**4. Node Degree** $\deg(v) \in \mathbb{R}^1$: The number of connections in the propagation tree. Low-degree nodes have limited structural context for message passing.

**5. Original BERT Features** $\mathbf{x}_v \in \mathbb{R}^{768}$: The raw text embedding from BERT, providing semantic context for the router.

**Routing Feature Vector**:

$$\mathbf{f}_v = [\mathbf{z}_G(v) \| u_v \| \hat{h}_v \| \deg(v) \| \mathbf{x}_v] \in \mathbb{R}^{256 + 1 + 1 + 1 + 768} = \mathbb{R}^{1027}$$

### 3.4.3 Step 2: The Node Router

The **Node Router** $\pi$ is a lightweight linear layer that maps routing features to a routing probability:

$$a_v = \pi(\mathbf{f}_v) = \sigma(\mathbf{w}^\top \mathbf{f}_v) \in [0, 1]$$

where $\sigma$ is the sigmoid function. A higher $a_v$ indicates that the node is more likely to benefit from LLM consultation.

**Top-K Selection Strategy**: Rather than applying a fixed threshold (which requires calibration), GLANCE uses a **top-K selection** strategy per mini-batch:

$$R_g = \text{TopK}\left(\{a_v : v \in g\}, k\right)$$

where $k$ is the routing budget (number of nodes routed per graph) and $R_g$ is the set of routed nodes in graph $g$.

**Curriculum Routing (K-Decay)**: The routing budget follows an exponential decay schedule during training:

$$k(\text{epoch}) = k_{end} + (k_{start} - k_{end}) \cdot \gamma^{\text{epoch} - 1}$$

where:
- $k_{start} = 12$: Initial budget (route most nodes for exploration)
- $k_{end} = 3$: Final budget (route fewer nodes as model learns)
- $\gamma = 0.5$: Decay rate

This curriculum-style approach starts with aggressive routing (helping the router learn from diverse examples) and gradually reduces to selective routing (focusing on high-value cases).

### 3.4.4 Step 3: Multi-Hop LLM Encoder

For each routed node, the **LLM Encoder** generates contextual embeddings by processing the node's text in the context of its neighborhood. Unlike standard approaches that generate a single embedding, GLANCE produces **multi-hop embeddings** at three structural levels:

**Level 0 (Ego)**: The node's own text only
```
Prompt_0(v): "Predict the node's category from the provided context.
              Possible categories: [rumour, non-rumour].
              EGO:
              {text_v}
              Category?"
```

**Level 1 (1-hop)**: The ego node plus its direct neighbors
```
Prompt_1(v): "Predict the node's category from the provided context.
              Possible categories: [rumour, non-rumour].
              EGO:
              {text_v}
              HOP1:
              - {neighbor_1_text}
              - {neighbor_2_text}
              ...
              Category?"
```

**Level 2 (2-hop)**: The ego node plus 2-hop neighborhood
```
Prompt_2(v): "Predict the node's category from the provided context.
              Possible categories: [rumour, non-rumour].
              EGO:
              {text_v}
              HOP2:
              - {2hop_neighbor_1_text}
              - {2hop_neighbor_2_text}
              ...
              Category?"
```

Each prompt is encoded using the LLM embedding API (text-embedding-3-large) to produce a 3072-dimensional embedding. The three level embeddings are concatenated:

$$\mathbf{z}_L(v) = [\mathbf{z}_{L,0}(v) \| \mathbf{z}_{L,1}(v) \| \mathbf{z}_{L,2}(v)] \in \mathbb{R}^{9216}$$

This multi-hop design aligns with the aggregation patterns in advanced GNNs and captures:
- **Ego context**: The core semantic content
- **1-hop context**: Direct engagement (replies, quotes)
- **2-hop context**: Broader conversational threads

**Neighborhood Sampling**: To manage computational cost and prompt length, we sample up to 5 neighbors per hop. This is particularly important for PHEME threads with many replies.

**Embedding Caching**: The LLM Encoder maintains a cache to avoid re-encoding identical prompts across training epochs, significantly reducing API calls.

### 3.4.5 Step 4: Refiner MLP

The **Refiner MLP** $C$ fuses Bi-GAT and LLM embeddings for routed nodes:

$$\hat{p}_{C,v} = \text{softmax}\left(C\left(\left[\mathbf{z}_G(v) \| \mathbf{z}_L(v)\right]\right)\right)$$

Architecture:
- Input: Concatenation of Bi-GAT (256-dim) and LLM (9216-dim) embeddings → 9472 dimensions
- Layer 1: Linear(9472, 256) + ReLU + Dropout(0.3)
- Layer 2: Linear(256, 128) + ReLU + Dropout(0.3)
- Output: Linear(128, 2) → class logits

### 3.4.6 Graph-Level Refined Prediction

For graph-level classification:
1. **Non-routed nodes**: Retain their Bi-GAT embeddings $\mathbf{z}_G(v)$
2. **Routed nodes**: Replace embeddings with refined signal (softmax of refiner logits expanded to embedding space)
3. **Graph embedding**: Mean pooling over all node embeddings
4. **Classification**: Graph-level classifier produces final logits

The refined graph embedding is:

$$\mathbf{h}_g^{refined} = \frac{1}{|V_g|} \sum_{v \in V_g} \tilde{\mathbf{z}}_v$$

where $\tilde{\mathbf{z}}_v = \mathbf{z}_G(v) + \text{Expand}(\text{softmax}(\text{Refiner}(\mathbf{z}_G(v), \mathbf{z}_L(v))))$ for routed nodes.

### 3.4.7 Training the Router: Advantage-Based Policy Gradient

Since LLM calls are non-differentiable (discrete API calls), the router cannot be trained via standard backpropagation. We use a **policy gradient-inspired approach** that treats routing as a contextual bandit problem.

**Counterfactual Rewards**: For each routed node, we compare the loss when using the LLM vs. relying solely on Bi-GAT:

For a routed node $v$:
$$\ell_{GNN}^{(v)} = -\sum_{k=1}^{C} \mathbb{1}[y_v = k] \log p_{GNN,k}^{(v)}$$
$$\ell_{LLM}^{(v)} = -\sum_{k=1}^{C} \mathbb{1}[y_v = k] \log p_{C,k}^{(v)}$$

The **advantage** of routing is:
$$r_v = \ell_{GNN}^{(v)} - \ell_{LLM}^{(v)} - \beta$$

where $\beta = 0.2$ is an LLM cost penalty that discourages unnecessary LLM calls.

For non-routed nodes:
$$r_v = -\ell_{GNN}^{(v)}$$

**Interpretation**:
- Positive $r_v$: LLM reduced prediction loss enough to offset its cost → good routing decision
- Negative $r_v$: LLM hurt performance or wasn't used when it could have helped → poor routing decision

**Router Loss**: The router is optimized to maximize expected advantage:

$$\mathcal{L}_{router} = -\mathbb{E}[r_v \cdot \log \pi(\mathbf{f}_v)] + \lambda_{ent} \cdot H[\pi]$$

where:
- $\log \pi(\mathbf{f}_v) = \log a_v$ for routed nodes (encourage high probability)
- $\log (1 - \pi(\mathbf{f}_v)) = \log(1 - a_v)$ for non-routed nodes (discourage unnecessary routing)
- $H[\pi] = -(a_v \log a_v + (1 - a_v) \log(1 - a_v))$ is the entropy bonus with weight $\lambda_{ent} = 0.01$

**Combined Loss**: The final training objective combines prediction loss and routing loss:

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda_{router} \cdot \mathcal{L}_{router}$$

where $\mathcal{L}_{pred} = -\sum_{g \in \mathcal{B}} [y_g \log(\hat{y}_g^{refined}) + (1 - y_g) \log(1 - \hat{y}_g^{refined})]$.

### 3.4.8 Training Procedure

Phase 3 training proceeds as follows:

1. **Freeze Bi-GAT and Homophily Estimator**: These components are pre-trained and not updated during GLANCE training.

2. **Initialize LLM Encoder**: The LLM encoder is a frozen API interface (no training, just inference).

3. **Train Router and Refiner**: Only the router $\pi$ and refiner MLP $C$ are trained, along with the graph-level classifier heads.

4. **Forward Pass**:
   - Generate Bi-GAT embeddings (frozen, no gradients)
   - Compute routing features and scores
   - Select top-K nodes per graph
   - Query LLM for routed nodes (frozen)
   - Compute refined embeddings via refiner
   - Aggregate to graph level and classify

5. **Backward Pass**:
   - Compute prediction loss on refined graph predictions
   - Compute counterfactual rewards and router loss
   - Update router and refiner parameters only

## 3.5 Data Processing Pipeline

### 3.5.1 Text Feature Extraction

Each tweet is encoded using **BERT (bert-base-uncased)** to produce a 768-dimensional text embedding:

$$\mathbf{x}_v = \text{BERT}_{CLS}(t_v) \in \mathbb{R}^{768}$$

The [CLS] token representation captures the overall semantic meaning of the tweet and serves as the node feature for all subsequent processing.

### 3.5.2 Graph Construction

Each PHEME thread is parsed into a rooted tree structure:

- **Root node**: The source tweet (original claim)
- **Child nodes**: All reply tweets in the thread
- **Top-down edges**: Parent → Child (direction of information propagation)
- **Bottom-up edges**: Child → Parent (reverse direction)
- **Root mask**: Boolean vector marking the source node

### 3.5.3 Dataset Split

The PHEME Germanwings Crash dataset is split with stratification:
- **Training**: 70% of threads
- **Validation**: 15% of threads
- **Testing**: 15% of threads

Stratified sampling ensures consistent class ratios across splits.

## 3.6 Implementation Details

### 3.6.1 Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| **BERT Encoder** | Model | bert-base-uncased |
| | Embedding dimension | 768 |
| | Max sequence length | 128 |
| **Bi-GAT** | Hidden dimension | 128 |
| | Attention heads | 4 |
| | Dropout | 0.3 |
| | Number of layers | 2 |
| | GNN embedding dim | 256 (2 × 128) |
| **Homophily Estimator** | Hidden dimension | 128 |
| | Output classes | 2 |
| | Dropout | 0.3 |
| **Router** | Input dimension | 1027 |
| | Output | sigmoid probability |
| | Initial K budget | 12 |
| | Final K budget | 3 |
| | K decay rate | 0.5 |
| **LLM Encoder** | Model | text-embedding-3-large |
| | Embedding dimension | 3072 |
| | Multi-hop dim | 9216 (3 × 3072) |
| | Batch size | 64 |
| | Neighbor sample size | 5 |
| **Refiner MLP** | Input dimension | 9472 (256 + 9216) |
| | Hidden dimensions | 256 → 128 |
| | Output | 2 (logits) |
| **Training** | Batch size | 32 |
| | Learning rate | 5 × 10⁻⁴ |
| | Weight decay | 10⁻³ |
| | Gradient clip | 1.0 |
| | Phase 1 epochs | 150 |
| | Phase 2 epochs | 100 |
| | Phase 3 epochs | 150 |
| | Early stopping patience | 20 |
| | LLM cost penalty (β) | 0.2 |
| | Entropy weight (λ_ent) | 0.01 |
| | Router loss weight (λ_router) | 1.0 |
| | MC dropout passes | 5 |

### 3.6.2 Computational Considerations

- **Bi-GAT and Homophily Estimator** are trained on GPU with efficient message passing via PyTorch Geometric
- **LLM Encoder** uses API calls to a remote embedding service; responses are cached to minimize redundant requests
- **Router training** maintains a balance between exploration (high K) and exploitation (low K) through curriculum scheduling

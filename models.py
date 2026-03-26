"""
Model components for GLANCE + Bi-GAT Rumor Detection.

Components:
    1. BiGAT          - Bidirectional Graph Attention Network backbone
    2. HomophilyEstimator - MLP Q for estimating local homophily
    3. NodeRouter      - Lightweight router deciding which nodes to query LLM
    4. LLMEncoder      - Multi-hop BERT encoder for routed nodes
    5. RefinerMLP      - Fuses GNN + LLM embeddings
    6. BiGATClassifier - Standalone Bi-GAT for Phase 1 pre-training
    7. GLANCE          - Full GLANCE framework combining all components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import degree as compute_degree
from collections import defaultdict

import config


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Bi-GAT Backbone
# ═══════════════════════════════════════════════════════════════════════════════

class TopDownGAT(nn.Module):
    """Top-down GAT: information flows from root (parent) to leaves (children)."""

    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        # Layer 2 input: hidden_dim * heads (concat), output: hidden_dim (average heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=False)
        # Root feature projection for skip connection in layer 2
        self.root_proj = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, td_edge_index, root_features, batch=None):
        """
        Args:
            x: [N, in_dim] node features
            td_edge_index: [2, E_td] top-down (parent->child) edges
            root_features: [N, in_dim] root features broadcast to all nodes
            batch: [N] batch assignment
        Returns:
            h_td: [N, hidden_dim] top-down embeddings
        """
        # Layer 1: multi-head attention with concat, ReLU
        h = self.conv1(x, td_edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 2: multi-head attention with average, concat with root features
        # Concatenate hidden with projected root features before conv2
        root_proj = self.root_proj(root_features)
        h_with_root = h  # conv2 takes hidden_dim * heads input
        h = self.conv2(h_with_root, td_edge_index)

        # Add root skip connection
        h = h + root_proj

        return h


class BottomUpGAT(nn.Module):
    """Bottom-up GAT: information flows from leaves (children) to root (parent)."""

    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.root_proj = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bu_edge_index, root_features, batch=None):
        """
        Args:
            x: [N, in_dim] node features
            bu_edge_index: [2, E_bu] bottom-up (child->parent) edges
            root_features: [N, in_dim] root features broadcast to all nodes
            batch: [N] batch assignment
        Returns:
            h_bu: [N, hidden_dim] bottom-up embeddings
        """
        h = self.conv1(x, bu_edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        root_proj = self.root_proj(root_features)
        h = self.conv2(h, bu_edge_index)
        h = h + root_proj

        return h


class BiGAT(nn.Module):
    """
    Bidirectional Graph Attention Network.
    Combines top-down and bottom-up GAT to capture bidirectional information flow.
    Output: concatenation of TD and BU embeddings per node.
    """

    def __init__(self, in_dim=config.BERT_FEATURE_DIM,
                 hidden_dim=config.BIGAT_HIDDEN_DIM,
                 heads=config.BIGAT_HEADS,
                 dropout=config.BIGAT_DROPOUT):
        super().__init__()
        self.td_gat = TopDownGAT(in_dim, hidden_dim, heads, dropout)
        self.bu_gat = BottomUpGAT(in_dim, hidden_dim, heads, dropout)
        self.embed_dim = 2 * hidden_dim  # TD + BU concatenated

    def forward(self, x, td_edge_index, bu_edge_index, root_mask, batch=None):
        """
        Args:
            x: [N, in_dim]
            td_edge_index: [2, E_td]
            bu_edge_index: [2, E_bu]
            root_mask: [N] bool, True for root nodes
            batch: [N] batch assignment vector
        Returns:
            z_G: [N, embed_dim] node embeddings (TD || BU)
        """
        # Broadcast root features to all nodes in each graph
        root_features = self._broadcast_root(x, root_mask, batch)

        h_td = self.td_gat(x, td_edge_index, root_features, batch)
        h_bu = self.bu_gat(x, bu_edge_index, root_features, batch)

        z_G = torch.cat([h_td, h_bu], dim=-1)  # [N, 2*hidden_dim]
        return z_G

    def _broadcast_root(self, x, root_mask, batch):
        """Broadcast root node features to all nodes in the same graph."""
        if batch is None:
            # Single graph: all nodes get the same root
            root_feat = x[root_mask][0]  # [in_dim]
            return root_feat.unsqueeze(0).expand(x.size(0), -1)

        # Batched: each graph has its own root
        num_graphs = batch.max().item() + 1
        root_features = torch.zeros_like(x)

        for g in range(num_graphs):
            graph_mask = batch == g
            graph_root_mask = root_mask & graph_mask
            if graph_root_mask.any():
                root_feat = x[graph_root_mask][0]
                root_features[graph_mask] = root_feat.unsqueeze(0).expand(
                    graph_mask.sum().item(), -1)

        return root_features


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Homophily Estimator
# ═══════════════════════════════════════════════════════════════════════════════

class HomophilyEstimator(nn.Module):
    """
    MLP Q that predicts pseudo-labels from node features.
    Used to estimate soft local homophily (GLANCE Eq. 1).
    """

    def __init__(self, in_dim=config.BERT_FEATURE_DIM,
                 hidden_dim=config.HOMOPHILY_HIDDEN_DIM,
                 num_classes=config.HOMOPHILY_NUM_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """Returns logits [N, num_classes]."""
        return self.mlp(x)

    def predict_proba(self, x):
        """Returns probability distribution [N, num_classes]."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def compute_soft_homophily(proba, edge_index, num_nodes):
    """
    Compute soft local homophily estimate (GLANCE Eq. 1).
    h_hat_v = p_Q,v . (1/|N(v)| * sum_{u in N(v)} p_Q,u)

    Args:
        proba: [N, C] probability distributions
        edge_index: [2, E] undirected edges
        num_nodes: int
    Returns:
        homophily: [N] soft homophily scores
    """
    src, dst = edge_index
    # Aggregate neighbor probabilities
    neighbor_sum = torch.zeros(num_nodes, proba.size(1), device=proba.device)
    neighbor_count = torch.zeros(num_nodes, 1, device=proba.device)

    neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, proba.size(1)), proba[src])
    neighbor_count.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, device=proba.device))

    # Avoid division by zero
    neighbor_count = neighbor_count.clamp(min=1)
    neighbor_avg = neighbor_sum / neighbor_count  # [N, C]

    # Dot product: p_v . avg_neighbor_p
    homophily = (proba * neighbor_avg).sum(dim=-1)  # [N]
    return homophily


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Node Router
# ═══════════════════════════════════════════════════════════════════════════════

class NodeRouter(nn.Module):
    """
    Lightweight router that decides whether to query the LLM for each node.
    Input: routing features f_v = [z_G(v), uncertainty, homophily, degree, x_v]
    Output: a_v = sigma(w^T f_v) in [0, 1]
    """

    def __init__(self, input_dim=config.ROUTER_INPUT_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, routing_features):
        """
        Args:
            routing_features: [N, input_dim]
        Returns:
            scores: [N] routing probabilities
        """
        return torch.sigmoid(self.linear(routing_features).squeeze(-1))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLM Encoder (Multi-hop BERT)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMEncoder(nn.Module):
    """
    Multi-hop LLM embedding encoder using a real embedding API.
    Sends serialized neighborhood text prompts to the LLM embedding API
    and produces embeddings at 3 structural levels (GLANCE Sec 5.1.2):

        z_L0: ego text only
        z_L1: ego + 1-hop neighbor texts (concatenated with [SEP])
        z_L2: ego + 2-hop neighbor texts (concatenated with [SEP])

    Output: z_L(v) = [z_L0 || z_L1 || z_L2], dim = 3 * 3072 = 9216
    No trainable parameters - this is a frozen LLM component.
    """

    def __init__(self, neighbor_sample_size=config.NEIGHBOR_SAMPLE_SIZE):
        super().__init__()
        self.neighbor_sample_size = neighbor_sample_size
        self.embed_dim = config.LLM_API_EMBED_DIM  # 3072

        from openai import OpenAI
        self.client = OpenAI(
            base_url=config.LLM_API_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
        self.model_name = config.LLM_EMBEDDING_MODEL

        # Cache to avoid re-encoding identical prompts
        self._cache = {}

    def forward(self, routed_mask, x, edge_index, raw_texts, device):
        """
        Generate multi-hop LLM embeddings for routed nodes via API.

        Args:
            routed_mask: [N] bool mask of routed nodes
            x:           [N, feat_dim] (unused, kept for interface compat)
            edge_index:  [2, E] bidirectional edges
            raw_texts:   list[str] of length N, raw tweet text per node
            device:      torch device
        Returns:
            z_L: [K, 9216] multi-hop LLM embeddings (K = num routed)
        """
        routed_indices = routed_mask.nonzero(as_tuple=True)[0]
        K = routed_indices.size(0)

        if K == 0:
            return torch.zeros(0, config.LLM_EMBED_DIM, device=device)

        # Build adjacency from edge_index
        adj = defaultdict(set)
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            adj[s].add(d)

        # Build prompts for 3 structural levels
        ego_prompts = []
        hop1_prompts = []
        hop2_prompts = []

        for nid in routed_indices.tolist():
            ego_text = raw_texts[nid] if nid < len(raw_texts) else ""
            if not ego_text.strip():
                ego_text = "[empty tweet]"

            # Level 0: ego only
            ego_prompts.append(
                f"Predict the node's category from the provided context.\n"
                f"Possible categories: [rumour, non-rumour].\n"
                f"EGO:\n{ego_text}\nCategory?"
            )

            # Level 1: ego + 1-hop neighbors
            neighbors_1 = list(adj.get(nid, set()))[:self.neighbor_sample_size]
            hop1_parts = [ego_text]
            for nb in neighbors_1:
                nb_text = raw_texts[nb] if nb < len(raw_texts) else ""
                if nb_text.strip():
                    hop1_parts.append(nb_text)
            hop1_prompts.append(
                f"Predict the node's category from the provided context.\n"
                f"Possible categories: [rumour, non-rumour].\n"
                f"EGO:\n{ego_text}\n"
                f"HOP1:\n- " + "\n- ".join(hop1_parts[1:]) + "\nCategory?"
                if len(hop1_parts) > 1 else
                f"Predict the node's category from the provided context.\n"
                f"Possible categories: [rumour, non-rumour].\n"
                f"EGO:\n{ego_text}\nCategory?"
            )

            # Level 2: ego + 2-hop neighbors
            neighbors_2 = set()
            for nb1 in neighbors_1:
                for nb2 in adj.get(nb1, set()):
                    if nb2 != nid and nb2 not in set(neighbors_1):
                        neighbors_2.add(nb2)
            neighbors_2 = list(neighbors_2)[:self.neighbor_sample_size]
            hop2_parts = [ego_text]
            for nb in neighbors_2:
                nb_text = raw_texts[nb] if nb < len(raw_texts) else ""
                if nb_text.strip():
                    hop2_parts.append(nb_text)
            hop2_prompts.append(
                f"Predict the node's category from the provided context.\n"
                f"Possible categories: [rumour, non-rumour].\n"
                f"EGO:\n{ego_text}\n"
                f"HOP2:\n- " + "\n- ".join(hop2_parts[1:]) + "\nCategory?"
                if len(hop2_parts) > 1 else
                f"Predict the node's category from the provided context.\n"
                f"Possible categories: [rumour, non-rumour].\n"
                f"EGO:\n{ego_text}\nCategory?"
            )

        # Call embedding API for each level
        z_L0 = self._embed_batch(ego_prompts, device)    # [K, 3072]
        z_L1 = self._embed_batch(hop1_prompts, device)   # [K, 3072]
        z_L2 = self._embed_batch(hop2_prompts, device)   # [K, 3072]

        z_L = torch.cat([z_L0, z_L1, z_L2], dim=-1)     # [K, 9216]
        return z_L

    def _embed_batch(self, texts, device):
        """Call the embedding API in batches and return tensor."""
        all_embeddings = []
        batch_size = config.LLM_API_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Check cache
            uncached_indices = []
            uncached_texts = []
            cached_results = {}
            for j, text in enumerate(batch_texts):
                cache_key = hash(text)
                if cache_key in self._cache:
                    cached_results[j] = self._cache[cache_key]
                else:
                    uncached_indices.append(j)
                    uncached_texts.append(text)

            # Call API for uncached texts
            if uncached_texts:
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=uncached_texts,
                    )
                    for k, emb_data in enumerate(response.data):
                        orig_idx = uncached_indices[k]
                        emb = emb_data.embedding
                        cached_results[orig_idx] = emb
                        self._cache[hash(batch_texts[orig_idx])] = emb
                except Exception as e:
                    print(f"  [LLM API Warning] Embedding call failed: {e}")
                    # Fallback: zero embeddings for failed calls
                    for k in uncached_indices:
                        if k not in cached_results:
                            cached_results[k] = [0.0] * self.embed_dim

            # Assemble in order
            for j in range(len(batch_texts)):
                all_embeddings.append(cached_results[j])

        tensor = torch.tensor(all_embeddings, dtype=torch.float32, device=device)
        return tensor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Refiner MLP
# ═══════════════════════════════════════════════════════════════════════════════

class RefinerMLP(nn.Module):
    """
    Fuses GNN and LLM embeddings for routed nodes.
    Input: [z_G(v) || z_L(v)] -> refined logits
    """

    def __init__(self, gnn_dim=config.GNN_EMBED_DIM,
                 llm_dim=config.LLM_EMBED_DIM,
                 hidden_dim=config.REFINER_HIDDEN_DIM,
                 num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gnn_dim + llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z_G, z_L):
        """
        Args:
            z_G: [K, gnn_dim] GNN embeddings for routed nodes
            z_L: [K, llm_dim] LLM embeddings for routed nodes
        Returns:
            logits: [K, num_classes]
        """
        combined = torch.cat([z_G, z_L], dim=-1)
        return self.mlp(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Standalone Bi-GAT Classifier (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

class BiGATClassifier(nn.Module):
    """Bi-GAT with graph pooling and MLP head for graph-level rumor detection."""

    def __init__(self, in_dim=config.BERT_FEATURE_DIM,
                 hidden_dim=config.BIGAT_HIDDEN_DIM,
                 heads=config.BIGAT_HEADS,
                 dropout=config.BIGAT_DROPOUT,
                 num_classes=2):
        super().__init__()
        self.bigat = BiGAT(in_dim, hidden_dim, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.bigat.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        z_G = self.bigat(data.x, data.td_edge_index, data.bu_edge_index,
                         data.root_mask, batch)
        # Graph-level pooling
        graph_emb = global_mean_pool(z_G, batch)  # [B, embed_dim]
        logits = self.classifier(graph_emb)  # [B, num_classes]
        return logits, z_G

    def get_node_embeddings(self, data):
        """Get node-level embeddings without classification."""
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        z_G = self.bigat(data.x, data.td_edge_index, data.bu_edge_index,
                         data.root_mask, batch)
        return z_G


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Full GLANCE Model
# ═══════════════════════════════════════════════════════════════════════════════

class GLANCE(nn.Module):
    """
    GLANCE: GNN with LLM Assistance for Neighbor- and Context-aware Embeddings.
    Adapted for graph-level rumor detection with Bi-GAT backbone.

    Training: Only router and refiner are trained. GNN and LLM are frozen.
    """

    def __init__(self, bigat_classifier, homophily_estimator, llm_encoder=None):
        super().__init__()
        # Frozen components
        self.bigat = bigat_classifier.bigat
        self.gnn_head = bigat_classifier.classifier
        for param in self.bigat.parameters():
            param.requires_grad = False
        for param in self.gnn_head.parameters():
            param.requires_grad = False

        self.homophily_est = homophily_estimator
        for param in self.homophily_est.parameters():
            param.requires_grad = False

        # LLM encoder (frozen BERT)
        self.llm_encoder = llm_encoder

        # Trainable components
        self.router = NodeRouter(input_dim=config.ROUTER_INPUT_DIM)
        self.refiner = RefinerMLP(
            gnn_dim=config.GNN_EMBED_DIM,
            llm_dim=config.LLM_EMBED_DIM,
            hidden_dim=config.REFINER_HIDDEN_DIM,
            num_classes=2,
        )

        # GNN prediction head for graph-level (used for non-routed path)
        self.graph_classifier = nn.Sequential(
            nn.Linear(config.GNN_EMBED_DIM, config.BIGAT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.BIGAT_DROPOUT),
            nn.Linear(config.BIGAT_HIDDEN_DIM, 2),
        )

        # Refined graph classifier (after fusion)
        self.refined_graph_classifier = nn.Sequential(
            nn.Linear(config.GNN_EMBED_DIM, config.BIGAT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.BIGAT_DROPOUT),
            nn.Linear(config.BIGAT_HIDDEN_DIM, 2),
        )

    def forward(self, data, k_budget, training=True, raw_texts=None):
        """
        Forward pass with routing.

        Args:
            data: PyG batch
            k_budget: int, number of nodes to route per graph
            training: bool
            raw_texts: list[str] of length N, raw tweet texts for LLM encoder
        Returns:
            dict with keys:
                - logits_gnn: [B, 2] GNN-only predictions
                - logits_refined: [B, 2] refined predictions (for graphs with routed nodes)
                - router_scores: [N] routing probabilities
                - routed_mask: [N] bool mask of routed nodes
        """
        device = data.x.device
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

        # Step 1: Get GNN embeddings (frozen)
        with torch.no_grad():
            z_G = self.bigat(data.x, data.td_edge_index, data.bu_edge_index,
                             data.root_mask, batch)

        # Step 1b: Compute routing features
        routing_features = self._build_routing_features(
            z_G, data.x, data.edge_index, data.degree, data, batch, device)

        # Router scores
        router_scores = self.router(routing_features)  # [N]

        # Top-k routing per graph
        routed_mask = self._topk_routing(router_scores, batch, k_budget)

        # Step 2: Get LLM embeddings for routed nodes
        if routed_mask.any() and self.llm_encoder is not None and raw_texts is not None:
            z_L = self.llm_encoder(routed_mask, data.x, data.edge_index,
                                   raw_texts, device)
        else:
            z_L = torch.zeros(routed_mask.sum().item(), config.LLM_EMBED_DIM, device=device)

        # Step 3: Compute predictions
        # GNN-only path (graph-level)
        graph_emb_gnn = global_mean_pool(z_G, batch)  # [B, GNN_EMBED_DIM]
        logits_gnn = self.graph_classifier(graph_emb_gnn)  # [B, 2]

        # Refined path: replace routed node embeddings
        num_routed = routed_mask.sum().item()
        if num_routed > 0:
            # Get refined node-level logits for routed nodes
            z_G_routed = z_G[routed_mask]
            node_refined_logits = self.refiner(z_G_routed, z_L)  # [K, 2]

            # Create refined node embeddings by using refiner's hidden representation
            # For graph-level: mix refined info back into graph embedding
            z_G_refined = z_G.clone()
            # Use softmax of refined logits as a soft signal to update embeddings
            refined_weights = F.softmax(node_refined_logits, dim=-1)  # [K, 2]

            # Project refined logits back to embedding space for pooling
            refined_signal = torch.zeros_like(z_G)
            refined_signal[routed_mask] = torch.cat([
                refined_weights,
                torch.zeros(num_routed,
                            config.GNN_EMBED_DIM - 2, device=device)
            ], dim=-1)
            z_G_refined = z_G + refined_signal

            graph_emb_refined = global_mean_pool(z_G_refined, batch)
            logits_refined = self.refined_graph_classifier(graph_emb_refined)
        else:
            logits_refined = logits_gnn

        result = {
            "logits_gnn": logits_gnn,
            "logits_refined": logits_refined,
            "router_scores": router_scores,
            "routed_mask": routed_mask,
        }

        return result

    def _build_routing_features(self, z_G, x, edge_index, degree, data, batch, device):
        """
        Build routing feature vector for each node.
        f_v = [z_G(v), uncertainty, homophily_estimate, degree, x_v]
        """
        N = z_G.size(0)

        # 1. GNN embeddings: [N, GNN_EMBED_DIM]
        # Already have z_G

        # 2. Uncertainty via MC dropout
        uncertainty = self._compute_uncertainty(x, data, batch, device)  # [N, 1]

        # 3. Estimated homophily
        with torch.no_grad():
            proba = self.homophily_est.predict_proba(x)  # [N, C]
        homophily = compute_soft_homophily(proba, edge_index, N)  # [N]
        homophily = homophily.unsqueeze(-1)  # [N, 1]

        # 4. Degree: [N, 1]
        deg = degree.unsqueeze(-1).to(device)  # [N, 1]

        # 5. Original features: [N, BERT_FEATURE_DIM]
        # x is already [N, 768]

        routing_features = torch.cat([z_G.detach(), uncertainty, homophily, deg, x], dim=-1)
        return routing_features

    def _compute_uncertainty(self, x, data, batch, device):
        """Estimate uncertainty via MC dropout (multiple forward passes)."""
        self.bigat.train()  # Enable dropout
        predictions = []

        for _ in range(config.UNCERTAINTY_FORWARD_PASSES):
            with torch.no_grad():
                z = self.bigat(data.x, data.td_edge_index, data.bu_edge_index,
                               data.root_mask, batch)
                # Simple uncertainty: variance of embedding norms
                predictions.append(z.norm(dim=-1, keepdim=True))

        self.bigat.eval()

        preds = torch.stack(predictions, dim=0)  # [T, N, 1]
        uncertainty = preds.var(dim=0)  # [N, 1]
        return uncertainty

    def _topk_routing(self, scores, batch, k):
        """Select top-k nodes per graph for routing."""
        N = scores.size(0)
        routed = torch.zeros(N, dtype=torch.bool, device=scores.device)

        if batch is None:
            # Single graph
            k_actual = min(k, N)
            if k_actual > 0:
                _, top_idx = scores.topk(k_actual)
                routed[top_idx] = True
        else:
            num_graphs = batch.max().item() + 1
            for g in range(num_graphs):
                mask = batch == g
                graph_scores = scores[mask]
                k_actual = min(k, graph_scores.size(0))
                if k_actual > 0:
                    _, top_idx = graph_scores.topk(k_actual)
                    graph_indices = mask.nonzero(as_tuple=True)[0]
                    routed[graph_indices[top_idx]] = True

        return routed

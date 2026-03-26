"""
Configuration for GLANCE + Bi-GAT Rumor Detection on PHEME Dataset.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "germanwings-crash-all-rnr-threads")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ─── Dataset ─────────────────────────────────────────────────────────────────
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
RANDOM_SEED = 42
SKIP_UNCLEAR = True  # skip threads annotated as "unclear"

# ─── BERT (Node Feature Encoder) ─────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_FEATURE_DIM = 768
BERT_MAX_LENGTH = 128  # max token length per tweet
NEIGHBOR_SAMPLE_SIZE = 5  # max neighbors per hop for LLM prompts

# ─── LLM API (for GLANCE LLM Encoder) ───────────────────────────────────────
LLM_API_BASE_URL = "https://api.vectorengine.ai/v1"
LLM_API_KEY = "YOUR_API_KEY_HERE"
LLM_EMBEDDING_MODEL = "text-embedding-3-large"  # 3072-dim embeddings
LLM_API_EMBED_DIM = 3072  # dimension of API embedding model
LLM_API_BATCH_SIZE = 64  # max texts per API call
LLM_API_MAX_TOKENS = 512  # max tokens per text input

# ─── Bi-GAT Backbone ────────────────────────────────────────────────────────
BIGAT_HIDDEN_DIM = 128
BIGAT_HEADS = 4
BIGAT_NUM_LAYERS = 2
BIGAT_DROPOUT = 0.3
# Output dim per direction = BIGAT_HIDDEN_DIM (after averaging heads in layer 2)
# Total GNN embedding dim = 2 * BIGAT_HIDDEN_DIM (TD + BU concatenated)
GNN_EMBED_DIM = 2 * BIGAT_HIDDEN_DIM  # 256

# ─── Homophily Estimator (MLP Q) ────────────────────────────────────────────
HOMOPHILY_HIDDEN_DIM = 128
HOMOPHILY_NUM_CLASSES = 2  # pseudo-labels for rumor detection context

# ─── Router ──────────────────────────────────────────────────────────────────
# Routing features: [z_G(v), uncertainty, homophily_estimate, degree, x_v]
# Dimension: GNN_EMBED_DIM + 1 + 1 + 1 + BERT_FEATURE_DIM
ROUTER_INPUT_DIM = GNN_EMBED_DIM + 1 + 1 + 1 + BERT_FEATURE_DIM  # 256+1+1+1+768 = 1027

# ─── LLM Encoder (multi-hop API embeddings) ──────────────────────────────────
# ego + 1-hop + 2-hop, each from the API embedding model
LLM_EMBED_DIM = 3 * LLM_API_EMBED_DIM  # 3 * 3072 = 9216

# ─── Refiner MLP ────────────────────────────────────────────────────────────
REFINER_INPUT_DIM = GNN_EMBED_DIM + LLM_EMBED_DIM  # 256 + 2304 = 2560
REFINER_HIDDEN_DIM = 256

# ─── GLANCE Training ────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
EPOCHS_PHASE1 = 150  # Bi-GAT pre-training
EPOCHS_PHASE2 = 100  # Homophily MLP pre-training
EPOCHS_PHASE3 = 150  # GLANCE router + refiner training
PATIENCE = 20  # early stopping patience

# Router & routing budget
BETA = 0.2  # LLM query cost penalty
LAMBDA_ROUTER = 1.0  # weight for router loss
LAMBDA_ENT = 0.01  # entropy regularization weight
K_START_RATIO = 1.0  # K_start = K_START_RATIO * avg_nodes_per_graph
K_END_RATIO = 0.25  # K_end = K_END_RATIO * K_start
K_DECAY_RATE = 0.5  # exponential decay factor

# Dropout for uncertainty estimation
UNCERTAINTY_DROPOUT = 0.3
UNCERTAINTY_FORWARD_PASSES = 5

# ─── Device ──────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

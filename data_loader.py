"""
Data loader for PHEME Germanwings Crash dataset.
Parses conversation threads into PyTorch Geometric graph objects with BERT embeddings.
"""
import os
import json
import pickle
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from collections import defaultdict

import config


def _parse_structure(structure_dict, parent=None, edges_td=None, edges_bu=None):
    """Recursively parse structure.json into top-down and bottom-up edge lists."""
    if edges_td is None:
        edges_td = []
    if edges_bu is None:
        edges_bu = []

    for node_id, children in structure_dict.items():
        node_id = str(node_id)
        if parent is not None:
            edges_td.append((str(parent), node_id))  # parent -> child
            edges_bu.append((node_id, str(parent)))  # child -> parent
        if isinstance(children, dict):
            _parse_structure(children, parent=node_id,
                             edges_td=edges_td, edges_bu=edges_bu)
        elif isinstance(children, list):
            pass  # leaf node
    return edges_td, edges_bu


def _collect_node_ids_from_structure(structure_dict, node_ids=None):
    """Collect all node IDs from the structure."""
    if node_ids is None:
        node_ids = []
    for node_id, children in structure_dict.items():
        node_ids.append(str(node_id))
        if isinstance(children, dict):
            _collect_node_ids_from_structure(children, node_ids)
    return node_ids


def _load_tweet_text(filepath):
    """Load tweet text from a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tweet = json.load(f)
        return tweet.get("text", "")
    except (json.JSONDecodeError, FileNotFoundError):
        return ""


def _get_neighbors(edge_list, node_id, hop=1):
    """Get neighbors up to `hop` hops from node_id given a list of (src, dst) edges."""
    adj = defaultdict(set)
    for src, dst in edge_list:
        adj[src].add(dst)

    current = {node_id}
    visited = {node_id}
    neighbors_by_hop = {}

    for h in range(1, hop + 1):
        next_level = set()
        for n in current:
            for nb in adj.get(n, []):
                if nb not in visited:
                    next_level.add(nb)
                    visited.add(nb)
        neighbors_by_hop[h] = next_level
        current = next_level

    return neighbors_by_hop


def load_pheme_dataset(dataset_dir=None):
    """
    Load PHEME dataset and return list of raw thread data dicts.
    Each dict: {thread_id, label, structure, node_ids, texts, root_id,
                edges_td, edges_bu}
    """
    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    threads = []

    for label_name, label_val in [("rumours", 1), ("non-rumours", 0)]:
        label_dir = os.path.join(dataset_dir, label_name)
        if not os.path.exists(label_dir):
            continue

        for thread_id in os.listdir(label_dir):
            thread_path = os.path.join(label_dir, thread_id)
            if not os.path.isdir(thread_path):
                continue

            # Read annotation
            ann_path = os.path.join(thread_path, "annotation.json")
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                is_rumour = ann.get("is_rumour", "")
                if config.SKIP_UNCLEAR and is_rumour == "unclear":
                    continue
            else:
                continue

            # Read structure
            struct_path = os.path.join(thread_path, "structure.json")
            if not os.path.exists(struct_path):
                continue
            with open(struct_path, "r", encoding="utf-8") as f:
                structure = json.load(f)

            # Parse edges
            edges_td, edges_bu = _parse_structure(structure)
            node_ids = _collect_node_ids_from_structure(structure)

            if len(node_ids) == 0:
                continue

            root_id = node_ids[0]

            # Load texts for all nodes
            texts = {}
            # Source tweet
            src_dir = os.path.join(thread_path, "source-tweets")
            if os.path.exists(src_dir):
                for fname in os.listdir(src_dir):
                    if fname.endswith(".json"):
                        nid = fname.replace(".json", "")
                        texts[nid] = _load_tweet_text(
                            os.path.join(src_dir, fname))

            # Reactions
            react_dir = os.path.join(thread_path, "reactions")
            if os.path.exists(react_dir):
                for fname in os.listdir(react_dir):
                    if fname.endswith(".json"):
                        nid = fname.replace(".json", "")
                        texts[nid] = _load_tweet_text(
                            os.path.join(react_dir, fname))

            threads.append({
                "thread_id": thread_id,
                "label": label_val,
                "node_ids": node_ids,
                "root_id": root_id,
                "texts": texts,
                "edges_td": edges_td,
                "edges_bu": edges_bu,
            })

    return threads


def encode_texts_bert(texts_list, tokenizer, model, device, max_length=128):
    """Encode a list of texts into BERT [CLS] embeddings. Returns [N, 768] tensor."""
    model.eval()
    embeddings = []

    batch_size = 64
    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i:i + batch_size]
        # Replace empty strings
        batch_texts = [t if t.strip() else "[UNK]" for t in batch_texts]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_emb.cpu())

    return torch.cat(embeddings, dim=0)


def build_pyg_graphs(threads, tokenizer, model, device):
    """Convert raw thread data into PyG Data objects with BERT features."""
    graphs = []

    for thread in threads:
        node_ids = thread["node_ids"]
        n = len(node_ids)
        if n == 0:
            continue

        # Build node ID -> index mapping
        id2idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Get texts in order
        texts_ordered = []
        for nid in node_ids:
            texts_ordered.append(thread["texts"].get(nid, ""))

        # BERT embeddings
        x = encode_texts_bert(texts_ordered, tokenizer, model, device,
                              max_length=config.BERT_MAX_LENGTH)

        # Build edge indices
        td_src, td_dst = [], []
        for src, dst in thread["edges_td"]:
            if src in id2idx and dst in id2idx:
                td_src.append(id2idx[src])
                td_dst.append(id2idx[dst])

        bu_src, bu_dst = [], []
        for src, dst in thread["edges_bu"]:
            if src in id2idx and dst in id2idx:
                bu_src.append(id2idx[src])
                bu_dst.append(id2idx[dst])

        # Bidirectional edge_index (union of TD and BU)
        all_src = td_src + bu_src
        all_dst = td_dst + bu_dst

        if len(all_src) == 0:
            # Single-node graph: add self-loop
            all_src = [0]
            all_dst = [0]
            td_src = [0]
            td_dst = [0]
            bu_src = [0]
            bu_dst = [0]

        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        td_edge_index = torch.tensor([td_src, td_dst], dtype=torch.long)
        bu_edge_index = torch.tensor([bu_src, bu_dst], dtype=torch.long)

        # Root mask
        root_idx = id2idx.get(thread["root_id"], 0)
        root_mask = torch.zeros(n, dtype=torch.bool)
        root_mask[root_idx] = True

        # Degree per node (in bidirectional graph)
        degree = torch.zeros(n, dtype=torch.float)
        for idx in all_dst:
            degree[idx] += 1

        data = Data(
            x=x,
            edge_index=edge_index,
            td_edge_index=td_edge_index,
            bu_edge_index=bu_edge_index,
            y=torch.tensor(thread["label"], dtype=torch.long),
            root_mask=root_mask,
            degree=degree,
            num_nodes=n,
        )

        # Store raw texts as a tensor of indices into a global text store
        # We'll store texts separately and attach the thread_id for lookup
        data.thread_id = thread["thread_id"]

        graphs.append(data)

    return graphs


# ─── Global text store ──────────────────────────────────────────────────────
# Maps thread_id -> list of raw text strings (one per node, in order)
_TEXT_STORE = {}


def build_text_store(threads):
    """Build a global lookup from thread_id -> ordered node texts."""
    global _TEXT_STORE
    _TEXT_STORE = {}
    for thread in threads:
        node_ids = thread["node_ids"]
        texts_ordered = [thread["texts"].get(nid, "") for nid in node_ids]
        _TEXT_STORE[thread["thread_id"]] = texts_ordered


def get_raw_texts_for_batch(batch_data):
    """
    Retrieve raw texts for a batched PyG Data object.
    Returns a flat list of strings of length N (total nodes in batch).
    """
    if not hasattr(batch_data, "thread_id"):
        return [""] * batch_data.x.size(0)

    thread_ids = batch_data.thread_id
    if isinstance(thread_ids, str):
        thread_ids = [thread_ids]

    texts = []
    for tid in thread_ids:
        texts.extend(_TEXT_STORE.get(tid, []))

    # Pad/truncate to match actual node count
    N = batch_data.x.size(0)
    if len(texts) < N:
        texts.extend([""] * (N - len(texts)))
    elif len(texts) > N:
        texts = texts[:N]

    return texts


def get_dataloaders(graphs=None, batch_size=None):
    """
    Split graphs into train/val/test and return DataLoaders.
    If graphs is None, loads and processes the full dataset.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if graphs is None:
        graphs = load_and_cache_dataset()

    labels = [g.y.item() for g in graphs]

    # Stratified split
    train_idx, temp_idx = train_test_split(
        range(len(graphs)),
        test_size=1.0 - config.SPLIT_RATIOS[0],
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = config.SPLIT_RATIOS[1] / (config.SPLIT_RATIOS[1] + config.SPLIT_RATIOS[2])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_ratio,
        stratify=temp_labels,
        random_state=config.RANDOM_SEED,
    )

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: train={len(train_graphs)}, val={len(val_graphs)}, "
          f"test={len(test_graphs)}")
    print(f"  Train: {sum(1 for g in train_graphs if g.y.item()==1)} rumour, "
          f"{sum(1 for g in train_graphs if g.y.item()==0)} non-rumour")

    return train_loader, val_loader, test_loader


def load_and_cache_dataset():
    """Load dataset with caching of BERT embeddings. Also builds the text store."""
    cache_path = os.path.join(config.CACHE_DIR, "pheme_graphs.pkl")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Always load threads for the text store
    threads = load_pheme_dataset()
    build_text_store(threads)
    print(f"Built text store for {len(threads)} threads")

    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs from cache")
        return graphs

    print("Processing PHEME dataset (first time, will cache)...")
    print(f"Found {len(threads)} threads "
          f"({sum(1 for t in threads if t['label']==1)} rumour, "
          f"{sum(1 for t in threads if t['label']==0)} non-rumour)")

    # Load BERT
    print(f"Loading BERT model: {config.BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME)
    bert_model = bert_model.to(config.DEVICE)
    bert_model.eval()

    print("Encoding tweets with BERT...")
    graphs = build_pyg_graphs(threads, tokenizer, bert_model, config.DEVICE)
    print(f"Built {len(graphs)} graphs")

    # Cache
    with open(cache_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Cached dataset to {cache_path}")

    # Free BERT from GPU
    del bert_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return graphs


if __name__ == "__main__":
    graphs = load_and_cache_dataset()
    print(f"\nTotal graphs: {len(graphs)}")

    # Print some statistics
    node_counts = [g.num_nodes for g in graphs]
    edge_counts = [g.edge_index.size(1) for g in graphs]
    print(f"Nodes per graph: min={min(node_counts)}, max={max(node_counts)}, "
          f"mean={np.mean(node_counts):.1f}")
    print(f"Edges per graph: min={min(edge_counts)}, max={max(edge_counts)}, "
          f"mean={np.mean(edge_counts):.1f}")

    # Test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(graphs)
    for batch in train_loader:
        print(f"\nSample batch: {batch}")
        print(f"  x shape: {batch.x.shape}")
        print(f"  edge_index shape: {batch.edge_index.shape}")
        print(f"  y: {batch.y}")
        break

"""
Training pipeline for GLANCE + Bi-GAT Rumor Detection.

Phase 1: Pre-train Bi-GAT backbone on graph classification
Phase 2: Pre-train Homophily Estimator MLP Q
Phase 3: Train GLANCE (router + refiner) with advantage-based objective
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import numpy as np

import config
from models import (BiGATClassifier, HomophilyEstimator, LLMEncoder,
                    GLANCE, compute_soft_homophily)
from data_loader import get_raw_texts_for_batch
from evaluate import evaluate_model, evaluate_glance


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Pre-train Bi-GAT
# ═══════════════════════════════════════════════════════════════════════════════

def train_bigat(train_loader, val_loader, device=config.DEVICE):
    """Pre-train Bi-GAT classifier for graph-level rumor detection."""
    print("\n" + "=" * 70)
    print("PHASE 1: Pre-training Bi-GAT Backbone")
    print("=" * 70)

    model = BiGATClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.EPOCHS_PHASE1 + 1):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(batch)
            loss = criterion(logits, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * batch.y.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} "
                  f"F1: {val_metrics['f1_macro']:.4f}")

        # Early stopping
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"Best Val F1: {best_val_f1:.4f}")

    # Save checkpoint
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "bigat_phase1.pt")
    torch.save(best_state, ckpt_path)
    print(f"Saved Bi-GAT checkpoint to {ckpt_path}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Pre-train Homophily Estimator
# ═══════════════════════════════════════════════════════════════════════════════

def train_homophily_estimator(train_loader, val_loader, device=config.DEVICE):
    """
    Pre-train MLP Q to predict pseudo-labels from node features.
    Uses graph labels as supervision signal for all nodes in each graph.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Pre-training Homophily Estimator (MLP Q)")
    print("=" * 70)

    model = HomophilyEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.EPOCHS_PHASE2 + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Predict pseudo-labels for each node
            logits = model(batch.x)  # [N, 2]

            # Node-level labels: propagate graph label to all nodes
            if batch.batch is not None:
                node_labels = batch.y[batch.batch]  # [N]
            else:
                node_labels = batch.y.expand(batch.x.size(0))

            loss = criterion(logits, node_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * logits.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == node_labels).sum().item()
            total += logits.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x)
                if batch.batch is not None:
                    node_labels = batch.y[batch.batch]
                else:
                    node_labels = batch.y.expand(batch.x.size(0))
                pred = logits.argmax(dim=-1)
                val_correct += (pred == node_labels).sum().item()
                val_total += logits.size(0)

        val_acc = val_correct / val_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"Best Val Acc: {best_val_acc:.4f}")

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "homophily_phase2.pt")
    torch.save(best_state, ckpt_path)
    print(f"Saved Homophily Estimator checkpoint to {ckpt_path}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Train GLANCE (Router + Refiner)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_k_budget(epoch, total_epochs):
    """Exponential K-decay schedule for routing budget."""
    # Average nodes per graph is ~10 for PHEME Germanwings
    k_start = 12  # route most nodes initially
    k_end = 3     # route fewer nodes at end
    r = config.K_DECAY_RATE

    k = round(k_end + (k_start - k_end) * (r ** (epoch - 1)))
    return max(k, 1)


def train_glance(train_loader, val_loader, bigat_model, homophily_model,
                 device=config.DEVICE):
    """
    Train GLANCE router and refiner with advantage-based objective.
    GNN (Bi-GAT) and LLM (BERT) are frozen.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Training GLANCE (Router + Refiner)")
    print("=" * 70)

    # Initialize LLM encoder
    llm_encoder = LLMEncoder().to(device)

    # Build GLANCE model
    glance = GLANCE(bigat_model, homophily_model, llm_encoder).to(device)

    # Only optimize trainable parameters (router + refiner + graph classifiers)
    trainable_params = [p for p in glance.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.EPOCHS_PHASE3 + 1):
        glance.train()
        # But keep frozen parts in eval mode
        glance.bigat.eval()
        glance.homophily_est.eval()

        k_budget = _compute_k_budget(epoch, config.EPOCHS_PHASE3)

        total_loss = 0
        total_pred_loss = 0
        total_router_loss = 0
        correct = 0
        total = 0
        total_routed = 0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            raw_texts = get_raw_texts_for_batch(batch)
            optimizer.zero_grad()

            # Forward pass
            result = glance(batch, k_budget, training=True, raw_texts=raw_texts)

            logits_gnn = result["logits_gnn"]         # [B, 2]
            logits_refined = result["logits_refined"]  # [B, 2]
            router_scores = result["router_scores"]    # [N]
            routed_mask = result["routed_mask"]         # [N]

            B = batch.y.size(0)
            N = router_scores.size(0)

            # ── Prediction Loss ──────────────────────────────────────────
            # Use refined logits as the primary prediction
            loss_pred = F.cross_entropy(logits_refined, batch.y)

            # ── Router Loss (Advantage-based) ────────────────────────────
            # Compute per-graph advantage
            loss_gnn_per_graph = criterion(logits_gnn, batch.y)      # [B]
            loss_llm_per_graph = criterion(logits_refined, batch.y)  # [B]

            # Advantage: benefit of routing
            # Positive reward = LLM path is better than GNN path
            advantage = loss_gnn_per_graph - loss_llm_per_graph - config.BETA  # [B]

            # Broadcast advantage to node level
            if batch.batch is not None:
                # For routed nodes: advantage from their graph
                node_advantage = torch.zeros(N, device=device)
                for g in range(B):
                    graph_mask = batch.batch == g
                    routed_in_graph = routed_mask & graph_mask
                    not_routed_in_graph = (~routed_mask) & graph_mask

                    if routed_in_graph.any():
                        node_advantage[routed_in_graph] = advantage[g]
                    if not_routed_in_graph.any():
                        node_advantage[not_routed_in_graph] = -loss_gnn_per_graph[g]
            else:
                node_advantage = torch.zeros(N, device=device)
                node_advantage[routed_mask] = advantage[0] if B > 0 else 0
                node_advantage[~routed_mask] = -loss_gnn_per_graph[0] if B > 0 else 0

            # Policy gradient loss
            log_probs = torch.log(router_scores.clamp(min=1e-8))
            log_1_minus = torch.log((1 - router_scores).clamp(min=1e-8))

            # For routed nodes: reinforce routing decision
            # For non-routed nodes: reinforce not-routing decision
            policy_log_prob = torch.where(routed_mask, log_probs, log_1_minus)

            # Entropy bonus
            entropy = -(router_scores * log_probs +
                        (1 - router_scores) * log_1_minus)

            loss_router = -(node_advantage.detach() * policy_log_prob -
                            config.LAMBDA_ENT * entropy).mean()

            # ── Combined Loss ────────────────────────────────────────────
            loss = loss_pred + config.LAMBDA_ROUTER * loss_router

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.GRAD_CLIP)
            optimizer.step()

            # Track metrics
            total_loss += loss.item() * B
            total_pred_loss += loss_pred.item() * B
            total_router_loss += loss_router.item() * B
            pred = logits_refined.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            total += B
            total_routed += routed_mask.sum().item()
            total_nodes += N

        train_loss = total_loss / total
        train_acc = correct / total
        route_pct = total_routed / max(total_nodes, 1) * 100

        # Validate
        val_metrics = evaluate_glance(glance, val_loader, k_budget, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | K={k_budget} "
                  f"Route: {route_pct:.1f}% | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} "
                  f"F1: {val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in glance.state_dict().items()
                          if "llm_encoder" not in k}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best state (excluding LLM encoder weights which are frozen)
    current_state = glance.state_dict()
    current_state.update(best_state)
    glance.load_state_dict(current_state)
    print(f"Best Val F1: {best_val_f1:.4f}")

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "glance_phase3.pt")
    torch.save(best_state, ckpt_path)
    print(f"Saved GLANCE checkpoint to {ckpt_path}")

    return glance

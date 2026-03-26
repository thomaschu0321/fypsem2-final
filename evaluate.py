"""
Evaluation utilities for GLANCE + Bi-GAT Rumor Detection.
Computes Accuracy, Precision, Recall, F1 (per-class and macro), AUC-ROC.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

import config
from data_loader import get_raw_texts_for_batch


def evaluate_model(model, data_loader, device=config.DEVICE):
    """
    Evaluate a BiGATClassifier model.
    Returns dict of metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    return _compute_metrics(all_labels, all_preds, all_probs)


def evaluate_glance(glance_model, data_loader, k_budget, device=config.DEVICE):
    """
    Evaluate the full GLANCE model.
    Returns dict of metrics + routing statistics.
    """
    glance_model.eval()
    glance_model.bigat.eval()
    glance_model.homophily_est.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_routed = 0
    total_nodes = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            raw_texts = get_raw_texts_for_batch(batch)
            result = glance_model(batch, k_budget, training=False,
                                  raw_texts=raw_texts)

            logits = result["logits_refined"]
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

            total_routed += result["routed_mask"].sum().item()
            total_nodes += result["router_scores"].size(0)

    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    metrics["route_percentage"] = total_routed / max(total_nodes, 1) * 100
    metrics["total_routed"] = total_routed
    metrics["total_nodes"] = total_nodes

    return metrics


def _compute_metrics(labels, preds, probs):
    """Compute classification metrics."""
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_rumour": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_rumour": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_rumour": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_nonrumour": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_nonrumour": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_nonrumour": f1_score(labels, preds, pos_label=0, zero_division=0),
    }

    # AUC-ROC (only if both classes present)
    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    else:
        metrics["auc_roc"] = 0.0

    return metrics


def print_evaluation_report(metrics, title="Evaluation Results"):
    """Print a formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  {'─' * 56}")
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'─' * 56}")
    print(f"  {'Rumour':<15} {metrics['precision_rumour']:<12.4f} "
          f"{metrics['recall_rumour']:<12.4f} {metrics['f1_rumour']:<12.4f}")
    print(f"  {'Non-Rumour':<15} {metrics['precision_nonrumour']:<12.4f} "
          f"{metrics['recall_nonrumour']:<12.4f} {metrics['f1_nonrumour']:<12.4f}")
    print(f"  {'─' * 56}")
    print(f"  {'Macro Avg':<15} {metrics['precision_macro']:<12.4f} "
          f"{metrics['recall_macro']:<12.4f} {metrics['f1_macro']:<12.4f}")

    if "route_percentage" in metrics:
        print(f"  {'─' * 56}")
        print(f"  Routing: {metrics['total_routed']}/{metrics['total_nodes']} "
              f"nodes ({metrics['route_percentage']:.1f}%)")

    print(f"{'=' * 60}\n")


def compare_models(bigat_metrics, glance_metrics):
    """Print comparison between Bi-GAT baseline and GLANCE."""
    print(f"\n{'=' * 70}")
    print(f"  Model Comparison: Bi-GAT Baseline vs GLANCE + Bi-GAT")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<20} {'Bi-GAT':<15} {'GLANCE':<15} {'Delta':<15}")
    print(f"  {'─' * 66}")

    for key, label in [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 (Macro)"),
        ("f1_rumour", "F1 (Rumour)"),
        ("f1_nonrumour", "F1 (Non-Rumour)"),
        ("auc_roc", "AUC-ROC"),
        ("precision_rumour", "Prec (Rumour)"),
        ("recall_rumour", "Rec (Rumour)"),
    ]:
        v1 = bigat_metrics.get(key, 0)
        v2 = glance_metrics.get(key, 0)
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {v1:<15.4f} {v2:<15.4f} {sign}{delta:<14.4f}")

    print(f"{'=' * 70}\n")

"""
GLANCE + Bi-GAT Rumor Detection on PHEME Dataset
Main entry point for training and evaluation.

Usage:
    python run.py                    # Run full pipeline (Phase 1 + 2 + 3 + eval)
    python run.py --phase 1          # Only Phase 1: pre-train Bi-GAT
    python run.py --phase 2          # Only Phase 2: pre-train Homophily MLP
    python run.py --phase 3          # Only Phase 3: train GLANCE
    python run.py --eval-only        # Evaluate from checkpoints
    python run.py --skip-phase1      # Skip Phase 1 (load from checkpoint)
    python run.py --skip-phase2      # Skip Phase 2 (load from checkpoint)
"""
import argparse
import os
import torch
import numpy as np
import random

import config
from data_loader import load_and_cache_dataset, get_dataloaders
from models import BiGATClassifier, HomophilyEstimator, LLMEncoder, GLANCE
from train import train_bigat, train_homophily_estimator, train_glance
from evaluate import (evaluate_model, evaluate_glance,
                      print_evaluation_report, compare_models)


def set_seed(seed=config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bigat_from_checkpoint(device):
    """Load pre-trained Bi-GAT from checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "bigat_phase1.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Bi-GAT checkpoint not found: {ckpt_path}")
    model = BiGATClassifier().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded Bi-GAT from {ckpt_path}")
    return model


def load_homophily_from_checkpoint(device):
    """Load pre-trained Homophily Estimator from checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "homophily_phase2.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Homophily checkpoint not found: {ckpt_path}")
    model = HomophilyEstimator().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded Homophily Estimator from {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="GLANCE + Bi-GAT Rumor Detection on PHEME Dataset")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run only a specific phase")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate from checkpoints only")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1, load Bi-GAT from checkpoint")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip Phase 2, load Homophily MLP from checkpoint")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    device = args.device
    set_seed()

    print("=" * 70)
    print("  GLANCE + Bi-GAT Rumor Detection on PHEME (Germanwings Crash)")
    print(f"  Device: {device}")
    print("=" * 70)

    # ── Load Dataset ─────────────────────────────────────────────────────
    print("\nLoading dataset...")
    graphs = load_and_cache_dataset()
    train_loader, val_loader, test_loader = get_dataloaders(graphs)

    # ── Evaluation Only Mode ─────────────────────────────────────────────
    if args.eval_only:
        print("\nEvaluation-only mode")
        bigat_model = load_bigat_from_checkpoint(device)
        bigat_metrics = evaluate_model(bigat_model, test_loader, device)
        print_evaluation_report(bigat_metrics, "Bi-GAT Baseline (Test Set)")

        # Try loading GLANCE
        glance_ckpt = os.path.join(config.CHECKPOINT_DIR, "glance_phase3.pt")
        if os.path.exists(glance_ckpt):
            homophily_model = load_homophily_from_checkpoint(device)
            llm_encoder = LLMEncoder().to(device)
            glance = GLANCE(bigat_model, homophily_model, llm_encoder).to(device)
            state = torch.load(glance_ckpt, map_location=device, weights_only=True)
            current = glance.state_dict()
            current.update(state)
            glance.load_state_dict(current)

            glance_metrics = evaluate_glance(glance, test_loader, k_budget=3,
                                             device=device)
            print_evaluation_report(glance_metrics, "GLANCE + Bi-GAT (Test Set)")
            compare_models(bigat_metrics, glance_metrics)
        return

    # ── Phase 1: Pre-train Bi-GAT ────────────────────────────────────────
    if args.phase is None or args.phase == 1:
        if args.skip_phase1:
            bigat_model = load_bigat_from_checkpoint(device)
        else:
            bigat_model = train_bigat(train_loader, val_loader, device)

        # Evaluate Phase 1
        test_metrics = evaluate_model(bigat_model, test_loader, device)
        print_evaluation_report(test_metrics, "Phase 1: Bi-GAT Baseline (Test Set)")

        if args.phase == 1:
            return
    else:
        bigat_model = load_bigat_from_checkpoint(device)

    # ── Phase 2: Pre-train Homophily Estimator ───────────────────────────
    if args.phase is None or args.phase == 2:
        if args.skip_phase2:
            homophily_model = load_homophily_from_checkpoint(device)
        else:
            homophily_model = train_homophily_estimator(
                train_loader, val_loader, device)

        if args.phase == 2:
            return
    else:
        homophily_model = load_homophily_from_checkpoint(device)

    # ── Phase 3: Train GLANCE ────────────────────────────────────────────
    if args.phase is None or args.phase == 3:
        glance = train_glance(train_loader, val_loader,
                              bigat_model, homophily_model, device)

        # Final evaluation
        print("\n" + "=" * 70)
        print("  FINAL EVALUATION ON TEST SET")
        print("=" * 70)

        bigat_metrics = evaluate_model(bigat_model, test_loader, device)
        print_evaluation_report(bigat_metrics, "Bi-GAT Baseline (Test Set)")

        glance_metrics = evaluate_glance(glance, test_loader, k_budget=3,
                                         device=device)
        print_evaluation_report(glance_metrics, "GLANCE + Bi-GAT (Test Set)")

        compare_models(bigat_metrics, glance_metrics)


if __name__ == "__main__":
    main()

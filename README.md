# GLANCE + Bi-GAT Rumor Detection

A PyTorch implementation of **GLANCE** (Graph-adaptive LLM Augmentation via Neighborhood Context Estimation) combined with **Bi-GAT** (Bidirectional Graph Attention Network) for rumor detection on the PHEME dataset.

## Overview

This project implements a hybrid approach that combines:
- **Bi-GAT**: Bidirectional Graph Attention Network for encoding tweet propagation graphs
- **GLANCE Router**: Learns to dynamically query an LLM for challenging nodes
- **LLM Augmentation**: Uses API-based embeddings from text-embedding-3-large model

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- Transformers >= 4.30.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

## Dataset

Place the PHEME dataset in `germanwings-crash-all-rnr-threads/`.

## Configuration

Edit `config.py` to set:
- `LLM_API_KEY`: Your API key for the LLM service
- `LLM_API_BASE_URL`: Base URL for the embedding API
- Training hyperparameters (batch size, learning rate, epochs)

## Training

### Phase 1: Pre-train Bi-GAT
```bash
python train.py --phase 1
```

### Phase 2: Pre-train Homophily Estimator
```bash
python train.py --phase 2
```

### Phase 3: Train GLANCE Router & Refiner
```bash
python train.py --phase 3
```

## Evaluation

```bash
python evaluate.py
```

## Project Structure

```
.
├── config.py          # Configuration and hyperparameters
├── data_loader.py     # PHEME dataset loader
├── models.py          # Bi-GAT, Router, Refiner, GLANCE model
├── train.py           # Training pipeline
├── evaluate.py        # Evaluation script
├── run.py             # Main entry point
└── germanwings-crash-all-rnr-threads/  # Dataset
```

## Citation

If you use this code, please cite the original GLANCE paper.

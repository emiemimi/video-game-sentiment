---
license: mit
library_name: pytorch
tags:
- text-classification
- embeddings
- sentiment-analysis
- video-games
- reddit
---

# Video Game Sentiment Model

This repository contains PyTorch classifiers trained on Qwen3 0.6B embeddings for gaming-community sentiment classification.

## Task

Input: a gaming-related comment.

Output sentiment:

- `positive`
- `negative`
- `mixed`
- `neutral`

## Training

- Dataset: balanced 10k gaming Reddit comments
- Embedding model: `qwen3-embedding:0.6b`
- Embedding dimension: 1,024
- Split: 70% train, 15% dev, 15% test
- Classifiers:
  - Gaussian Naive Bayes implemented with PyTorch tensors
  - Feed-forward neural network implemented with vanilla PyTorch

## Results

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|
| Gaussian Naive Bayes | 0.5200 | 0.5176 | 0.5179 |
| Neural Network | 0.6000 | 0.6006 | 0.6016 |

The neural network checkpoint is the recommended model.

## Limitations

Labels were created with LLM assistance and only minimal human review. The model may have learned from noisy labels, which likely contributes to modest performance.


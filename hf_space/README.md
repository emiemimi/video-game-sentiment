---
title: Video Game Sentiment Demo
sdk: gradio
python_version: "3.10"
app_file: app.py
---

# Video Game Sentiment Demo

This Space predicts sentiment for video-game community comments.

It runs on free Hugging Face CPU hardware and does not use paid APIs, paid Inference Providers, or paid GPU hardware.

The app loads `Qwen/Qwen3-Embedding-0.6B` with `sentence-transformers`, then applies the trained PyTorch sentiment classifier.

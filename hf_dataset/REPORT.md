# Video Game Sentiment Classifier Report

## Introduction

The goal of this project was to build a classifier that identifies how people feel about video-game-related topics in online community comments. Instead of using formal game reviews only, the project focuses on Reddit-style gaming discussion, where language is informal, opinionated, sarcastic, and often context-dependent.

The final task is sentiment classification. Given a gaming-related comment, the model predicts one of four labels:

- `positive`
- `negative`
- `mixed`
- `neutral`

This is useful because gaming communities produce large amounts of unstructured feedback. A classifier like this could help summarize player reactions to games, publishers, monetization, performance problems, updates, and community issues.

## Dataset and Labeling Process

The source dataset was `gk4u/reddit_dataset_218` from Hugging Face. I filtered it to gaming-related communities, including subreddits such as `gaming`, `videogames`, `pcgaming`, `Steam`, `NintendoSwitch`, `PS5`, `truegaming`, `Games`, and `gamingnews`.

The cleaning process removed:

- duplicate comments
- deleted or removed content
- spam-like comments
- very short comments
- low-quality URL-heavy rows

After filtering, I created a labeled dataset and then a balanced 10,000-row training subset. The final balanced dataset contains:

| Sentiment | Rows |
|---|---:|
| positive | 2,676 |
| negative | 2,545 |
| neutral | 2,500 |
| mixed | 2,279 |

The labels were created with LLM assistance. Codex used a sub-agent named Jason to pre-label the dataset according to a labeling guide. Human review was minimal. This is an important limitation: some labels may be wrong or inconsistent, and this label noise may explain part of the poor or modest model performance.

Only the balanced 10k dataset is intended for Hugging Face dataset upload.

## Embeddings and Models

The project uses embeddings because gaming comments can express the same sentiment with many different words. For example, performance complaints might mention FPS drops, lag, stuttering, poor optimization, or crashes. Embeddings represent comments by semantic meaning rather than exact keywords.

The latest training run used:

- Embedding model: `qwen3-embedding:0.6b`
- Embedding dimension: 1,024
- Dataset: balanced 10k comments
- Split: 70% train, 15% dev, 15% test

Two classifiers were trained using vanilla PyTorch:

1. Gaussian Naive Bayes
2. Feed-forward neural network

The neural network architecture was:

```text
Linear(input_dim, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 4)
```

Training used cross-entropy loss, AdamW, early stopping, and macro F1 on the dev set as the main selection metric.

## Results

The dataset was split into:

| Split | Rows |
|---|---:|
| Train | 7,000 |
| Dev | 1,500 |
| Test | 1,500 |

Final test results:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Gaussian Naive Bayes | 0.5200 | 0.5176 | 0.5179 |
| Neural Network | 0.6000 | 0.6006 | 0.6016 |

The best neural network checkpoint came from epoch 10.

## Analysis

The neural network outperformed Naive Bayes. This is expected because dense embedding dimensions are not independent, while Naive Bayes assumes feature independence. The neural network can learn more flexible decision boundaries over the embedding vectors.

The results are still modest. Several factors likely contributed:

- Reddit comments are short and often require context from the thread.
- Some comments are sarcastic or ambiguous.
- `mixed` and `neutral` are difficult to distinguish.
- The labels were generated with LLM assistance and received only minimal human review.
- Badly labeled examples may have trained the models to learn inconsistent patterns.

The confusion matrix shows that the model often confuses mixed sentiment with positive or negative sentiment. This is reasonable because mixed comments often contain both praise and criticism, and the model only sees the isolated text.

## Use of Codex

Codex was used throughout the project.

First, Codex helped define and refine the project idea. The initial idea was a customer support ticket router, but the project shifted to gaming sentiment because it was more personally relevant and better suited to informal social-media text.

Codex helped find possible online datasets and then filtered the large Reddit dataset from Hugging Face. It created scripts to keep only gaming-related subreddits and remove duplicates, spam, deleted content, and very short comments.

Codex also supported the labeling workflow. A Codex sub-agent named Jason pre-labeled the data using the labeling guide. The labels were then reviewed only minimally by a human, which is a limitation of the final dataset.

For modeling, Codex implemented the PyTorch training script. The script generates embeddings with Ollama, caches them, splits the dataset, trains Gaussian Naive Bayes and a neural network, evaluates both models, and saves metrics and model checkpoints.

Finally, Codex helped package the project for GitHub and Hugging Face by preparing documentation, model artifacts, dataset artifacts, and a Gradio demo.

## Limitations and Future Work

The main limitation is label quality. Minimal human review means the dataset may contain noisy or incorrect labels. A stronger version of this project should manually review a larger sample, measure inter-annotator agreement, and correct low-confidence examples.

The project currently predicts only sentiment. A useful extension would add a second classifier for the reason behind the sentiment, such as gameplay, graphics, performance, bugs, price, company reputation, monetization, multiplayer, or updates.

Another improvement would compare multiple embedding models and tune the neural network architecture. The final demo uses `Qwen/Qwen3-Embedding-0.6B` so that it can run without paid hardware, but larger models may improve performance.

## Conclusion

The project successfully built an end-to-end video-game sentiment classifier using embeddings, PyTorch classifiers, a custom Reddit-based dataset, and a deployable Hugging Face demo. The neural network performed better than the Naive Bayes baseline, but the modest scores show that label quality and comment ambiguity are major challenges.


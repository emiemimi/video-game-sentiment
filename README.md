# Video Game Sentiment Classifier
___
*This project was entirely made with Codex. See the attached report for more details and discussion on this.*
___

This project classifies the sentiment of video-game community comments using text embeddings and PyTorch classifiers.

The final model predicts one of four sentiment labels:

- `positive`
- `negative`
- `mixed`
- `neutral`

## Project Structure

- `data/reddit_gaming_balanced_10k.csv`: final balanced dataset used for training.
- `scripts/filter_reddit_gaming.py`: filters Reddit data to gaming communities.
- `scripts/create_labeling_sample.py`: creates a balanced sample for labeling.
- `scripts/train_sentiment_pytorch.py`: generates embeddings, trains models, and evaluates them.
- `models/`: trained PyTorch model artifacts.
- `reports/sentiment_pytorch_metrics.json`: evaluation metrics.
- `hf_dataset/`: Hugging Face dataset repo contents.
- `hf_model/`: Hugging Face model repo contents.
- `hf_space/`: Hugging Face Gradio demo contents.
- `REPORT.md`: full project report.

## Dataset

The source data came from the Hugging Face dataset `gk4u/reddit_dataset_218`. I filtered it to gaming-related subreddits and removed duplicates, deleted/removed content, spam-like text, and very short comments.

The Hugging Face dataset upload contains only the final balanced 10k dataset.

## Training

The latest run used:

- Embeddings: `qwen3-embedding:0.6b`
- Training data: `data/reddit_gaming_balanced_10k.csv`
- Split: 70% train, 15% dev, 15% test
- Classifiers:
  - Gaussian Naive Bayes implemented with PyTorch tensors
  - Feed-forward neural network implemented in vanilla PyTorch

Run training:

```bash
ollama pull qwen3-embedding:0.6b
python3 scripts/train_sentiment_pytorch.py
```

## Results

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|
| Gaussian Naive Bayes | 0.5200 | 0.5176 | 0.5179 |
| Neural Network | 0.6000 | 0.6006 | 0.6016 |

The neural network is the stronger model, but the results are modest. A likely reason is label noise: the labels were created with LLM assistance and only minimal human review.

## Demo

The Hugging Face Space in `hf_space/` is designed to run on free CPU hardware. It uses `Qwen/Qwen3-Embedding-0.6B` through `sentence-transformers` and the trained PyTorch classifier.

No paid API calls or paid Hugging Face hardware are required.


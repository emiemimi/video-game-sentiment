from pathlib import Path

import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn


EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
NN_MODEL_PATH = Path("models/sentiment_neural_net.pt")
NB_MODEL_PATH = Path("models/sentiment_naive_bayes_torch.pt")


class SentimentNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_torch_file(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

nn_checkpoint = load_torch_file(NN_MODEL_PATH)
id_to_label = {
    int(key): value for key, value in nn_checkpoint["id_to_label"].items()
}
label_order = [id_to_label[index] for index in range(len(id_to_label))]

nn_model = SentimentNet(
    input_dim=int(nn_checkpoint["input_dim"]),
    num_classes=int(nn_checkpoint["num_classes"]),
)
nn_model.load_state_dict(nn_checkpoint["model_state_dict"])
nn_model.eval()

nb_params = load_torch_file(NB_MODEL_PATH)


def embed_text(text: str) -> torch.Tensor:
    embedding = embedding_model.encode(
        [text],
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return torch.tensor(embedding, dtype=torch.float32)


def predict_neural_network(x: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = nn_model(x)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {
        label_order[index]: float(probabilities[index])
        for index in range(len(label_order))
    }


def predict_naive_bayes(x: torch.Tensor) -> dict[str, float]:
    means = nb_params["means"]
    variances = nb_params["variances"]
    log_priors = nb_params["log_priors"]
    x_expanded = x.unsqueeze(1)
    log_likelihood = -0.5 * (
        torch.log(2.0 * torch.tensor(np.pi) * variances)
        + ((x_expanded - means) ** 2) / variances
    ).sum(dim=2)
    log_probs = log_likelihood + log_priors
    probabilities = torch.softmax(log_probs, dim=1)[0].cpu().numpy()
    return {
        label_order[index]: float(probabilities[index])
        for index in range(len(label_order))
    }


def classify(comment: str, classifier: str) -> tuple[str, float, dict[str, float]]:
    text = comment.strip()
    if not text:
        return "neutral", 0.0, {label: 0.0 for label in label_order}

    x = embed_text(text)
    if classifier == "Gaussian Naive Bayes":
        probabilities = predict_naive_bayes(x)
    else:
        probabilities = predict_neural_network(x)

    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]
    return prediction, confidence, probabilities


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(
            label="Gaming comment",
            lines=5,
            placeholder="The combat is fun, but the game runs terribly on my PC.",
        ),
        gr.Radio(
            choices=["Neural Network", "Gaussian Naive Bayes"],
            value="Neural Network",
            label="Classifier",
        ),
    ],
    outputs=[
        gr.Label(label="Predicted sentiment"),
        gr.Number(label="Confidence"),
        gr.Label(label="Class probabilities"),
    ],
    title="Video Game Sentiment Classifier",
    examples=[
        ["The gameplay is smooth and I love the boss fights.", "Neural Network"],
        ["The game looks good but the performance is awful.", "Neural Network"],
        ["I am just waiting for the patch notes before deciding.", "Neural Network"],
    ],
    cache_examples=False,
)


if __name__ == "__main__":
    demo.launch(ssr_mode=False)

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


DEFAULT_INPUT = "data/reddit_gaming_balanced_10k.csv"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
DEFAULT_EMBEDDINGS_CACHE = "data/cache/qwen3_embedding_0.6b_balanced_10k_embeddings.npy"
DEFAULT_METADATA_CACHE = "data/cache/qwen3_embedding_0.6b_balanced_10k_metadata.csv"
DEFAULT_METRICS_PATH = "reports/sentiment_pytorch_metrics.json"
DEFAULT_NN_PATH = "models/sentiment_neural_net.pt"
DEFAULT_NB_PATH = "models/sentiment_naive_bayes_torch.pt"
DEFAULT_LABEL_MAP_PATH = "models/sentiment_label_mapping.json"


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


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("").astype(str)
    df = df[df["text"].str.strip().ne("") & df["sentiment"].str.strip().ne("")]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("Input dataset has no usable rows after removing empty text/labels.")
    return df


def cache_metadata(df: pd.DataFrame, model: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": np.arange(len(df), dtype=np.int64),
            "text_hash": [text_hash(text) for text in df["text"].tolist()],
            "sentiment": df["sentiment"].tolist(),
            "embedding_model": model,
        }
    )


def load_cached_embeddings(
    embeddings_path: Path,
    metadata_path: Path,
    expected_metadata: pd.DataFrame,
) -> np.ndarray | None:
    if not embeddings_path.exists() or not metadata_path.exists():
        return None

    cached_metadata = pd.read_csv(metadata_path)
    if len(cached_metadata) != len(expected_metadata):
        return None

    comparable_columns = ["row_index", "text_hash", "sentiment", "embedding_model"]
    if any(column not in cached_metadata.columns for column in comparable_columns):
        return None
    if cached_metadata[comparable_columns].astype(str).equals(
        expected_metadata[comparable_columns].astype(str)
    ):
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] == len(expected_metadata):
            print(f"Loaded cached embeddings from {embeddings_path}")
            return embeddings.astype(np.float32)
    return None


def request_embeddings(
    texts: list[str],
    model: str,
    ollama_url: str,
    batch_size: int,
    timeout: int,
) -> np.ndarray:
    all_embeddings: list[list[float]] = []
    session = requests.Session()

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding with Ollama"):
        batch = texts[start : start + batch_size]
        payload = {"model": model, "input": batch}
        try:
            response = session.post(ollama_url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Ensure Ollama is running and reachable at "
                f"{ollama_url}."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                "Ollama embedding request failed. "
                f"Status={response.status_code}; body={response.text[:1000]}"
            )

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                "Ollama response did not include embeddings. "
                f"Response keys: {sorted(data.keys())}"
            )
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(embeddings)} embeddings for a batch of {len(batch)} texts."
            )
        all_embeddings.extend(embeddings)

    return np.asarray(all_embeddings, dtype=np.float32)


def get_embeddings(
    df: pd.DataFrame,
    model: str,
    ollama_url: str,
    batch_size: int,
    timeout: int,
    embeddings_path: Path,
    metadata_path: Path,
) -> np.ndarray:
    expected_metadata = cache_metadata(df, model)
    cached = load_cached_embeddings(embeddings_path, metadata_path, expected_metadata)
    if cached is not None:
        return cached

    print("No valid embedding cache found. Generating embeddings with Ollama.")
    embeddings = request_embeddings(
        texts=df["text"].tolist(),
        model=model,
        ollama_url=ollama_url,
        batch_size=batch_size,
        timeout=timeout,
    )
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    expected_metadata.to_csv(metadata_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved embedding metadata to {metadata_path}")
    return embeddings


def encode_labels(labels: list[str]) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    label_names = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    return encoded, label_to_id, id_to_label


def stratified_split(
    y: np.ndarray,
    train_frac: float,
    dev_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    dev_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(np.unique(y)):
        class_indices = np.flatnonzero(y == label)
        rng.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(round(n_total * train_frac))
        n_dev = int(round(n_total * dev_frac))
        n_train = max(1, min(n_train, n_total - 2))
        n_dev = max(1, min(n_dev, n_total - n_train - 1))

        train_indices.extend(class_indices[:n_train].tolist())
        dev_indices.extend(class_indices[n_train : n_train + n_dev].tolist())
        test_indices.extend(class_indices[n_train + n_dev :].tolist())

    train_indices = np.asarray(train_indices, dtype=np.int64)
    dev_indices = np.asarray(dev_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(dev_indices)
    rng.shuffle(test_indices)
    return train_indices, dev_indices, test_indices


def fit_gaussian_nb(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    var_smoothing: float,
) -> dict[str, torch.Tensor]:
    means = []
    variances = []
    priors = []
    for class_id in range(num_classes):
        class_x = x_train[y_train == class_id]
        if len(class_x) == 0:
            raise ValueError(f"Class {class_id} has no training rows.")
        means.append(class_x.mean(dim=0))
        variances.append(class_x.var(dim=0, unbiased=False) + var_smoothing)
        priors.append(torch.tensor(len(class_x) / len(x_train), dtype=torch.float32))

    return {
        "means": torch.stack(means),
        "variances": torch.stack(variances),
        "log_priors": torch.log(torch.stack(priors)),
        "var_smoothing": torch.tensor(var_smoothing, dtype=torch.float32),
    }


def predict_gaussian_nb(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    means = params["means"]
    variances = params["variances"]
    log_priors = params["log_priors"]
    x_expanded = x.unsqueeze(1)
    log_likelihood = -0.5 * (
        torch.log(2.0 * torch.tensor(math.pi) * variances)
        + ((x_expanded - means) ** 2) / variances
    ).sum(dim=2)
    log_probs = log_likelihood + log_priors
    return torch.argmax(log_probs, dim=1)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_to_label: dict[int, str],
) -> dict:
    num_classes = len(id_to_label)
    matrix = confusion_matrix_np(y_true, y_pred, num_classes)
    per_class = {}
    f1_values = []
    weighted_f1_sum = 0.0
    total = len(y_true)

    for class_id in range(num_classes):
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - matrix[class_id, class_id])
        fn = float(matrix[class_id, :].sum() - matrix[class_id, class_id])
        support = int(matrix[class_id, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        weighted_f1_sum += f1 * support
        per_class[id_to_label[class_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = float((y_true == y_pred).mean()) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": float(weighted_f1_sum / total) if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def train_neural_net(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_dev: torch.Tensor,
    y_dev: torch.Tensor,
    num_classes: int,
    id_to_label: dict[int, str],
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[SentimentNet, list[dict], int]:
    torch.manual_seed(seed)
    model = SentimentNet(input_dim=x_train.shape[1], num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(x_train, y_train)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)

        model.eval()
        with torch.no_grad():
            dev_pred = model(x_dev).argmax(dim=1).cpu().numpy()
        dev_metrics = compute_metrics(y_dev.cpu().numpy(), dev_pred, id_to_label)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / len(x_train),
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_accuracy": dev_metrics["accuracy"],
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:02d} loss={epoch_record['train_loss']:.4f} "
            f"dev_macro_f1={epoch_record['dev_macro_f1']:.4f} "
            f"dev_accuracy={epoch_record['dev_accuracy']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def evaluate_model_predictions(
    name: str,
    y_dev: np.ndarray,
    dev_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    id_to_label: dict[int, str],
) -> dict:
    result = {
        "dev": compute_metrics(y_dev, dev_pred, id_to_label),
        "test": compute_metrics(y_test, test_pred, id_to_label),
    }
    print_metrics(name, result)
    return result


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    for split in ["dev", "test"]:
        split_metrics = metrics[split]
        print(
            f"  {split}: accuracy={split_metrics['accuracy']:.4f} "
            f"macro_f1={split_metrics['macro_f1']:.4f} "
            f"weighted_f1={split_metrics['weighted_f1']:.4f}"
        )
        for label, label_metrics in split_metrics["per_class"].items():
            print(
                f"    {label}: precision={label_metrics['precision']:.4f} "
                f"recall={label_metrics['recall']:.4f} "
                f"f1={label_metrics['f1']:.4f} "
                f"support={label_metrics['support']}"
            )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PyTorch sentiment classifiers on Ollama embeddings."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--ollama-timeout", type=int, default=120)
    parser.add_argument("--embeddings-cache", default=DEFAULT_EMBEDDINGS_CACHE)
    parser.add_argument("--metadata-cache", default=DEFAULT_METADATA_CACHE)
    parser.add_argument("--metrics-output", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--nn-output", default=DEFAULT_NN_PATH)
    parser.add_argument("--nb-output", default=DEFAULT_NB_PATH)
    parser.add_argument("--label-map-output", default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--var-smoothing", type=float, default=1e-6)
    args = parser.parse_args()

    start_time = time.time()
    df = load_dataset(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    embeddings = get_embeddings(
        df=df,
        model=args.embedding_model,
        ollama_url=args.ollama_url,
        batch_size=args.embedding_batch_size,
        timeout=args.ollama_timeout,
        embeddings_path=Path(args.embeddings_cache),
        metadata_path=Path(args.metadata_cache),
    )
    if embeddings.shape[0] != len(df):
        raise ValueError(
            f"Embedding row count mismatch: embeddings={embeddings.shape[0]} dataset={len(df)}"
        )
    print(f"Embedding matrix shape: {embeddings.shape}")

    y, label_to_id, id_to_label = encode_labels(df["sentiment"].tolist())
    train_idx, dev_idx, test_idx = stratified_split(
        y=y,
        train_frac=0.70,
        dev_frac=0.15,
        seed=args.seed,
    )
    print(f"Split sizes: train={len(train_idx)} dev={len(dev_idx)} test={len(test_idx)}")

    x_tensor = torch.tensor(embeddings, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
    x_dev, y_dev = x_tensor[dev_idx], y_tensor[dev_idx]
    x_test, y_test = x_tensor[test_idx], y_tensor[test_idx]

    nb_params = fit_gaussian_nb(
        x_train=x_train,
        y_train=y_train,
        num_classes=len(label_to_id),
        var_smoothing=args.var_smoothing,
    )
    with torch.no_grad():
        nb_dev_pred = predict_gaussian_nb(nb_params, x_dev).cpu().numpy()
        nb_test_pred = predict_gaussian_nb(nb_params, x_test).cpu().numpy()
    nb_metrics = evaluate_model_predictions(
        "Gaussian Naive Bayes",
        y_dev.cpu().numpy(),
        nb_dev_pred,
        y_test.cpu().numpy(),
        nb_test_pred,
        id_to_label,
    )

    nn_model, nn_history, best_epoch = train_neural_net(
        x_train=x_train,
        y_train=y_train,
        x_dev=x_dev,
        y_dev=y_dev,
        num_classes=len(label_to_id),
        id_to_label=id_to_label,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    nn_model.eval()
    with torch.no_grad():
        nn_dev_pred = nn_model(x_dev).argmax(dim=1).cpu().numpy()
        nn_test_pred = nn_model(x_test).argmax(dim=1).cpu().numpy()
    nn_metrics = evaluate_model_predictions(
        "Neural Network",
        y_dev.cpu().numpy(),
        nn_dev_pred,
        y_test.cpu().numpy(),
        nn_test_pred,
        id_to_label,
    )

    label_mapping_payload = {
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
    }
    save_json(Path(args.label_map_output), label_mapping_payload)

    Path(args.nb_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **nb_params,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "embedding_model": args.embedding_model,
        },
        args.nb_output,
    )

    Path(args.nn_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": nn_model.state_dict(),
            "input_dim": embeddings.shape[1],
            "num_classes": len(label_to_id),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "embedding_model": args.embedding_model,
            "best_epoch": best_epoch,
            "architecture": "Linear(input_dim, 256) -> ReLU -> Dropout(0.2) -> Linear(256, num_classes)",
        },
        args.nn_output,
    )

    metrics_payload = {
        "input": args.input,
        "embedding_model": args.embedding_model,
        "embedding_shape": list(embeddings.shape),
        "split_sizes": {
            "train": int(len(train_idx)),
            "dev": int(len(dev_idx)),
            "test": int(len(test_idx)),
        },
        "label_mapping": label_mapping_payload,
        "classifiers": {
            "gaussian_naive_bayes": nb_metrics,
            "neural_network": {
                **nn_metrics,
                "history": nn_history,
                "best_epoch": best_epoch,
            },
        },
        "runtime_seconds": time.time() - start_time,
    }
    save_json(Path(args.metrics_output), metrics_payload)

    print(f"\nSaved metrics to {args.metrics_output}")
    print(f"Saved Naive Bayes parameters to {args.nb_output}")
    print(f"Saved neural network checkpoint to {args.nn_output}")
    print(f"Saved label mapping to {args.label_map_output}")


if __name__ == "__main__":
    main()

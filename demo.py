"""Run python demo.py --interactive to compare six saved models."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn

ROOT = Path(__file__).resolve().parent
CLASSES = ["neutral/positive", "negative attitude", "derogatory"]


class FeedforwardNN(nn.Module):
    """Same mean-pooling model and parameter names as the training notebook."""

    def __init__(self, weights, settings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        layers = []
        width = weights.shape[1]
        config = settings["config"]
        for hidden, activation in zip(config["hidden_sizes"], config["activations"]):
            layers += [nn.Linear(width, hidden),
                       nn.ReLU() if activation == "relu" else nn.Sigmoid(),
                       nn.Dropout(settings["dropout"])]
            width = hidden
        layers.append(nn.Linear(width, 3))
        self.classifier = nn.Sequential(*layers)

    def forward(self, X):
        mask = (X != 0).unsqueeze(-1)
        vectors = self.embedding(X)
        pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.log_softmax(self.classifier(pooled), dim=1)


class LSTMClassifier(nn.Module):
    """Same packed-sequence LSTM as the training notebook."""

    def __init__(self, weights, settings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        config = settings["config"]
        self.lstm = nn.LSTM(weights.shape[1], config["hidden_size"], batch_first=True)
        self.dropout = nn.Dropout(settings["dropout"])
        self.classifier = nn.Linear(config["hidden_size"], 3)
        self.output_activation = config["output_activation"]

    def forward(self, X):
        lengths = (X != 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(X), lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        logits = self.classifier(self.dropout(hidden[-1]))
        if self.output_activation == "softmax":
            return torch.log_softmax(logits, dim=1)
        log_scores = nn.functional.logsigmoid(logits)
        return log_scores - torch.logsumexp(log_scores, dim=1, keepdim=True)


def encode_texts(texts, vocabulary, max_length):
    """Reuse training tokenization, vocabulary, truncation, and right padding."""
    X = np.zeros((len(texts), max_length), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", text.lower())
        ids = [vocabulary.get(word, 1) for word in tokens[:max_length]] or [1]
        X[i, :len(ids)] = ids
    return X


def load_neural_models():
    metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
    vocabulary = json.loads((ROOT / "artifacts/vocabulary.json").read_text())
    results = pd.read_csv(ROOT / "all_experiments.csv")
    winners = results[results["selected_for_family"].astype(str).str.lower().isin(["true", "1"])]
    models = {}
    for family, name, model_class in [("feedforward", "Feedforward", FeedforwardNN),
                                       ("lstm", "LSTM", LSTMClassifier)]:
        rows = winners[winners["model"] == ("Feedforward NN" if family == "feedforward" else "LSTM")]
        if len(rows) != 1:
            raise ValueError(f"Expected one validation winner for {name}.")
        run = rows.iloc[0]["run_id"]
        folder = ROOT / "results" / family
        settings = json.loads((folder / f"{run}_settings.json").read_text())
        if settings["preparation_id"] != metadata["preparation_id"]:
            raise ValueError("Checkpoint and vocabulary preparations differ.")
        state = torch.load(folder / f"{run}.pt", map_location="cpu", weights_only=True)
        model = model_class(state["embedding.weight"], settings)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    return models, vocabulary, metadata


def predict_neural(model, X):
    with torch.inference_mode():
        batches = [model(torch.from_numpy(X[i:i + 256])).argmax(dim=1).numpy()
                   for i in range(0, len(X), 256)]
    return np.concatenate(batches)


def clean_messages(messages):
    # The ML notebook uses a different preprocessing step from the neural models.
    return pd.Series(messages, dtype=str).str.lower().str.replace(
        r"http\S+", "", regex=True).str.strip()


def predict_all(messages, neural, ml, vocabulary, metadata):
    X = encode_texts(messages, vocabulary, metadata["max_length"])
    predictions = {name: predict_neural(model, X) for name, model in neural.items()}
    cleaned = clean_messages(messages)
    for name, model in ml.items():
        predictions[name] = model.predict(cleaned)
    return predictions


def evaluate(test, neural, ml, vocabulary, metadata):
    predictions = predict_all(test["comment_text"].astype(str).tolist(),
                              neural, ml, vocabulary, metadata)
    output = ROOT / "demo_results"
    output.mkdir(exist_ok=True)
    summary, per_class = [], []
    for name, predicted in predictions.items():
        binary = name == "SVM"
        y = np.where(test["label"] == 0, 1, -1) if binary else test["label"]
        labels = [1, -1] if binary else [0, 1, 2]
        names = ["neutral/positive", "toxic"] if binary else CLASSES
        report = classification_report(y, predicted, labels=labels,
                                       target_names=names, output_dict=True, zero_division=0)
        macro = report["macro avg"]
        summary.append({"Model": "SVM (binary)" if binary else name,
                        "Accuracy": accuracy_score(y, predicted),
                        "Macro P": macro["precision"], "Macro R": macro["recall"],
                        "Macro F1": macro["f1-score"]})
        for label, class_name in zip(labels, names):
            per_class.append({"model": name, "class_id": label,
                              "class_name": class_name, **report[class_name]})
        pd.DataFrame(confusion_matrix(y, predicted, labels=labels),
                     index=names, columns=names).to_csv(output / f"{name}_confusion_matrix.csv")
    summary = pd.DataFrame(summary).set_index("Model") * 100
    summary.to_csv(output / "summary.csv", float_format="%.2f")
    pd.DataFrame(per_class).to_csv(output / "per_class_scores.csv", index=False)
    print("\nTest scores (%)")
    print(summary.to_string(float_format=lambda value: f"{value:.2f}"))
    print("SVM uses 2 classes: neutral/positive vs toxic. Others use 3 classes.")


def show_message(message, neural, ml, vocabulary, metadata):
    if not message.strip():
        return
    predictions = predict_all([message], neural, ml, vocabulary, metadata)
    for name, predicted in predictions.items():
        label = int(predicted[0])
        if name == "SVM":
            result = "neutral/positive" if label == 1 else "toxic"
            print(f"{'SVM (binary)':14s}: {result}")
        else:
            print(f"{name:14s}: {CLASSES[label]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="Try your own messages.")
    parser.add_argument("--text", help="Predict one message.")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    neural, vocabulary, metadata = load_neural_models()
    model_path = ROOT / "results/ml/models.joblib"
    if not model_path.exists():
        raise FileNotFoundError("ML models are missing. Run prepare_ml_models.py first.")
    ml = joblib.load(model_path)

    # --text is useful for a quick prediction without evaluating all 6,600 rows.
    if args.text is not None:
        show_message(args.text, neural, ml, vocabulary, metadata)
        return
    test_path = ROOT / "test.csv"
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != metadata["csv_sha256"]["test"]:
        raise ValueError("test.csv does not match the saved test split.")
    test = pd.read_csv(test_path)
    evaluate(test, neural, ml, vocabulary, metadata)
    if args.interactive:
        print("\nType a message, or quit to exit.")
        while True:
            try:
                message = input("\nMessage> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if message.strip().lower() in {"quit", "exit"}:
                break
            show_message(message, neural, ml, vocabulary, metadata)


if __name__ == "__main__":
    main()

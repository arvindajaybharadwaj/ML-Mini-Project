"""Evaluate the two validation-selected checkpoints, then optionally try live text.

Run: python demo.py
Live demo: python demo.py --interactive
One message: python demo.py --text "gg well played team"

Requires: numpy, pandas, torch, scikit-learn. No training or embedding downloads.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn


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


def select_checkpoints(csv_path):
    """Use the existing validation selections, never the highest test score."""
    results = pd.read_csv(csv_path)
    if "model" in results.columns:
        results["family"] = results["model"].map({"Feedforward NN": "feedforward", "LSTM": "lstm"})
    required = {"family", "run_id"}
    if not required.issubset(results.columns):
        raise ValueError("The results CSV needs model/family and run_id columns.")
    if "selected_for_family" in results.columns:
        selected = results["selected_for_family"].astype(str).str.lower().isin(["true", "1"])
        winners = results.loc[selected].copy()
    else:
        # Also accept the original evaluation/all_validation_results.csv.
        if "validation_macro_f1" in results.columns:
            results = results.rename(columns={"validation_macro_f1": "val_macro_f1"})
        if "val_macro_f1" not in results.columns:
            raise ValueError("CSV needs selected_for_family or validation macro F1.")
        if results["val_macro_f1"].isna().any():
            raise ValueError("Validation macro F1 contains missing scores.")
        columns = ["val_macro_f1"] + (["val_loss"] if "val_loss" in results.columns else []) + ["run_id"]
        ascending = [False] + ([True] if "val_loss" in results.columns else []) + [True]
        winners = results.sort_values(columns, ascending=ascending).groupby("family").head(1)
    counts = winners["family"].value_counts().to_dict()
    if counts != {"feedforward": 1, "lstm": 1}:
        raise ValueError("Expected exactly one selected feedforward model and one selected LSTM.")
    return winners.set_index("family").loc[["feedforward", "lstm"]].reset_index()


def load_model(project, family, run_id, metadata):
    directory = project / "results" / family
    settings = json.loads((directory / f"{run_id}_settings.json").read_text())
    if settings["preparation_id"] != metadata["preparation_id"]:
        raise ValueError("Checkpoint and vocabulary come from different preparations.")
    if settings["family"] != family:
        raise ValueError("Checkpoint settings do not match the selected family.")
    # The checkpoint already contains its embedding matrix: no Gensim needed.
    state = torch.load(directory / f"{run_id}.pt", map_location="cpu", weights_only=True)
    weights = state["embedding.weight"]
    if weights.shape[0] != metadata["vocab_size"]:
        raise ValueError("Checkpoint embedding size does not match the vocabulary.")
    model_class = FeedforwardNN if family == "feedforward" else LSTMClassifier
    model = model_class(weights, settings)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, settings


def predict(model, X, batch_size=256):
    with torch.inference_mode():
        batches = [model(torch.from_numpy(X[start:start + batch_size])).exp().numpy()
                   for start in range(0, len(X), batch_size)]
    return np.concatenate(batches)


def evaluate(models, X, test, class_names, output):
    """Calculate fresh scores from checkpoints, rather than display saved scores."""
    summaries, per_class = [], []
    predictions = test[["comment_text", "label"]].rename(columns={"label": "true_label"}).copy()
    y = test["label"].to_numpy()
    for family, (model, settings) in models.items():
        probabilities = predict(model, X)
        predicted = probabilities.argmax(axis=1)
        report = classification_report(y, predicted, labels=[0, 1, 2],
                                       target_names=class_names, output_dict=True, zero_division=0)
        row = {"model": family, "embedding": settings["embedding"],
               "configuration": settings["config"]["name"],
               "accuracy": accuracy_score(y, predicted),
               "macro_precision": report["macro avg"]["precision"],
               "macro_recall": report["macro avg"]["recall"],
               "macro_f1": report["macro avg"]["f1-score"]}
        summaries.append(row)
        print(f"\n{family.upper()} | {settings['embedding']} | {settings['config']['name']}")
        print(classification_report(y, predicted, labels=[0, 1, 2],
                                    target_names=class_names, digits=4, zero_division=0))
        matrix = pd.DataFrame(confusion_matrix(y, predicted, labels=[0, 1, 2]),
                              index=class_names, columns=class_names)
        matrix.index.name = "true_class"
        matrix.columns.name = "predicted_class"
        print("Confusion matrix (rows = true; columns = predicted):")
        print(matrix.to_string())
        matrix.to_csv(output / f"{family}_confusion_matrix.csv")
        for label, name in enumerate(class_names):
            per_class.append({"model": family, "class_id": label, "class_name": name,
                              **report[name]})
        predictions[f"{family}_predicted_label"] = predicted
        for label in [0, 1, 2]:
            predictions[f"{family}_probability_class_{label}"] = probabilities[:, label]
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class_scores.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    print("\nCOMPARISON (all scores below are percentages)")
    display = summary.set_index("model")[["accuracy", "macro_precision", "macro_recall", "macro_f1"]] * 100
    print(display.to_string(float_format=lambda value: f"{value:.2f}"))


def demo_message(text, models, vocabulary, metadata):
    if not text.strip():
        print("Please enter a nonempty message.")
        return
    X = encode_texts([text], vocabulary, metadata["max_length"])
    print(f"\nMessage: {text}")
    for family, (model, _) in models.items():
        probabilities = predict(model, X)[0]
        label = int(probabilities.argmax())
        print(f"  {family:12s}: {metadata['class_names'][label]} (class {label})")
        print("    " + "; ".join(f"{name}: {probabilities[i]:.1%}"
                                    for i, name in enumerate(metadata["class_names"])))
    unknown = int((X == 1).sum())
    if unknown:
        print(f"  Vocabulary note: {unknown} token(s) mapped to <UNK>.")


def main():
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, Path.cwd(), Path("/Users/arvind/Documents/Sem5/ML/Mini Project")]
    default_project = next((p for p in candidates if (p / "artifacts/metadata.json").exists()), here)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=Path, default=default_project,
                        help="Folder containing test.csv, artifacts/, and results/.")
    parser.add_argument("--results-csv", type=Path, help="The condensed all_experiments.csv or original validation CSV.")
    parser.add_argument("--output-dir", type=Path, help="Where to save fresh evaluation CSVs (default: project/demo_results).")
    parser.add_argument("--interactive", action="store_true", help="After evaluation, type messages until you enter quit.")
    parser.add_argument("--text", help="Classify one supplied message after evaluating the test set.")
    args = parser.parse_args()
    project = args.project_dir.expanduser().resolve()
    if args.results_csv:
        results_csv = args.results_csv.expanduser().resolve()
    else:
        choices = [here / "all_experiments.csv", project / "dl_results/all_experiments.csv",
                   project / "report_results/all_experiments.csv", project / "all_experiments.csv",
                   project / "results/evaluation/all_validation_results.csv"]
        results_csv = next((p for p in choices if p.exists()), choices[-1])
    metadata = json.loads((project / "artifacts/metadata.json").read_text())
    vocabulary = json.loads((project / "artifacts/vocabulary.json").read_text())
    if len(vocabulary) != metadata["vocab_size"] or vocabulary.get("<PAD>") != 0 or vocabulary.get("<UNK>") != 1:
        raise ValueError("Vocabulary does not match the saved preparation.")
    test_path = project / "test.csv"
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != metadata["csv_sha256"]["test"]:
        raise ValueError("test.csv differs from the test split recorded during preparation.")
    test = pd.read_csv(test_path)
    if test.empty or test[["comment_text", "label"]].isna().any().any() or not test["label"].isin([0, 1, 2]).all():
        raise ValueError("Test data must have nonmissing comment_text and labels 0, 1, 2.")
    X = encode_texts(test["comment_text"].astype(str).tolist(), vocabulary, metadata["max_length"])
    winners = select_checkpoints(results_csv)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Results CSV: {results_csv}")
    print(f"Test messages: {len(test):,}; device: CPU")
    print("Selection: saved validation winners; inference from checkpoint weights.")
    models = {}
    for row in winners.itertuples():
        models[row.family] = load_model(project, row.family, row.run_id, metadata)
        print(f"Loaded {row.family}: {row.run_id}")
    output = (args.output_dir or project / "demo_results").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluate(models, X, test, metadata["class_names"], output)
    print(f"\nSaved fresh evaluation results to: {output}")
    if args.text:
        demo_message(args.text, models, vocabulary, metadata)
    if args.interactive:
        print("\nLive demo: enter a chat message. Type quit to finish.")
        while True:
            try:
                text = input("\nMessage> ")
            except (EOFError, KeyboardInterrupt):
                print("\nDemo finished.")
                break
            if text.strip().lower() in {"quit", "exit"}:
                break
            demo_message(text, models, vocabulary, metadata)


if __name__ == "__main__":
    main()

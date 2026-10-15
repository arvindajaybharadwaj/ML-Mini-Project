"""Train the four ML winners once. The demo only loads their saved models.

Run: python prepare_ml_models.py --train /path/to/train.csv
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=root / "train.csv")
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    messages = train["comment_text"].astype(str).str.lower().str.replace(
        r"http\S+", "", regex=True).str.strip()
    y = train["label"].to_numpy()

    # Use the notebook's published results to pick one model per ML family.
    results = pd.read_csv(root / "results.csv")
    prefixes = {"SVM": "SVM", "Naive Bayes": "Naive Bayes",
                "KNN": "KNN", "Random Forest": "Random Forest"}
    selected = {}
    for family, prefix in prefixes.items():
        rows = results[results["model"].str.startswith(prefix)]
        selected[family] = rows.sort_values(
            ["macro_f1", "accuracy"], ascending=False).iloc[0]["model"]

    # Same word-list construction as the notebook. Sort ties for repeatability.
    def word_counts(label_mask):
        return Counter(word for message in messages[label_mask]
                       for word in sorted(set(re.findall(r"[a-z']+", message))))

    neutral, toxic = word_counts(y == 0), word_counts(y > 0)
    remove = set("you are this your like why what how was only who know because really "
                 "have with that they just can get for the and not but all from its too when".split())
    candidates = [word for word, count in toxic.items()
                  if count >= 15 and count >= 4 * (neutral.get(word, 0) + 1)
                  and len(word) > 2 and word not in remove]
    phrases = sorted(candidates, key=lambda word: (-toxic[word], word))[:165]

    options = {
        **{f"SVM {kernel} (binary)": make_pipeline(TfidfVectorizer(), SVC(kernel=kernel))
           for kernel in ["linear", "rbf", "poly"]},
        "Naive Bayes baseline (word list)": make_pipeline(
            CountVectorizer(vocabulary=phrases), MultinomialNB()),
        "Naive Bayes (TF-IDF)": make_pipeline(TfidfVectorizer(), MultinomialNB()),
        **{f"KNN K={k}": make_pipeline(TfidfVectorizer(), KNeighborsClassifier(n_neighbors=k))
           for k in [1, 2, 3, 6]},
        "Random Forest (Gini)": make_pipeline(TfidfVectorizer(),
            RandomForestClassifier(criterion="gini", random_state=42, n_jobs=-1)),
        "Random Forest (Entropy)": make_pipeline(TfidfVectorizer(),
            RandomForestClassifier(criterion="entropy", random_state=42, n_jobs=-1)),
        "Random Forest (Entropy, max_depth=32)": make_pipeline(TfidfVectorizer(),
            RandomForestClassifier(criterion="entropy", max_depth=32, random_state=42, n_jobs=-1)),
    }
    models = {}
    for family, name in selected.items():
        print(f"Training {name}...", flush=True)
        target = np.where(y == 0, 1, -1) if family == "SVM" else y
        models[family] = options[name].fit(messages, target)

    output = root / "results" / "ml"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output / "models.joblib", compress=3)
    pd.DataFrame(selected.items(), columns=["family", "model"]).to_csv(
        output / "selected_models.csv", index=False)
    print(f"Saved models (scikit-learn {sklearn.__version__}) to {output}")


if __name__ == "__main__":
    main()

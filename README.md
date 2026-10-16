# Classifying Sentiment in the Language of PvP Game Communications

1. **Name:** Arvind Ajay Bharadwaj, **SRN:** PES1UG24AM052
2. **Name:** Amogh B Shetty, **SRN:** PES1UG24AM034

Classify game chat messages using six trained models: a Feedforward Neural Network, LSTM, SVM, Naive Bayes, KNN, and Random Forest.

The demo compares their test scores and lets you enter messages to see each model's prediction. It loads saved models, so training is not required.

## Installation

Use Python 3.12. Download or clone the project, then open a terminal in the folder containing `demo.py`.

Create and activate a virtual environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Use the scikit-learn version pinned in `requirements.txt` to load the saved ML models.

## Usage

### Interactive demo

```bash
python demo.py --interactive
```

The demo evaluates the test set and prints accuracy, macro precision, macro recall, and macro F1. It then prompts for a message:

```text
Message> gg well played team
Feedforward   : neutral/positive
LSTM          : neutral/positive
SVM (binary)  : neutral/positive
Naive Bayes   : neutral/positive
KNN           : neutral/positive
Random Forest : neutral/positive
```

Enter another message to continue, or type `quit` to exit.

### Classify one message

```bash
python demo.py --text "gg well played team"
```

This predicts the message without evaluating the test set.

### Evaluate the test set

```bash
python demo.py
```

Evaluation saves these files in `demo_results/`:

- `summary.csv`: accuracy and macro scores, expressed as percentages.
- `per_class_scores.csv`: precision, recall, F1, and support for each class. Scores range from 0 to 1.
- `<model>_confusion_matrix.csv`: one matrix per model, with true classes as rows and predicted classes as columns.

## Models and labels

| Model          | Configuration                                 | Output classes |
| -------------- | --------------------------------------------- | -------------- |
| Feedforward NN | Word2Vec 100d, ReLU layers of 64 and 32 units | Three classes  |
| LSTM           | Word2Vec 100d, 128 hidden units               | Three classes  |
| SVM            | Linear kernel, TF-IDF                         | Two classes    |
| Naive Bayes    | Word-list features                            | Three classes  |
| KNN            | K = 6, TF-IDF                                 | Three classes  |
| Random Forest  | Gini criterion, TF-IDF                        | Three classes  |

The three-class models predict **neutral/positive**, **negative attitude**, or **derogatory**. SVM predicts **neutral/positive** or **toxic**, combining the two negative classes. Its scores describe a different task and should be interpreted separately.

## Required files

Keep this directory structure when running the demo:

```text
.
├── demo.py
├── requirements.txt
├── all_experiments.csv
├── test.csv
├── artifacts/
│   ├── metadata.json
│   └── vocabulary.json
└── results/
    ├── feedforward/
    │   ├── word2vec_100__relu_relu.pt
    │   └── word2vec_100__relu_relu_settings.json
    ├── lstm/
    │   ├── word2vec_100__h128_softmax.pt
    │   └── word2vec_100__h128_softmax_settings.json
    └── ml/
        └── models.joblib
```

`test.csv` is required for evaluation. Single-message predictions with `--text` do not require it.

## Training and experiments

`Amogh_ML_classifiers.ipynb` contains the classical ML experiments. The notebook is not required to run the demo: `models.joblib` already contains the fitted classifiers and their vectorizers.

To rebuild the ML models, provide `train.csv` and keep `results.csv` beside `prepare_ml_models.py`, then run:

```bash
python prepare_ml_models.py --train train.csv
```

The script trains on the training split and saves the models to `results/ml/models.joblib`. It selects one model per ML family using macro F1 from `results.csv`, with accuracy breaking ties. These are the notebook's reported test scores, so this reproduces its existing selections rather than providing an independent model-selection evaluation.

The neural models use the saved validation selections in `all_experiments.csv`.

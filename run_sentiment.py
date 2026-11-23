#!/usr/bin/env python3
"""Sentiment pipelines for the Practical Machine Learning joint project.

Representations required by the brief:
  1. Bag-of-words / count vectorization
  2. Term frequency (TF)
  3. Inverse document frequency (IDF)
  4. TF-IDF
  5. Dense Word2Vec embeddings (mean-pooled)

Every learned representation is classified with the same L2 logistic
regression so the comparison reflects the text representation. A small
lexicon baseline is included so the learned pipelines have a reference point.
Final metrics are computed only on ttestPos.en and ttestNeg.en.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

PROJECT = ROOT.parent
DEFAULT_ZIP = PROJECT / "Sentiment Analysis Twitter.zip"

SEED = 42
MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
VAL_FRACTION = 0.1
C_GRID = (0.1, 1.0, 4.0)
W2V_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 5

POS_WORDS = {
    "love", "loved", "loves", "good", "great", "happy", "thanks", "thank",
    "awesome", "nice", "best", "amazing", "glad", "fun", "excited", "yay",
    "wonderful", "beautiful", "cool", "win", "won", "congrats", "sweet",
    "cute", "haha", "lol", "lmao", "enjoy", "enjoyed", "excellent", "perfect",
    "proud", "wow", "smile", "smiling", "fantastic", "pleased",
}
NEG_WORDS = {
    "sad", "sadly", "bad", "hate", "hates", "hated", "miss", "missing",
    "sorry", "worst", "terrible", "awful", "sick", "cry", "crying", "ugly",
    "bored", "boring", "angry", "mad", "upset", "hurt", "hurts", "pain",
    "sucks", "suck", "stupid", "annoyed", "annoying", "lonely", "depressed",
    "worried", "worry", "scared", "horrible", "miserable", "fail", "failed",
}
NEGATION = {
    "not", "no", "never", "nothing", "nobody", "neither", "nor", "without",
    "dont", "doesnt", "didnt", "cant", "wont", "isnt", "wasnt", "arent",
    "hardly", "barely",
}
LAUGHTER = {"lol", "lmao", "haha", "hehe", "rofl"}
SARCASM_PHRASES = (
    "yeah right",
    "thanks a lot",
    "just great",
    "oh great",
    "lucky me",
    "as if",
)

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    """Lowercase, strip URLs and mentions, expand contractions, keep words."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].replace('""', '"')
    text = text.lower()
    text = (
        text.replace("&amp;", " and ")
        .replace("&lt;", " ")
        .replace("&gt;", " ")
        .replace("&quot;", " ")
    )
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = text.replace("n't", " not ")
    text = text.replace("'m", " am ").replace("'re", " are ")
    text = text.replace("'ll", " will ").replace("'ve", " have ")
    text = _NON_ALPHA.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def load_lines(zip_path: Path, name: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(name).decode("utf-8", errors="replace")
    return raw.splitlines()


def sample_indices(n_items: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    n_sample = min(n_sample, n_items)
    return rng.choice(n_items, size=n_sample, replace=False)


def stratified_split(
    labels: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    val_parts = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def make_vectorizer(kind: str) -> CountVectorizer | TfidfVectorizer:
    common = dict(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        lowercase=False,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    if kind == "bow":
        return CountVectorizer(**common)
    if kind == "tf":
        return TfidfVectorizer(use_idf=False, norm="l2", sublinear_tf=False, **common)
    if kind == "idf":
        # Binary presence isolates the IDF weight from raw term frequency.
        return TfidfVectorizer(
            use_idf=True, binary=True, norm="l2", smooth_idf=True, **common
        )
    if kind == "tfidf":
        return TfidfVectorizer(
            use_idf=True, binary=False, norm="l2", sublinear_tf=True, smooth_idf=True, **common
        )
    raise ValueError(kind)


def fit_logistic(features, labels, C: float) -> LogisticRegression:
    clf = LogisticRegression(C=C, solver="liblinear", max_iter=500)
    clf.fit(features, labels)
    return clf


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_neg": float(precision[0]),
        "recall_neg": float(recall[0]),
        "f1_neg": float(f1[0]),
        "precision_pos": float(precision[1]),
        "recall_pos": float(recall[1]),
        "f1_pos": float(f1[1]),
        "support_neg": int(support[0]),
        "support_pos": int(support[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def top_linear_features(clf: LogisticRegression, vectorizer, k: int = 12) -> dict:
    names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_.ravel()
    positive = np.argsort(coef)[-k:][::-1]
    negative = np.argsort(coef)[:k]
    return {
        "positive": [{"term": names[i], "weight": float(coef[i])} for i in positive],
        "negative": [{"term": names[i], "weight": float(coef[i])} for i in negative],
    }


def idf_extremes(vectorizer: TfidfVectorizer, k: int = 8) -> dict:
    names = np.array(vectorizer.get_feature_names_out())
    idf = vectorizer.idf_
    high = np.argsort(idf)[-k:][::-1]
    low = np.argsort(idf)[:k]
    return {
        "highest_idf": [{"term": names[i], "idf": float(idf[i])} for i in high],
        "lowest_idf": [{"term": names[i], "idf": float(idf[i])} for i in low],
    }


def tune_sparse(kind: str, train_text, val_text, y_train, y_val, c_grid) -> tuple[float, float, int]:
    vectorizer = make_vectorizer(kind)
    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)
    best_c, best_score = c_grid[0], -1.0
    for c_value in c_grid:
        clf = fit_logistic(x_train, y_train, c_value)
        score = f1_score(y_val, clf.predict(x_val), average="macro", zero_division=0)
        if score > best_score:
            best_c, best_score = c_value, float(score)
    return best_c, best_score, int(len(vectorizer.vocabulary_))


def fit_sparse_final(kind: str, train_text, y_train, test_text, c_value: float):
    vectorizer = make_vectorizer(kind)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    clf = fit_logistic(x_train, y_train, c_value)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1]
    features = top_linear_features(clf, vectorizer)
    extra = {"n_features": int(len(vectorizer.vocabulary_))}
    if kind == "tfidf":
        extra["idf_extremes"] = idf_extremes(vectorizer)
    return pred, proba, features, extra


def tokenise(texts: list[str]) -> list[list[str]]:
    return [text.split() for text in texts]


def train_word2vec(sentences: list[list[str]], epochs: int) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=W2V_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=1,
        negative=5,
        epochs=epochs,
        seed=SEED,
        workers=1,
    )


def embed_sentences(sentences: list[list[str]], model: Word2Vec) -> np.ndarray:
    vectors = model.wv
    embedded = np.zeros((len(sentences), model.vector_size), dtype=np.float32)
    for i, tokens in enumerate(sentences):
        rows = [vectors[token] for token in tokens if token in vectors]
        if rows:
            embedded[i] = np.mean(rows, axis=0)
    return embedded


def embedding_polarity(model: Word2Vec, direction: np.ndarray, k: int = 12) -> dict:
    words = model.wv.index_to_key[:5000]
    matrix = np.vstack([model.wv[word] for word in words])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    unit = direction / (np.linalg.norm(direction) + 1e-9)
    scores = matrix @ unit
    positive = np.argsort(scores)[-k:][::-1]
    negative = np.argsort(scores)[:k]
    return {
        "positive": [{"term": words[i], "alignment": float(scores[i])} for i in positive],
        "negative": [{"term": words[i], "alignment": float(scores[i])} for i in negative],
    }


def tune_embeddings(train_sentences, val_sentences, y_train, y_val, epochs: int, c_grid):
    model = train_word2vec(train_sentences, epochs)
    x_train = embed_sentences(train_sentences, model)
    x_val = embed_sentences(val_sentences, model)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    best_c, best_score = c_grid[0], -1.0
    for c_value in c_grid:
        clf = fit_logistic(x_train, y_train, c_value)
        score = f1_score(y_val, clf.predict(x_val), average="macro", zero_division=0)
        if score > best_score:
            best_c, best_score = c_value, float(score)
    return best_c, best_score, len(model.wv)


def fit_embeddings_final(train_sentences, y_train, test_sentences, c_value: float, epochs: int):
    model = train_word2vec(train_sentences, epochs)
    raw_train = embed_sentences(train_sentences, model)
    raw_test = embed_sentences(test_sentences, model)
    empty = int(np.sum(np.linalg.norm(raw_test, axis=1) == 0))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(raw_train)
    x_test = scaler.transform(raw_test)
    clf = fit_logistic(x_train, y_train, c_value)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1]
    polarity = embedding_polarity(model, clf.coef_.ravel())
    extra = {"vocab_size": int(len(model.wv)), "oov_or_empty_test_tweets": empty}
    return pred, proba, polarity, extra


def lexicon_predict(sentences: list[list[str]]) -> tuple[np.ndarray, int]:
    pred = np.zeros(len(sentences), dtype=int)
    ties = 0
    for i, tokens in enumerate(sentences):
        score = sum(token in POS_WORDS for token in tokens) - sum(token in NEG_WORDS for token in tokens)
        if score == 0:
            ties += 1
            pred[i] = 1
        else:
            pred[i] = 1 if score > 0 else 0
    return pred, ties


def token_count(text: str) -> int:
    return 0 if not text else len(text.split())


def slice_masks(cleaned: list[str]) -> dict[str, np.ndarray]:
    tokens = [text.split() for text in cleaned]
    short = np.array([len(toks) <= 3 for toks in tokens])
    long = np.array([len(toks) >= 25 for toks in tokens])
    negation = np.array([any(tok in NEGATION for tok in toks) for toks in tokens])
    laughter = np.array([any(tok in LAUGHTER for tok in toks) for toks in tokens])
    sarcasm = np.array([any(phrase in text for phrase in SARCASM_PHRASES) for text in cleaned])
    return {
        "short_le_3_tokens": short,
        "long_ge_25_tokens": long,
        "contains_negation": negation,
        "contains_laughter": laughter,
        "sarcasm_phrase": sarcasm,
    }


def slice_rows(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, masks: dict) -> list[dict]:
    rows = []
    for name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            accuracy = None
            errors = 0
        else:
            accuracy = float(accuracy_score(y_true[mask], y_pred[mask]))
            errors = int(np.sum(y_true[mask] != y_pred[mask]))
        rows.append(
            {
                "pipeline": model_name,
                "slice": name,
                "n": n,
                "n_positive": int(np.sum(y_true[mask] == 1)) if n else 0,
                "n_negative": int(np.sum(y_true[mask] == 0)) if n else 0,
                "accuracy": accuracy,
                "errors": errors,
            }
        )
    return rows


def confident_errors(y_true, proba, original, cleaned, positive_label_mistakes: bool, k: int = 8):
    """High-confidence false positives or false negatives."""
    pred = (proba >= 0.5).astype(int)
    if positive_label_mistakes:
        # Predicted positive, truly negative.
        candidates = np.flatnonzero((y_true == 0) & (pred == 1))
        order = candidates[np.argsort(proba[candidates])[::-1]]
        kind = "false_positive"
    else:
        candidates = np.flatnonzero((y_true == 1) & (pred == 0))
        order = candidates[np.argsort(proba[candidates])]
        kind = "false_negative"
    rows = []
    for index in order[:k]:
        rows.append(
            {
                "type": kind,
                "true_label": "positive" if y_true[index] == 1 else "negative",
                "predicted_positive_probability": float(proba[index]),
                "original": original[index],
                "cleaned": cleaned[index],
            }
        )
    return rows


def disagreement_examples(y_true, pred_a, pred_b, proba_a, original, cleaned, k: int = 5):
    a_only = np.flatnonzero((pred_a == y_true) & (pred_b != y_true))
    b_only = np.flatnonzero((pred_b == y_true) & (pred_a != y_true))
    # Prefer confident corrections.
    a_only = a_only[np.argsort(np.abs(proba_a[a_only] - 0.5))[::-1]] if len(a_only) else a_only
    rows = {"a_correct_b_wrong": [], "b_correct_a_wrong": []}
    for index in a_only[:k]:
        rows["a_correct_b_wrong"].append(
            {
                "true_label": "positive" if y_true[index] == 1 else "negative",
                "original": original[index],
                "cleaned": cleaned[index],
            }
        )
    for index in b_only[:k]:
        rows["b_correct_a_wrong"].append(
            {
                "true_label": "positive" if y_true[index] == 1 else "negative",
                "original": original[index],
                "cleaned": cleaned[index],
            }
        )
    return {
        "a_correct_b_wrong_count": int(len(a_only)),
        "b_correct_a_wrong_count": int(len(b_only)),
        "examples": rows,
    }


def length_stats(texts: list[str]) -> dict:
    lengths = np.array([token_count(text) for text in texts])
    empty = int(np.sum(lengths == 0))
    return {
        "n": int(len(texts)),
        "empty_after_cleaning": empty,
        "mean_tokens": float(lengths.mean()) if len(lengths) else 0.0,
        "median_tokens": float(np.median(lengths)) if len(lengths) else 0.0,
        "p95_tokens": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
    }


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "pipeline",
        "accuracy",
        "macro_f1",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "precision_neg",
        "recall_neg",
        "f1_neg",
        "best_C",
        "val_macro_f1",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def save_slice_csv(path: Path, rows: list[dict]) -> None:
    fields = ["pipeline", "slice", "n", "n_positive", "n_negative", "accuracy", "errors"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_scores(path: Path, rows: list[dict]) -> None:
    names = [row["pipeline"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    macro_f1 = [row["macro_f1"] for row in rows]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    ax.bar(x + width / 2, macro_f1, width, label="Macro F1")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Test-set sentiment performance")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(path: Path, named_matrices: list[tuple[str, list]]) -> None:
    n = len(named_matrices)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, matrix) in zip(axes, named_matrices):
        data = np.array(matrix)
        image = ax.imshow(data, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], ["Pred neg", "Pred pos"])
        ax.set_yticks([0, 1], ["True neg", "True pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(data[i, j]), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    rng = np.random.default_rng(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else W2V_EPOCHS
    c_grid = (1.0,) if args.smoke else C_GRID
    per_class = 2000 if args.smoke else args.per_class

    print(f"Loading tweets from {args.zip}")
    train_pos = load_lines(args.zip, "train_Pos.en")
    train_neg = load_lines(args.zip, "train_Neg.en")
    test_pos_raw = load_lines(args.zip, "ttestPos.en")
    test_neg_raw = load_lines(args.zip, "ttestNeg.en")

    pos_idx = sample_indices(len(train_pos), per_class, rng)
    neg_idx = sample_indices(len(train_neg), per_class, rng)
    sample_raw = [train_pos[i] for i in pos_idx] + [train_neg[i] for i in neg_idx]
    sample_labels = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
    sample_clean = [clean_tweet(text) for text in sample_raw]

    test_raw = test_pos_raw + test_neg_raw
    test_labels = np.array([1] * len(test_pos_raw) + [0] * len(test_neg_raw))
    test_clean = [clean_tweet(text) for text in test_raw]

    order = rng.permutation(len(sample_clean))
    sample_raw = [sample_raw[i] for i in order]
    sample_clean = [sample_clean[i] for i in order]
    sample_labels = sample_labels[order]

    train_idx, val_idx = stratified_split(sample_labels, VAL_FRACTION, rng)
    stats = {
        "full_train_positive": len(train_pos),
        "full_train_negative": len(train_neg),
        "full_test_positive": len(test_pos_raw),
        "full_test_negative": len(test_neg_raw),
        "sample_per_class": int(per_class),
        "sample_total": int(len(sample_clean)),
        "train_fold": int(len(train_idx)),
        "validation_fold": int(len(val_idx)),
        "sample_length": length_stats(sample_clean),
        "test_length": length_stats(test_clean),
        "preprocessing": [
            "lowercase",
            "strip surrounding quotation marks",
            "remove URLs and @mentions",
            "keep hashtag words without the # character",
            "expand common contractions so negation is preserved",
            "remove punctuation and digits",
            "do not remove stopwords, because negation words carry sentiment",
        ],
        "vectorizer": {
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
        },
        "classifier": "L2 logistic regression (liblinear), C chosen on validation macro F1",
        "c_grid": list(c_grid),
        "embeddings": {
            "algorithm": "skip-gram Word2Vec",
            "dim": W2V_DIM,
            "window": W2V_WINDOW,
            "min_count": W2V_MIN_COUNT,
            "epochs": epochs,
            "pooling": "mean of in-vocabulary word vectors",
            "scaler": "StandardScaler fit on the training fold only",
        },
        "seed": SEED,
        "smoke": bool(args.smoke),
    }
    print(
        f"Sample {stats['sample_total']} tweets "
        f"({stats['train_fold']} train / {stats['validation_fold']} val); "
        f"test {len(test_clean)}"
    )

    pipelines = []
    predictions = {}
    probabilities = {}
    interpretations = {}

    lexicon_pred, ties = lexicon_predict(tokenise(test_clean))
    lexicon_metrics = classification_metrics(test_labels, lexicon_pred)
    lexicon_metrics.update(
        {
            "pipeline": "lexicon",
            "best_C": "",
            "val_macro_f1": "",
            "ties_predicted_positive": ties,
        }
    )
    pipelines.append(lexicon_metrics)
    predictions["lexicon"] = lexicon_pred
    probabilities["lexicon"] = lexicon_pred.astype(float)
    print(f"lexicon  test macro F1 {lexicon_metrics['macro_f1']:.4f}  ties {ties}")

    sparse_kinds = [
        ("bow", "Bag-of-words counts, unigrams and bigrams"),
        ("tf", "L2-normalised term frequency, IDF disabled"),
        ("idf", "Binary term presence multiplied by smoothed IDF"),
        ("tfidf", "Sublinear TF multiplied by smoothed IDF, then L2-normalised"),
    ]
    train_text = [sample_clean[i] for i in train_idx]
    val_text = [sample_clean[i] for i in val_idx]
    y_train = sample_labels[train_idx]
    y_val = sample_labels[val_idx]

    for kind, description in sparse_kinds:
        t0 = time.time()
        print(f"Tuning {kind} ...")
        best_c, val_score, _ = tune_sparse(kind, train_text, val_text, y_train, y_val, c_grid)
        pred, proba, features, extra = fit_sparse_final(
            kind, sample_clean, sample_labels, test_clean, best_c
        )
        metrics = classification_metrics(test_labels, pred)
        metrics.update(
            {
                "pipeline": kind,
                "description": description,
                "best_C": best_c,
                "val_macro_f1": val_score,
                **extra,
            }
        )
        pipelines.append(metrics)
        predictions[kind] = pred
        probabilities[kind] = proba
        interpretations[kind] = features
        print(
            f"{kind:6}  C={best_c:<3}  val F1 {val_score:.4f}  "
            f"test F1 {metrics['macro_f1']:.4f}  ({time.time() - t0:.1f}s)"
        )

    print("Tuning word2vec embeddings ...")
    t0 = time.time()
    train_sentences = tokenise(train_text)
    val_sentences = tokenise(val_text)
    all_sentences = tokenise(sample_clean)
    test_sentences = tokenise(test_clean)
    best_c, val_score, _ = tune_embeddings(
        train_sentences, val_sentences, y_train, y_val, epochs, c_grid
    )
    pred, proba, polarity, extra = fit_embeddings_final(
        all_sentences, sample_labels, test_sentences, best_c, epochs
    )
    metrics = classification_metrics(test_labels, pred)
    metrics.update(
        {
            "pipeline": "word2vec",
            "description": "Mean-pooled skip-gram Word2Vec vectors",
            "best_C": best_c,
            "val_macro_f1": val_score,
            **extra,
        }
    )
    pipelines.append(metrics)
    predictions["word2vec"] = pred
    probabilities["word2vec"] = proba
    interpretations["word2vec"] = polarity
    print(
        f"word2vec  C={best_c:<3}  val F1 {val_score:.4f}  "
        f"test F1 {metrics['macro_f1']:.4f}  ({time.time() - t0:.1f}s)"
    )

    masks = slice_masks(test_clean)
    slice_table = []
    for row in pipelines:
        name = row["pipeline"]
        slice_table.extend(slice_rows(name, test_labels, predictions[name], masks))

    best_name = max(
        (row for row in pipelines if row["pipeline"] != "lexicon"),
        key=lambda row: row["macro_f1"],
    )["pipeline"]
    errors = {
        "best_pipeline": best_name,
        "false_positives": confident_errors(
            test_labels, probabilities[best_name], test_raw, test_clean, True
        ),
        "false_negatives": confident_errors(
            test_labels, probabilities[best_name], test_raw, test_clean, False
        ),
        "tfidf_vs_word2vec": disagreement_examples(
            test_labels,
            predictions["tfidf"],
            predictions["word2vec"],
            probabilities["tfidf"],
            test_raw,
            test_clean,
        ),
        "bow_vs_tfidf": disagreement_examples(
            test_labels,
            predictions["bow"],
            predictions["tfidf"],
            probabilities["bow"],
            test_raw,
            test_clean,
        ),
    }

    save_metrics_csv(out / "metrics.csv", pipelines)
    save_slice_csv(out / "error_slices.csv", slice_table)
    plot_scores(out / "performance.png", pipelines)
    plot_confusion(
        out / "confusion_matrices.png",
        [(row["pipeline"], row["confusion_matrix"]) for row in pipelines],
    )
    payload = {
        "dataset": stats,
        "pipelines": pipelines,
        "interpretations": interpretations,
        "error_analysis": errors,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out} in {payload['elapsed_seconds']}s. Best pipeline: {best_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--per-class", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true", help="Small run used to check the pipeline.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

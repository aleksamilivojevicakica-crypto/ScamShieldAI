"""Train and evaluate the models on a reproducible split of UCI SMS Spam Collection."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from model import ARTIFACT, DATASET, MODEL_NAMES, TARGET_MODEL, group_key, make_models, normalize
from model import redact, short_message
from text_defense import restore_words, trained_vocabulary

SEED = 2026


def read_dataset(path):
    with ZipFile(path) as archive:
        raw = archive.read("SMSSpamCollection").decode("utf-8")
    messages = {}
    conflicting = set()
    rows = 0
    for line in raw.splitlines():
        if not line:
            continue
        label, text = line.split("\t", 1)
        key = normalize(text)
        target = {"ham": 0, "spam": 1}[label]
        rows += 1
        if key in messages and messages[key][1] != target:
            conflicting.add(key)
        messages.setdefault(key, (text, target))
    for key in conflicting:
        del messages[key]
    texts, labels = zip(*messages.values())
    return np.array(texts), np.array(labels, dtype=int), rows, len(conflicting)


def partition_validation(texts, labels, groups, trainval):
    folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    train_relative, validation_relative = next(
        folds.split(texts[trainval], labels[trainval], groups[trainval])
    )
    return trainval[train_relative], trainval[validation_relative]


def choose_threshold(labels, scores):
    """Pick the best validation F1 from a fixed threshold grid."""
    candidates = np.arange(0.05, 0.951, 0.025).round(3)
    return float(max(candidates, key=lambda t: (f1_score(labels, scores >= t), -abs(t - 0.5))))


def metrics(labels, predicted, scores):
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(labels), "spam": int(labels.sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "ap": float(average_precision_score(labels, scores)) if tp + fn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }



def evaluate_defense(pipeline, threshold, texts, labels):
    """Compare both scoring paths on identical held-out inputs."""
    vocabulary = trained_vocabulary(pipeline)
    word = re.compile(r"\b[A-Za-z]{5,}\b")

    def simulate_typing_error(message):
        def insert_dot(match):
            token = match.group()
            midpoint = len(token) // 2
            return token[:midpoint] + "." + token[midpoint:]

        return word.sub(insert_dot, message, count=2)

    results = {}
    for key, inputs in (("clean", texts), ("typing_errors", [simulate_typing_error(t) for t in texts])):
        restored = [restore_words(text, vocabulary) for text in inputs]
        results[key] = {}
        for version, messages in (("original", inputs), ("protected", restored)):
            scores = pipeline.predict_proba(messages)[:, 1]
            results[key][version] = metrics(labels, scores >= threshold, scores)
    return results


def error_examples(texts, labels, predicted, scores):
    misses = iter(np.flatnonzero((labels == 1) & ~predicted))
    alarms = iter(np.flatnonzero((labels == 0) & predicted))
    selected = []
    for stream in (misses, alarms, misses, alarms, misses):
        index = next(stream, None)
        if index is not None:
            selected.append(index)
    if len(selected) < 5:
        selected.extend(list(misses)[:5 - len(selected)])
        selected.extend(list(alarms)[:5 - len(selected)])
    return [
        {"text": redact(texts[i]), "label": "Spam" if labels[i] else "Ordinary",
         "prediction": "Spam" if predicted[i] else "Ordinary",
         "score": float(scores[i]), "short": short_message(texts[i])}
        for i in selected
    ]


def train(dataset):
    texts, labels, raw_count, conflicts = read_dataset(dataset)
    groups = np.array([group_key(text) or normalize(text) for text in texts])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer = iter(folds.split(texts, labels, groups))
    trainval, test_idx = next(outer)
    train_idx, val_idx = partition_validation(texts, labels, groups, trainval)
    test_text, test_y = texts[test_idx], labels[test_idx]
    bundle = {"sklearn_version": sklearn.__version__, "model": None}
    report = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "raw_count": raw_count, "unique_count": len(texts), "conflicting_keys": conflicts, "seed": SEED,
        "splits": {name: {"total": len(idx), "spam": int(labels[idx].sum())}
                   for name, idx in zip(("train", "validation", "test"), (train_idx, val_idx, test_idx))},
        "models": {},
    }

    audit_models = ("words_lr", TARGET_MODEL)
    audit_predictions = {name: np.zeros(len(texts), dtype=bool) for name in audit_models}

    def record(name, scores, threshold):
        predicted = scores >= threshold
        report["models"][name] = {
            "name": MODEL_NAMES[name], "threshold": threshold,
            "overall": metrics(test_y, predicted, scores),
        }
        if name == TARGET_MODEL:
            report["errors"] = error_examples(test_text, test_y, predicted, scores)

    record("always_ham", np.zeros(len(test_y)), 0.5)
    for name, pipeline in make_models().items():
        pipeline.fit(texts[train_idx], labels[train_idx])
        threshold = choose_threshold(labels[val_idx], pipeline.predict_proba(texts[val_idx])[:, 1])
        scores = pipeline.predict_proba(test_text)[:, 1]
        record(name, scores, threshold)
        if name in audit_models:
            audit_predictions[name][test_idx] = scores >= threshold
        if name == TARGET_MODEL:
            bundle["model"] = pipeline
            bundle["threshold"] = threshold
            report["defense_audit"] = evaluate_defense(pipeline, threshold, test_text, test_y)
    # Each row receives exactly one prediction from a model that did not see its group.
    # Inner validation selects the decision threshold within each outer fold.
    for trainval, outer_test in outer:
        inner_train, inner_val = partition_validation(texts, labels, groups, trainval)
        models = make_models()
        for name in audit_models:
            pipeline = models[name].fit(texts[inner_train], labels[inner_train])
            threshold = choose_threshold(labels[inner_val], pipeline.predict_proba(texts[inner_val])[:, 1])
            audit_predictions[name][outer_test] = pipeline.predict_proba(texts[outer_test])[:, 1] >= threshold

    short_all = np.array([short_message(text) for text in texts])
    report["length_audit"] = {}
    for name, predicted in audit_predictions.items():
        report["length_audit"][name] = {}
        for group, mask in (("short", short_all), ("long", ~short_all)):
            actual, alerts = labels[mask], predicted[mask]
            n_spam = int(actual.sum())
            report["length_audit"][name][group] = {
                "n": len(actual), "spam": n_spam, "tp": int(actual[alerts].sum()),
                "recall": float(recall_score(actual, alerts, zero_division=0)),
                "f1": float(f1_score(actual, alerts, zero_division=0)),
                "fp": int((actual[alerts] == 0).sum()),
            }
    bundle["report"] = report
    return bundle


def save_bundle(bundle, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete artifact with an atomic rename.
    with NamedTemporaryFile(dir=output.parent, suffix=".joblib", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET, help="Path to the original UCI ZIP.")
    parser.add_argument("--output", type=Path, default=ARTIFACT, help="Output model bundle.")
    parser.add_argument("--report", type=Path, help="Optional JSON results file.")
    args = parser.parse_args()
    bundle = train(args.dataset)
    save_bundle(bundle, args.output)
    report = bundle["report"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model to {args.output}")
    for row in report["models"].values():
        result = row["overall"]
        print(f"{row['name']}: precision {result['precision']:.3f}, "
              f"recall {result['recall']:.3f}, F1 {result['f1']:.3f}, "
              f"AP {result['ap']:.3f}")


if __name__ == "__main__":
    main()

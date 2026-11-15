"""Text classification and explanations for ScamShield AI."""

import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "sms-spam-collection.zip"
ARTIFACT = ROOT / "artifacts" / "experiment.joblib"
MAX_MESSAGE_LENGTH = 1500
SHORT_WORD_LIMIT = 10
TARGET_MODEL = "hybrid"
MODEL_NAMES = {
    "always_ham": "Always predict ordinary",
    "words_nb": "Word TF-IDF + Naive Bayes",
    "words_lr": "Word TF-IDF + logistic regression",
    "hybrid": "Word and character TF-IDF + logistic regression",
}
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
WORD_PATTERN = re.compile(r"\b\w+\b")


def normalize(text):
    """Canonicalize casing, equivalent Unicode forms and whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_key(text):
    """Keep identical and some templated SMS together when splitting."""
    masked = URL_PATTERN.sub(" linktoken ", normalize(text))
    masked = EMAIL_PATTERN.sub(" emailtoken ", masked)
    masked = re.sub(r"\d+", "numtoken", masked)
    return " ".join(WORD_PATTERN.findall(masked))


def redact(text):
    """Hide contact details and links before showing corpus error examples."""
    text = URL_PATTERN.sub("[link]", text)
    text = EMAIL_PATTERN.sub("[email]", text)
    return re.sub(r"\d(?:[\d().+ -]*\d)?", "[number]", text)


def short_message(text):
    return len(WORD_PATTERN.findall(normalize(text))) <= SHORT_WORD_LIMIT


def make_models():
    word_settings = dict(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "words_nb": Pipeline([
            ("features", TfidfVectorizer(**word_settings)),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "words_lr": Pipeline([
            ("features", TfidfVectorizer(**word_settings)),
            ("classifier", LogisticRegression(C=5, max_iter=1000)),
        ]),
        "hybrid": Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(**word_settings)),
                ("chars", TfidfVectorizer(
                    preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5),
                    min_df=3, max_features=20000, sublinear_tf=True,
                )),
            ])),
            ("classifier", LogisticRegression(C=5, max_iter=1000)),
        ]),
    }


def analyze(pipeline, threshold, message):
    """Explain the largest active features of the logistic regression score."""
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    row = features.transform([message])
    score = float(classifier.predict_proba(row)[0, 1])
    impacts = row.data * classifier.coef_[0, row.indices]
    influential = np.argsort(-np.abs(impacts), kind="stable")[:8]

    # Word features come first, followed by the character vectorizer's columns.
    word_vectorizer, char_vectorizer = [vectorizer for _, vectorizer in features.transformer_list]
    active = {int(row.indices[position]) for position in influential if impacts[position] != 0}
    word_count = len(word_vectorizer.vocabulary_)
    names = {
        index: (term, "Word")
        for term, index in word_vectorizer.vocabulary_.items() if index in active
    }
    names.update({
        index + word_count: (term.replace(" ", "␣"), "Fragment")
        for term, index in char_vectorizer.vocabulary_.items() if index + word_count in active
    })

    reasons = [
        {"term": names[int(row.indices[position])][0],
         "kind": names[int(row.indices[position])][1],
         "impact": float(impacts[position])}
        for position in influential if impacts[position] != 0
    ]
    return {"score": score, "threshold": threshold, "flagged": score >= threshold, "reasons": reasons}

"""Run the ScamShield AI local dashboard."""

import argparse

import joblib
import sklearn
from flask import Flask, jsonify, render_template, request

from model import ARTIFACT, DATASET, MAX_MESSAGE_LENGTH, SHORT_WORD_LIMIT, analyze, normalize
from text_defense import restore_words, trained_vocabulary
from phishing_signals import assess_phishing

EXAMPLES = (
    ("Ordinary message", "Can we meet outside the library after class?"),
    ("Promotional message", "Congratulations! Claim your free cash prize now. Reply WIN to enter."),
    ("Short reminder", "Your ticket is ready. See you at 7!"),
)


def create_app(bundle):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32_768
    app.add_template_filter(lambda value: f"{value:.1%}", "pct")
    model = bundle["model"]
    threshold = bundle["threshold"]
    vocabulary = trained_vocabulary(model)

    @app.get("/")
    def index():
        return render_template(
            "index.html", examples=EXAMPLES, report=bundle["report"],
            limit=MAX_MESSAGE_LENGTH, short_limit=SHORT_WORD_LIMIT,
        )

    @app.post("/analyze")
    def analyze_sms():
        data = request.get_json(silent=True)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, str) or not normalize(message):
            return jsonify(error="Enter a message to analyze."), 400
        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify(error=f"Use at most {MAX_MESSAGE_LENGTH} characters."), 400
        restored = restore_words(message, vocabulary)
        result = analyze(model, threshold, restored)
        result["phishing"] = assess_phishing(restored)
        result["ml_flagged"] = result["flagged"]
        result["flagged"] = result["ml_flagged"] or result["phishing"]["flagged"]
        if restored != message:
            result["original_score"] = float(model.predict_proba([message])[0, 1])
            result["restored_text"] = restored
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="The request is too large."), 413

    return app


def load_bundle():
    if not ARTIFACT.exists():
        from train_model import save_bundle, train

        print("First launch: training ScamShield AI on the included SMS dataset...", flush=True)
        bundle = train(DATASET)
        save_bundle(bundle, ARTIFACT)
        return bundle
    bundle = joblib.load(ARTIFACT)
    if bundle["sklearn_version"] != sklearn.__version__:
        raise SystemExit("Reinstall requirements.txt and run python train_model.py again.")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_app(load_bundle()).run(host="127.0.0.1", port=args.port)

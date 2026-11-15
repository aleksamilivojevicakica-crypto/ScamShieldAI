"""Recover recognizable split words before classifying an SMS."""

import re

# Require two ASCII letters on each side and keep URLs and addresses intact.
DOTTED_WORD = re.compile(r"(?<![\w.@/])([A-Za-z]{2,})\.([A-Za-z]{2,})(?![\w.@/])")
CONTACT = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)


def restore_words(message, vocabulary):
    """Join a dotted word when its complete spelling appears in training."""

    def restore(match):
        joined = "".join(match.groups())
        left, right = map(str.lower, match.groups())
        # Preserve punctuation between two recognizable standalone words.
        if joined.lower() in vocabulary and not (left in vocabulary and right in vocabulary):
            return joined
        return match.group()

    output = []
    start = 0
    for match in CONTACT.finditer(message):
        output.append(DOTTED_WORD.sub(restore, message[start:match.start()]))
        output.append(match.group())
        start = match.end()
    output.append(DOTTED_WORD.sub(restore, message[start:]))
    return "".join(output)


def trained_vocabulary(pipeline):
    features = pipeline.named_steps["features"]
    return features.transformer_list[0][1].vocabulary_

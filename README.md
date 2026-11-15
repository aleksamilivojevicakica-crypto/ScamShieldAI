# ScamShield AI

**Explainable SMS spam detection with financial phishing warnings**

**Veritas AI International High School Hackathon 2026** · **AI + Finance**  
**Participant:** Aleksa Milivojevic

> **Prototype notice:** ScamShield AI is an educational prototype, not financial advice. A spam score does not establish whether a sender or link is safe.

---

## The problem

A teenager with a first debit card receives an unexpected text about a payment, prize, or account restriction. The message may feel urgent, but deciding whether to trust it is difficult. **ScamShield AI offers a second look before someone replies, clicks, or shares information.**

## What ScamShield AI does

| Feature | Purpose |
|---|---|
| **SMS spam classifier** | Scores how closely an English SMS resembles messages in a historical spam dataset. |
| **Explainable predictions** | Highlights the words and character fragments contributing to the model's score. |
| **Financial phishing warning** | Independently flags combinations of banking/payment context, urgency, restrictions, and requests for action. |
| **Text protection** | Attempts to restore words split by stray periods, then compares scores before and after restoration. |
| **Evaluation dashboard** | Displays held-out test results, robustness checks, and examples of model errors. |

### Why a separate phishing warning?

A bank-suspension example received a **38.8% spam score** from the general SMS classifier. This motivated a separate **rule-based financial phishing warning**. It checks for combinations of financial context, restrictions or urgency, and requests for verification, credentials, or action.

The interface distinguishes **`Model spam score`** from **`Financial phishing warning`** and displays matched signals. **The rule-based warning does not change the ML probability.** It is not a trained or validated financial-fraud classifier; false positives and false negatives remain possible, including for legitimate security notifications.

## How it works

1. **Data:** Read 5,574 records from the public UCI SMS Spam Collection. After case, Unicode, and whitespace normalization, 5,159 unique messages remained.
2. **Leakage reduction:** Keep messages with the same normalized template together by masking URLs, email addresses, and numbers when grouping. This reduces duplicate/template leakage but cannot exclude every near-duplicate.
3. **Train/validate/test:** Use **3,095 / 1,032 / 1,032** messages. Choose alert thresholds by maximizing validation F1 over a fixed grid.
4. **Models:** Compare an always-ordinary baseline, word TF-IDF + Multinomial Naive Bayes, word TF-IDF + logistic regression, and word-and-character TF-IDF + logistic regression.
5. **Explanations:** Multiply active TF-IDF feature values by the deployed logistic model's learned coefficients; positive contributions raise the spam score.
6. **Additional checks:** Apply independent financial phishing heuristics and optionally restore words broken by stray periods using vocabulary learned from training SMS.

## Results

**Held-out historical SMS test set: 1,032 messages**

| Model | Precision | Recall | F1 | Average precision |
|:--|--:|--:|--:|--:|
| Always ordinary | Undefined | 0.0% | 0.0% | 12.4% |
| Word TF-IDF + Naive Bayes | 95.0% | 88.3% | 91.5% | 95.0% |
| Word TF-IDF + logistic regression | 98.2% | 86.7% | 92.1% | 96.4% |
| **Word + character TF-IDF + logistic regression** | **100.0%** | **88.3%** | **93.8%** | **98.0%** |

The deployed hybrid classifier identified **113 of 128 spam messages** with **0 false alerts among 904 ordinary messages** in this test split. One missed spam message consisted largely of a number, hours, and a cost; it may have lacked distinctive text features.

> **Interpretation:** These figures describe spam classification on a historical corpus, **not** accuracy against modern financial scams. The dataset includes advertisements and other unwanted messages.

### Short-message audit

A five-fold audit gave each message a prediction from a model trained on other groups; each fold selected its threshold on a separate validation partition.

- **Short SMS (up to 10 words):** both logistic models found **17/23** spam messages; adding character fragments reduced false alerts from **21 to 10** across **2,161** messages.
- **Longer SMS:** detected spam increased from **572/619 to 582/619**, with **10** false alerts for each model across **2,998** messages.

The short-spam sample is small, so these findings are exploratory.

### Text protection: controlled corruption experiment

Each scoring path received the same 1,032 original SMS and 1,032 copies with a fixed synthetic typing error. The rule inserted a stray period into at most two long words independently of labels and predictions. The trained classifier, vocabulary, and validation-selected threshold remained fixed.

| Input | Scoring path | Spam detected | Recall | False alerts | F1 |
|:--|:--|--:|--:|--:|--:|
| Original SMS | Original scorer | 113/128 | 88.3% | 0 | 93.8% |
| Original SMS | With word restoration | 113/128 | 88.3% | 0 | 93.8% |
| Synthetic typing error | Original scorer | 111/128 | 86.7% | 0 | 92.9% |
| **Synthetic typing error** | **With word restoration** | **112/128** | **87.5%** | **0** | **93.3%** |

Restoration recovered **one additional spam message** in this synthetic condition. The test addresses one specific kind of text distortion, not general adversarial robustness. The restoration preserves complete URLs and email addresses and leaves a period between two recognized words.

## Run locally

**Requirements:** Python and the dependencies in `requirements.txt`.

Open the folder containing `main.py`, then run in a Windows PowerShell or PyCharm terminal:

```powershell
py -m pip install -r requirements.txt
py main.py
```

On macOS/Linux, use `python3` instead of `py`. On first launch, the app trains from the included dataset and caches the model locally.

Open **http://127.0.0.1:7860**. If that port is occupied:

```powershell
py main.py --port 7861
```

To reproduce the report:

```powershell
py train_model.py --report results.json
```

Restart the app afterward. Stop the server with **Ctrl+C**.

### Suggested demo walkthrough

1. In **Message analyzer**, analyze the built-in promotional example and inspect its score and feature explanations.
2. Analyze a suspicious banking message and compare the independent financial phishing warning with the model spam score.
3. Open **Text protection** to view the held-out experiment comparing original and restored scoring.
4. Show **Robustness** and **Model performance** to explain both successes and limitations.

## Project files

| Path | Role |
|:--|:--|
| `main.py` | Flask server and analysis endpoints |
| `model.py` | Classifiers, scoring, and explanations |
| `train_model.py` | Dataset preparation, training, and evaluations |
| `text_defense.py` | Restoration of period-split words |
| `phishing_signals.py` | Independent financial phishing heuristics |
| `templates/`, `static/` | Web interface and assets |
| `data/` | Original dataset and accompanying materials |
| `results.json` | Recorded evaluation results |
| `requirements.txt` | Python dependencies |

## Limitations and next steps

- Evaluate on newer, verified English SMS, especially short messages.
- Build a separate labeled financial-scam evaluation set and measure false positives and false negatives for phishing heuristics.
- Test additional distortions and adversarial wording beyond the single synthetic period-insertion rule.
- Do not treat model confidence as proof of sender identity, link safety, or financial legitimacy.

## Credits and licenses

- **Dataset:** Almeida, T. and Hidalgo, J. (2011), [SMS Spam Collection](https://doi.org/10.24432/C5CC84), UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original UCI ZIP and readme are included in `data/`. Training normalizes and deduplicates messages; the UI masks contact details and numbers in error examples.
- **AI assistance:** [ChatGPT](https://chatgpt.com/) was used for project planning, generating Python and web-app code, testing, and drafting the write-up.
- **Software and assets:** [scikit-learn](https://scikit-learn.org/stable/), [Flask](https://flask.palletsprojects.com/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [joblib](https://joblib.readthedocs.io/), [threadpoolctl](https://github.com/joblib/threadpoolctl), and [DM Sans](https://fonts.google.com/specimen/DM+Sans) (SIL Open Font License, included with the project). All classifiers are trained locally on the included dataset.

---

**ScamShield AI** · *Think before you trust.*

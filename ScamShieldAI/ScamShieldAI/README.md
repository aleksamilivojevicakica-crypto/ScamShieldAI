# ScamShield AI

**Track:** AI + Finance · Veritas AI International High School Hackathon 2026  
**Team:** [Add participant name(s) before submission]

This project is a prototype and is not financial advice.

## The problem

A teen with a first debit card receives an unexpected SMS about a payment, prize or account. The app provides a second look before they reply or share information.

## What we built

A local web app that scores an English SMS and explains the words or fragments that influenced the prediction. It can restore words split by a stray period and show scores before and after restoration. The score measures resemblance to the historical spam class. Sender identity and link safety require independent verification.

## What we added

The Text protection feature restores a word split by a stray period when the joined spelling is in the vocabulary learned from training SMS. It keeps a period between two recognized words and preserves full URLs and email addresses. The same model and validation-selected threshold are used before and after restoration.

We started with the Scam Text Filter brief and asked whether short messages are harder to classify, and whether character fragments help. We used a five-fold audit: each message receives one prediction from a model trained on other groups. Each fold chooses its threshold on a separate validation partition.

Among short SMS (up to 10 words), both logistic models caught **17 of 23** spam messages. Adding fragments reduced false alerts from **21 to 10** across the 2,161 short messages. Among the 2,998 longer messages, recall rose from **572/619 to 582/619**, with 10 false alerts in each model. The short spam group is small, so its results are exploratory.

## How it works

We read 5,574 public UCI SMS records. After normalizing case, Unicode and whitespace, 5,159 unique messages remained. Our split keeps messages with the same normalized template together by masking URLs, email addresses, and numbers. This reduces leakage from duplicates and some templated messages. It cannot rule out every near-duplicate. The main split had 3,095 training, 1,032 validation and 1,032 test records. Model thresholds maximize validation F1 on a fixed grid.

TF-IDF gives features higher weights when they are prominent in a message and less common in the corpus. We compare the ordinary-only baseline with word TF-IDF using Multinomial Naive Bayes and logistic regression. The third ML model adds character fragments to the same word representation. The app explains the deployed logistic model by multiplying the active TF-IDF value by its learned coefficient. Positive values move the spam score up.

## Results

On the fixed 1,032-message test split:

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| Always ordinary | Undefined | 0.0% | 0.0% | 12.4% |
| Word TF-IDF + Naive Bayes | 95.0% | 88.3% | 91.5% | 95.0% |
| Word TF-IDF + logistic regression | 98.2% | 86.7% | 92.1% | 96.4% |
| Word and character TF-IDF + logistic regression | 100.0% | 88.3% | 93.8% | 98.0% |

The hybrid classifier on original SMS flagged 113 of 128 spam messages and none of the 904 ordinary messages in this split. One missed spam message contained mostly a number, hours and a cost. A plausible reason is that the model had few distinctive text features to use. The Robustness tab shows five labeled errors with contact details masked. High precision on this historical corpus does not guarantee the same results on recent SMS or actual financial scams. Spam includes advertising and other unwanted messages.

### Text protection on the same test split

Both scoring paths received the same 1,032 original SMS and the same 1,032 copies with a fixed synthetic typing error. The error inserts a stray period into at most two long words using a fixed rule, independently of the score and label. The classifier was fitted on the training split. Its vocabulary and alert threshold were fixed before scoring the test data.

| Input | Scoring path | Spam detected | Recall | False alerts | F1 |
|---|---|---:|---:|---:|---:|
| Original SMS | Original scorer | 113/128 | 88.3% | 0 | 93.8% |
| Original SMS | With word restoration | 113/128 | 88.3% | 0 | 93.8% |
| Synthetic typing error | Original scorer | 111/128 | 86.7% | 0 | 92.9% |
| Synthetic typing error | With word restoration | 112/128 | 87.5% | 0 | 93.3% |

The restoration recovered one additional spam message in the synthetic condition. This small result measures a single kind of corruption in a historical SMS dataset.

## What we'd do next

Evaluate on newer English SMS, especially short messages with verified labels, and measure performance on financial scams as a separate class.

## Sources

- Almeida, T. and Hidalgo, J. (2011). [SMS Spam Collection](https://doi.org/10.24432/C5CC84). UCI Machine Learning Repository. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original UCI ZIP and its readme are included in `data/`. Training normalizes and deduplicates messages, and the UI masks contact details and numbers in error examples.
- [ChatGPT](https://chatgpt.com/) was used for project planning, generating the Python and web app code, testing, and drafting the write-up.
- [scikit-learn](https://scikit-learn.org/stable/), [Flask](https://flask.palletsprojects.com/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [joblib](https://joblib.readthedocs.io/), [threadpoolctl](https://github.com/joblib/threadpoolctl), and [DM Sans](https://fonts.google.com/specimen/DM+Sans) under the included SIL Open Font License. All classifiers are trained locally on the included dataset.

## Run in PyCharm

Open the extracted `ScamShieldAI` folder containing `main.py`. On Windows, run this from the PyCharm terminal:

```powershell
py main.py
```

On a fresh computer, install the dependencies first with `py -m pip install -r requirements.txt`. On macOS or Linux, use `python3` in place of `py`.

The first launch trains from the included dataset and caches the model locally. Once the terminal prints its address, open [http://127.0.0.1:7860](http://127.0.0.1:7860). Use `py main.py --port 7861` if port 7860 is occupied. Stop the app with Ctrl+C. To reproduce the full report, run `py train_model.py --report results.json`, then restart the app.

For a short demo, open **Message analyzer** and analyze the built-in promotional example. Then open **Text protection**, load its sample and analyze it to show the restored word. Return to **Text protection** to show the test comparison.

`model.py` defines the classifiers and explanations. `text_defense.py` restores split words. `train_model.py` prepares the data and runs the comparisons. `main.py`, `templates/` and `static/` serve the app. `results.json` records the included experiment.

## Financial phishing signals (additional prototype feature)

The analyzer now includes a **separate, rule-based phishing warning** for English
financial messages combining account/payment context, restriction or urgency,
and a request for verification, credentials, or action. The warning **does not
change the trained model's probability**. The UI explicitly distinguishes
`Model spam score` from `Financial phishing warning` and lists matched signals.

This feature was added after noticing a bank-suspension phishing example scored
38.8% by the general SMS spam classifier. It is a heuristic demonstration,
**not a trained or validated financial-fraud classifier**. False positives and
false negatives are possible; legitimate security notices can contain similar
language. The historical SMS dataset performance tables measure the original
ML model only and must not be presented as validation of these new rules.

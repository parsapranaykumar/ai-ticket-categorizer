# Auto Email / Support Ticket Categorizer

## Overview

This project is a lightweight NLP classification system that reads an incoming
support email/ticket (subject + body) and automatically predicts which
department should handle it — **Billing**, **Technical**, **HR**, or
**General**. It uses classic, easy-to-understand machine learning
(TF-IDF + Multinomial Naive Bayes) rather than heavy embeddings or LLMs,
making it fast to train, easy to reason about, and simple to deploy.

## Problem Statement

Support teams often waste time manually reading and forwarding tickets to the
right department. Automatic ticket classification speeds up first-response
time by instantly routing a ticket to Billing, Technical, HR, or General
support, while flagging urgent tickets and low-confidence predictions for a
human to review — reducing manual triage effort and improving response times.

## Categories

- Billing
- Technical
- HR
- General

## Technologies

- Python
- Pandas / NumPy
- Scikit-learn (TF-IDF, Multinomial Naive Bayes, Logistic Regression)
- Matplotlib / Seaborn (confusion matrix visualization)
- Streamlit (optional interactive demo)

## ML Pipeline

```
Dataset (92 tickets, 4 categories)
        ↓
Preprocessing (lowercase, remove punctuation, clean whitespace, combine subject+body)
        ↓
Train/Test Split (80/20, stratified, random_state=42)
        ↓
TF-IDF Vectorization (fit on train only)
        ↓
Multinomial Naive Bayes (+ Logistic Regression comparison)
        ↓
Evaluation (accuracy, precision, recall, F1, confusion matrix)
        ↓
Real-time prediction on new, unseen tickets (with confidence + priority)
```

## Model Evaluation (actual results)

The dataset contains 92 tickets, split 80/20 (73 train / 19 test), stratified
by category, with `random_state=42` for reproducibility.

### Multinomial Naive Bayes — Test Accuracy: **78.95%**

```
              precision    recall  f1-score   support

     Billing       0.71      1.00      0.83         5
     General       0.67      1.00      0.80         4
          HR       1.00      0.40      0.57         5
   Technical       1.00      0.80      0.89         5

    accuracy                           0.79        19
   macro avg       0.85      0.80      0.77        19
weighted avg       0.85      0.79      0.77        19
```

**Confusion Matrix** (rows = actual, cols = predicted):

|             | Billing | General | HR  | Technical |
|-------------|:-------:|:-------:|:---:|:---------:|
| **Billing**    | 5       | 0       | 0   | 0         |
| **General**    | 0       | 4       | 0   | 0         |
| **HR**         | 2       | 1       | 2   | 0         |
| **Technical**  | 0       | 1       | 0   | 4         |

### Logistic Regression (comparison) — Test Accuracy: **89.47%**

```
              precision    recall  f1-score   support

     Billing       0.83      1.00      0.91         5
     General       1.00      1.00      1.00         4
          HR       0.80      0.80      0.80         5
   Technical       1.00      0.80      0.89         5

    accuracy                           0.89        19
   macro avg       0.91      0.90      0.90        19
weighted avg       0.90      0.89      0.89        19
```

### What the metrics mean here

- **Precision** (e.g. Billing = 0.71 for Naive Bayes) — of all tickets the
  model predicted as Billing, 71% were actually Billing. Some HR tickets are
  being mistaken for Billing, pulling this down.
- **Recall** (e.g. Billing = 1.00) — the model successfully catches *all*
  actual Billing tickets in the test set; none slip through to another queue.
- **F1-score** balances the two, useful for comparing categories directly.

### Where the model struggles

**HR is the weakest category** for Naive Bayes (recall 0.40 — only 2 of 5 HR
test tickets were correctly identified; 2 were misrouted to Billing and 1 to
General). This happens because several HR tickets in this small dummy dataset
share generic phrasing ("last month", "download", "request") with Billing and
General tickets, so the model has fewer uniquely HR-specific words to rely
on. Logistic Regression handles this confusion noticeably better (HR recall
0.80), which is why it's included as a comparison. With a larger, more
diverse real dataset, this confusion would be expected to shrink further.

### Predictions on 6 new, unseen tickets

| Ticket | Prediction | Confidence | Priority | Final Decision |
|---|---|---|---|---|
| "Double charge on my card" / "My credit card was charged twice..." | Billing | 61.3% | NORMAL | **Billing** |
| "Server is down" / "Our production server is down and the API is not working..." | Technical | 31.4% | URGENT | **Needs Human Review** |
| "Payslip download issue" / "I am unable to download my payslip..." | Billing | 32.7% | URGENT | **Needs Human Review** |
| "General question about pricing" / "I would like to know more about pricing plans..." | General | 35.1% | NORMAL | **Needs Human Review** |
| "Password reset link broken" / "The reset password link says it expired..." | Technical | 38.8% | NORMAL | **Needs Human Review** |
| "Question about office holidays" / "list of official holidays for this year?" | HR | 31.0% | NORMAL | **Needs Human Review** |

Note the "Payslip download issue" ticket is genuinely misclassified as
Billing by the Naive Bayes model — but because its confidence (32.7%) is
below the 60% threshold, it is correctly routed to **Needs Human Review**
instead of being silently misrouted. This is exactly the kind of case the
threshold is designed to catch.

## Edge Cases: Low-Confidence Handling

`predict_ticket()` uses `model.predict_proba()` to get the probability of
each category and takes the highest one as the **confidence score**. This
confidence is the classifier's *estimated* probability, not a guarantee of
correctness.

If the top confidence is **below 60%**, the ticket is **not** auto-assigned.
Instead, it's returned as `"Needs Human Review"`. This matters in a real
support-routing system because an overconfident wrong classification can
delay a customer's ticket far more than a short manual triage step would —
it's safer to have a human quickly look at an ambiguous ticket than to
silently misroute it.

A separate, simple **rule-based priority tag** (`URGENT` / `NORMAL`) is also
applied based on keywords like "urgent", "down", "not working", "critical" —
kept intentionally independent from the ML category prediction, since urgency
and department are different concerns.

## Project Structure

```
ticket-categorizer/
│
├── data/
│   └── tickets.csv
│
├── notebook/
│   └── ticket_categorizer.ipynb
│
├── ticket_categorizer.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script** (trains the model, prints evaluation, tests unseen tickets)
   ```bash
   python ticket_categorizer.py
   ```
   Or open and run `notebook/ticket_categorizer.ipynb` in Jupyter for the
   same pipeline with inline explanations and plots.

4. **Run the Streamlit demo** (optional)
   ```bash
   streamlit run app.py
   ```
   Then open the local URL Streamlit prints (typically `http://localhost:8501`)
   and enter a ticket subject/body to see a live prediction, confidence, and
   priority.

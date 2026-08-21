"""
Auto Email / Support Ticket Categorizer
-----------------------------------------
A lightweight NLP classification system that reads an incoming support
email/ticket and predicts which department should handle it:
Billing, Technical, HR, or General.

Pipeline:
    Raw text -> Preprocessing -> TF-IDF -> Multinomial Naive Bayes -> Category

Run this script directly to train the model, print evaluation metrics,
and test it on unseen tickets:

    python ticket_categorizer.py
"""

import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42  # fixed seed so results are reproducible
CONFIDENCE_THRESHOLD = 0.60  # below this, a ticket is flagged for human review

URGENT_KEYWORDS = ["urgent", "emergency", "down", "not working", "critical", "asap"]


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Basic text cleaning before vectorization.

    Why these steps:
    - Lowercasing: so "Invoice" and "invoice" are treated as the same word,
      otherwise the vectorizer would learn them as two different features.
    - Removing punctuation: punctuation carries little classification signal
      for this task and just adds noise/extra vocabulary.
    - Collapsing extra whitespace: keeps the token stream clean after
      punctuation removal.
    We deliberately do NOT remove stopwords or apply stemming here, because
    with such a small dataset aggressive preprocessing can strip away useful
    signal (e.g. "not working" vs "working") and TF-IDF already down-weights
    very common words on its own.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    """Load the ticket dataset, handle missing values, and build the
    combined text field used for training."""
    df = pd.read_csv(path)

    # Handle missing values: an empty subject/body shouldn't crash the
    # pipeline, so we fill missing text with an empty string.
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combine subject and body into a single text field. The subject often
    # contains a strong, concise signal (e.g. "Refund still pending"), so we
    # keep it together with the body rather than discarding it.
    df["text"] = df["subject"] + " " + df["body"]
    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# 6. Priority tagging (simple rule-based, kept separate from the ML model)
# ---------------------------------------------------------------------------
def get_priority(raw_text: str) -> str:
    """Rule-based priority tag based on urgent keywords in the raw ticket
    text. This is intentionally kept separate from the ML classifier -
    urgency and department are two different concerns."""
    text = raw_text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in text:
            return "URGENT"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Main training / evaluation flow
# ---------------------------------------------------------------------------
def main():
    df = load_dataset("data/tickets.csv")
    print(f"Loaded {len(df)} tickets across {df['category'].nunique()} categories.\n")
    print(df["category"].value_counts(), "\n")

    X = df["clean_text"]
    y = df["category"]

    # 80/20 train/test split, stratified so each category is represented
    # proportionally in both the training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    # ------------------------------------------------------------------
    # Feature extraction: TF-IDF
    # ------------------------------------------------------------------
    # Raw text can't be fed directly into scikit-learn classifiers - they
    # need numeric input. TF-IDF (Term Frequency - Inverse Document
    # Frequency) converts each ticket into a vector where each dimension is
    # a word, weighted by how important that word is to THIS ticket
    # relative to the whole dataset. Words that appear in almost every
    # ticket (e.g. "the", "please") get a low weight, while words that are
    # distinctive of a category (e.g. "invoice", "crashes", "payslip") get
    # a higher weight. This makes TF-IDF well suited for short, keyword-
    # driven text like support tickets.
    #
    # IMPORTANT: the vectorizer is fit ONLY on the training data to avoid
    # data leakage, then used to transform the test data.
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ------------------------------------------------------------------
    # Model: Multinomial Naive Bayes
    # ------------------------------------------------------------------
    # Multinomial Naive Bayes is a strong baseline for text classification
    # because: (1) it works well with sparse, high-dimensional TF-IDF
    # features, (2) it trains fast and needs very little data compared to
    # more complex models, and (3) its independence assumption between
    # words, while technically "naive", tends to work fine in practice for
    # bag-of-words style text problems like this one.
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("=" * 60)
    print("MULTINOMIAL NAIVE BAYES - EVALUATION")
    print("=" * 60)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df, "\n")

    # ------------------------------------------------------------------
    # Optional comparison: Logistic Regression
    # ------------------------------------------------------------------
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_pred)
    print("=" * 60)
    print("LOGISTIC REGRESSION - COMPARISON")
    print("=" * 60)
    print(f"Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_pred, zero_division=0))

    # ------------------------------------------------------------------
    # Real-time prediction function
    # ------------------------------------------------------------------
    def predict_ticket(subject: str, body: str) -> dict:
        """
        Predict the department for a new, unseen support ticket.

        Steps: combine subject+body -> clean text -> TF-IDF transform ->
        predict category -> compute confidence -> apply human-review
        threshold -> tag priority -> return final structured result.
        """
        raw_text = f"{subject} {body}"
        cleaned = clean_text(raw_text)
        vec = vectorizer.transform([cleaned])

        probabilities = model.predict_proba(vec)[0]
        predicted_category = model.classes_[probabilities.argmax()]
        confidence = float(probabilities.max())

        # Confidence represents the model's estimated probability for its
        # top choice - it is a statistical estimate, not a guarantee of
        # correctness, so it should not be treated as absolute certainty.
        if confidence < CONFIDENCE_THRESHOLD:
            final_decision = "Needs Human Review"
        else:
            final_decision = predicted_category

        priority = get_priority(raw_text)

        return {
            "category": predicted_category,
            "confidence": round(confidence, 4),
            "priority": priority,
            "final_decision": final_decision,
        }

    # ------------------------------------------------------------------
    # 8. Test with new, unseen tickets
    # ------------------------------------------------------------------
    new_tickets = [
        ("Double charge on my card", "My credit card was charged twice for this month's subscription, please refund."),
        ("Server is down", "Our production server is down and the API is not working at all, this is critical."),
        ("Payslip download issue", "I am unable to download my payslip from the HR portal for last month."),
        ("General question about pricing", "Hi, I would like to know more about your pricing plans and what they include."),
        ("Password reset link broken", "I clicked the reset password link but it says the link has expired."),
        ("Question about office holidays", "Can you tell me the list of official holidays for this year?"),
    ]

    print("=" * 60)
    print("PREDICTIONS ON NEW, UNSEEN TICKETS")
    print("=" * 60)
    for subject, body in new_tickets:
        result = predict_ticket(subject, body)
        print(f"\nTicket Subject: {subject}")
        print(f"Ticket Body:    {body}")
        print(f"Prediction:     {result['category']}")
        print(f"Confidence:     {result['confidence'] * 100:.1f}%")
        print(f"Priority:       {result['priority']}")
        print(f"Final Decision: {result['final_decision']}")

    return {
        "vectorizer": vectorizer,
        "model": model,
        "lr_model": lr_model,
        "predict_ticket": predict_ticket,
        "nb_accuracy": acc,
        "lr_accuracy": lr_acc,
    }


if __name__ == "__main__":
    main()

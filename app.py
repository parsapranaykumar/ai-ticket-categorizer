"""
Streamlit demo for the Auto Email / Support Ticket Categorizer.

Run with:
    streamlit run app.py
"""

import streamlit as st

from ticket_categorizer import main

st.set_page_config(page_title="Support Ticket Categorizer", page_icon="🎫")


@st.cache_resource
def load_pipeline():
    """Train the model once and cache it for the app's lifetime."""
    return main()


st.title("🎫 Auto Support Ticket Categorizer")
st.write(
    "Enter a ticket subject and body below. The model predicts which "
    "department should handle it, along with a confidence score and priority."
)

with st.spinner("Training model..."):
    pipeline = load_pipeline()

predict_ticket = pipeline["predict_ticket"]

st.info(
    f"Naive Bayes test accuracy: **{pipeline['nb_accuracy']:.1%}** | "
    f"Logistic Regression test accuracy: **{pipeline['lr_accuracy']:.1%}**"
)

subject = st.text_input("Ticket Subject", placeholder="e.g. Refund not received")
body = st.text_area(
    "Ticket Body",
    placeholder="e.g. I requested a refund a week ago and it still hasn't arrived.",
    height=120,
)

if st.button("Predict", type="primary"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject or body for the ticket.")
    else:
        result = predict_ticket(subject, body)

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Category", result["category"])
        col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        col3.metric("Priority", result["priority"])

        if result["final_decision"] == "Needs Human Review":
            st.warning(
                "⚠️ Confidence is below the 60% threshold. "
                "This ticket is flagged as **Needs Human Review** instead of "
                "being auto-assigned."
            )
        else:
            st.success(f"✅ Final Decision: **{result['final_decision']}**")

        st.caption(
            "Confidence reflects the model's estimated probability for its "
            "top prediction — it is not a guarantee of correctness."
        )

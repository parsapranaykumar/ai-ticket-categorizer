AI Support Ticket Categorizer

An AI-powered support ticket classification system that automatically categorizes support tickets into four departments:

Billing

Technical

HR

General

Live Demo

https://ai-ticket-categorizer-qwgstpicvhmu7uuje6wtib.streamlit.app/

GitHub

https://github.com/parsapranaykumar/ai-ticket-categorizer

Features

Support ticket classification

TF-IDF text feature extraction

Multinomial Naive Bayes classification

Logistic Regression comparison

Prediction confidence

Low-confidence human review

Urgent/Normal priority detection

Streamlit web interface

Machine Learning Pipeline

Support Ticket
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Model
      ↓
Category Prediction
      ↓
Confidence + Priority

Model Evaluation

Model

Test Accuracy

Multinomial Naive Bayes

78.9%

Logistic Regression

89.5%

Logistic Regression achieved the higher test accuracy on the current dataset.

Human Review

Predictions with confidence below 60% are flagged for human review instead of being automatically assigned.

Priority Detection

The application identifies urgent tickets using keywords such as:

urgent

emergency

down

not working

critical

asap

Technologies

Python

Pandas

NumPy

Scikit-learn

Streamlit

Matplotlib

Seaborn

Project Structure

ai-ticket-categorizer/
│
├── data/
│   └── tickets.csv
│
├── notebook/
│   └── ticket_categorizer.ipynb
│
├── app.py
├── ticket_categorizer.py
├── requirements.txt
├── README.md
└── .gitignore

Installation

Clone the repository:

git clone https://github.com/parsapranaykumar/ai-ticket-categorizer.git
cd ai-ticket-categorizer

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the Machine Learning Pipeline

python ticket_categorizer.py

This trains the models, evaluates their performance, and tests the classifier on unseen tickets.

Run the Streamlit Application

streamlit run app.py

Open:

http://localhost:8501

Example

Input:

Subject:
Double charge on my credit card

Body:
I was charged twice for my monthly subscription.

The application predicts the appropriate category and displays confidence and priority.

Future Improvements

Increase the size and diversity of the training dataset

Improve confidence calibration

Experiment with additional NLP models

Add ticket history and database storage

Improve the production deployment

Author

Parsa Pranay Kumar

GitHub: https://github.com/parsapranaykumar

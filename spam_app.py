import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# --- Load dataset ---
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['Category', 'Message']]
data = data.rename(columns={'Category':'label', 'Message':'message'})

# Encode labels
data['label_num'] = data['label'].map({'ham':0, 'spam':1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- Streamlit app ---
st.title("Spam Email Detector")
st.write("Type or paste an email below to check if it is spam or not.")

user_input = st.text_area("Enter your email:")

if st.button("Predict"):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    st.write("🚨 Spam" if prediction==1 else "✅ Not Spam")
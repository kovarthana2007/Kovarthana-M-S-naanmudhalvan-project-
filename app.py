import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Fake News Detection - Exposing the Truth")

input_text = st.text_area("Enter News Article Text")

if st.button("Analyze"):
    vectorized_text = vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)[0]

    if prediction == 'FAKE':
        st.error("Warning: This news is likely FAKE.")
    else:
        st.success("This news appears to be REAL.")

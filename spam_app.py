import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer from the pickle file
with open("regmode.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Streamlit app title and description
st.title("Spam Detection API")
st.write("Enter a message to check if it's spam or not.")

# Input field for the message
user_input = st.text_area("Enter the message:")

# Prediction logic
if st.button("Check Message"):
    if user_input:
        # Transform the input message
        user_input_vectorized = vectorizer.transform([user_input])
        
        # Predict spam or not
        prediction = model.predict(user_input_vectorized)
        prediction_proba = model.predict_proba(user_input_vectorized)[0][1]  # Probability of being spam

        # Display the result
        if prediction[0] == 1:
            st.error(f"The message is **Spam** with a probability of {prediction_proba:.2f}.")
        else:
            st.success(f"The message is **Not Spam** with a probability of {1 - prediction_proba:.2f}.")
    else:
        st.warning("Please enter a message to check.")

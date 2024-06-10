import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os

# Get the directory of the current script
current_directory = os.path.dirname(__file__)

# Construct the full path to the pickle file
pickle_file_path = os.path.join(current_directory, 'lr.pkl')

# Check if the pickle file exists
if not os.path.exists(pickle_file_path):
    st.error(f"The file 'lr.pkl' was not found in the directory {current_directory}")
else:
    # Load the pickle file
    with open(pickle_file_path, 'rb') as file:
        lr = pickle.load(file)

    st.title("Credit Card Fraud Detection")
    st.write("It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.")

    input = st.text_input("Enter all required features like Time, V1, V2, V3, V4, V5, V6, V7, ..., V24, V25, V26, V27, V28, Amount")

    submit = st.button('Submit')

    if submit:
        try:
            # Split the input string and convert to a numpy array
            features = np.asarray([float(i) for i in input.split(',')], dtype=np.float64)
            # Ensure the input features are in the correct shape
            if features.shape[0] == 30:  # Assuming the model expects 30 features
                prediction = lr.predict(features.reshape(1, -1))
                if prediction[0] == 0:
                    st.write("Legitimate Transaction")
                else:
                    st.write("Fraudulent Transaction")
            else:
                st.write("Please enter exactly 30 features.")
        except ValueError:
            st.write("Invalid input. Please ensure all inputs are numeric and properly formatted.")

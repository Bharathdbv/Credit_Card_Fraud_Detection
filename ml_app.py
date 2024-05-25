import pandas as pd
import numpy as np
import streamlit as st
import pickle



lr = pickle.load(open(r'C:\Users\BHARATH\OneDrive\Desktop\ML\AIML_proj\lr.pkl', 'rb'))
st.title("Credit Card Fruad Detection")
st.write("It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.")

input=st.text_input("Enter all Required like Time,V1,V2,V3,V4,V5,V6,V7,.,V24,V25,V26,V27,V28,Amount")

input_df_spilited=input.split(',')
submit=st.button('Submit')

if submit:
    features=np.asarray(input_df_spilited,dtype=np.float64)
    prediction=lr.predict(features.reshape(1,-1))
    if prediction[0]==0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fruad Transaction")

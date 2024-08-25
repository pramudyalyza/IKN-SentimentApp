import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from clean import process_text
from predict import predict_text

def main():
    st.set_page_config(
        page_title="IKN Sentiment Prediction App",
        page_icon="🔮",
        layout="wide"
    )

    st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        .title {
            text-align: center;
            color: #333;
            font-size: 24px;
            margin-bottom: 20px;
        }
        .input-container {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .predict-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    html_temp = """
    <div style="
        background-color: #7FA1C3;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        ">
        <h2 style="
            color: white;
            font-family: Arial, sans-serif;
            font-weight: 600;
            margin: 0;
            ">
            IKN Sentiment Prediction App
        </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        input_text = st.text_area("Provide the text about IKN you want to analyze:", height=100)

        with st.spinner('Processing...'):
            if st.button("Predict"):
                errorId, cleanOutput, errorMessage = process_text(input_text)
                predict_text(errorId, cleanOutput, errorMessage)

if __name__ == '__main__':
    main()

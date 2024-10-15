import json
import nltk
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from clean import process_text
from predict import predict_text
from streamlit_option_menu import option_menu


def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Sentiment Analysis Prediction Module
def sentiment_prediction():
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

# Sentiment Visualization Dashboard Module
def sentiment_dashboard(dfCopy):
        # Ensure retrieval_date is in datetime format
    dfCopy['retrieval_date'] = pd.to_datetime(dfCopy['retrieval_date']).dt.date

    html_temp = """
            <div style="background-color: #7FA1C3; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); text-align: center;">
                <h2 style="color: white; font-family: Arial, sans-serif; font-weight: 600; font-size:20px; margin: 0;">
                    DASHBOARD SENTIMEN ANALYSIS - PEMINDAHAN IBU KOTA NUSANTARA (IKN) KE KALIMATAN TIMUR
                </h2>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.header("Filter Options")
    sentiment_filter = st.sidebar.multiselect("Select Sentiments", options=dfCopy['sentimen'].unique(), default=dfCopy['sentimen'].unique())

    # Date input filter
    min_date = dfCopy['retrieval_date'].min()
    max_date = dfCopy['retrieval_date'].max()
    
    date_filter = st.sidebar.date_input("Filter by Date", value=(min_date, max_date))

    # Validate date filter
    if len(date_filter) == 2:
        if date_filter[0] < min_date or date_filter[0] > max_date:
            st.sidebar.error(f"To ensure accurate results, please select a date range that falls within our data availability. The earliest possible start date is {min_date}, and the latest is {max_date}.")
        elif date_filter[0] >= min_date and date_filter[0] <= max_date:
            # Filter dataframe based on date and sentiment selections
            filtered_df = dfCopy[(dfCopy['retrieval_date'] >= date_filter[0]) & (dfCopy['retrieval_date'] <= date_filter[1])]
            filtered_df = filtered_df[filtered_df['sentimen'].isin(sentiment_filter)]

            # Sentiment counts
            positive_count = filtered_df[filtered_df['sentimen'] == 'Positif'].shape[0]
            negative_count = filtered_df[filtered_df['sentimen'] == 'Negatif'].shape[0]
            total_count = filtered_df.shape[0]

            # Columns for gauges and proportion chart
            gauge_col_pos, gauge_col_neg, proportion_col = st.columns(3)

            # Positive sentiment gauge
            with gauge_col_pos:
                fig_gauge_pos = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=positive_count,
                    title={'text': "Positive Sentiments"},
                    gauge={
                        'axis': {'range': [None, total_count]},
                        'bar': {'color': '#457B9D'},
                    }
                ))
                st.plotly_chart(fig_gauge_pos, use_container_width=True)

            # Negative sentiment gauge
            with gauge_col_neg:
                fig_gauge_neg = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=negative_count,
                    title={'text': "Negative Sentiments"},
                    gauge={
                        'axis': {'range': [None, total_count]},
                        'bar': {'color': '#E63946'},
                    }
                ))
                st.plotly_chart(fig_gauge_neg, use_container_width=True)

            # Donut chart for sentiment proportions
            with proportion_col:
                sentiment_counts = filtered_df['sentimen'].value_counts()
                color_map = {'Positif': '#457B9D', 'Negatif': '#E63946'}

                fig_donut = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.6,
                    color=sentiment_counts.index,
                    color_discrete_map=color_map)
                
                fig_donut.update_traces(
                    textposition='outside', 
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#000000', width=2)))
                
                fig_donut.update_layout(
                    showlegend=False, 
                    margin=dict(l=40, r=40, t=30, b=10))
                
                st.plotly_chart(fig_donut, use_container_width=True)

            # Subheader for bar chart
            st.markdown("<h5 style='color: white; font-family: Arial, sans-serif; font-weight: 500;'>Sentiment Trends Over Time</h5>", unsafe_allow_html=True)

            # Group by date and sentiment for bar chart
            df_grouped = filtered_df.groupby(['retrieval_date', 'sentimen']).size().reset_index(name='counts')

            fig_bar = px.bar(df_grouped, x='retrieval_date', y='counts', color='sentimen', barmode='group', color_discrete_map=color_map)
            fig_bar.update_layout(
                height=350,
                margin=dict(l=40, r=40, t=10, b=10),
                xaxis_title="Date",
                yaxis_title="Count",
                legend_title="Sentiment",
                xaxis=dict(tickformat="%Y-%m-%d")
            )
            st.plotly_chart(fig_bar, use_container_width=True, height=400)

# About Page Module
def about():
    st.markdown("""
        <div style="
            background-color: #7FA1C3;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;">
            <h2 style="color: white; font-family: Arial, sans-serif; font-weight: 600;">
                About this App
            </h2>
        </div>
        <br>
        <p style="font-family: Arial, sans-serif; font-size: 16px; color: white;">
            The IKN Sentiment App allows users to explore and analyze sentiment data related to public opinions on the move of Indonesia's capital to Kalimantan Timur. It features a simple dashboard that updates daily with sentiment trends, pulling data from YouTube comments related to specific IKN videos.
        </p>
        <p style="font-family: Arial, sans-serif; font-size: 16px; color: white;">
            The app also includes a real-time sentiment prediction feature based on input text. This prediction is generated using three different machine learning models—KNN, Random Forest, and Decision Tree—where the final output is determined by a majority vote from these models.<br><br>The training data, which was manually scraped from YouTube, undergoes several preprocessing steps, including POS tagging that will be combined with TF-IDF vectorization in feature engineering process as additional input.
        </p>
        <p style="font-family: Arial, sans-serif; font-size: 16px; color: white;">
            Feel free to explore the app and use the various tools available to 
            gain insights into public sentiment on this critical national issue.
        </p>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.set_page_config(page_title="IKN Sentiment App", page_icon="🔮", layout="wide")

    download_nltk_resources()

    df = pd.read_parquet('Files/Result.parquet')

    with st.spinner('Processing...'):
        # Sidebar menu with the same options, also default to the dashboard
        st.markdown("""
            <style>
                .css-1d391kg {
                    font-size: 16px;
                }
                .css-1d391kg a {
                    font-weight: normal !important;
                }
            </style>""", unsafe_allow_html=True)

        with st.sidebar:
            sidebar_selected = option_menu("", ["Sentiment Dashboard", "Sentiment Prediction", "About"],
                                        icons=["bar-chart-line", "activity", "info-circle"],
                                        menu_icon="cast", default_index=0)

    # Display the correct page based on menu selection
    if sidebar_selected == "Sentiment Dashboard":
        dfCopy = df.copy()
        sentiment_dashboard(dfCopy)
    elif sidebar_selected == "Sentiment Prediction":
        sentiment_prediction()
    elif sidebar_selected == "About":
        about()

if __name__ == '__main__':
    main()

# modules/sentiment_analysis.py

import os
import torch
from typing import Optional, Dict

from transformers import pipeline
import streamlit as st
from dotenv import load_dotenv

from .utils import load_config, normalize_text

# Load environment variables from .env file, if it exists
load_dotenv()

# Load configuration from config.yaml
config = load_config('config/config.yaml')

# Function to load and cache the sentiment analysis pipeline
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline(model_name: str, device: int) -> Optional[pipeline]:
    """
    Loads the sentiment analysis pipeline from HuggingFace.

    Args:
        model_name (str): Name or path of the sentiment analysis model.
        device (int): Device to be used (-1 for CPU, 0 for the first GPU, etc.).

    Returns:
        Optional[pipeline]: Sentiment analysis pipeline or None if loading fails.
    """
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment analysis pipeline: {e}")
        return None

def analyze_sentiment(
    text: str,
    device: int = -1
) -> str:
    """
    Analyzes the sentiment of the provided text.

    Args:
        text (str): Text to be analyzed.
        device (int, optional): Device to be used (-1 for CPU, 0 for the first GPU, etc.). Defaults to -1.

    Returns:
        str: Sentiment label ("POSITIVE", "NEGATIVE", or "NEUTRAL").
    """
    if not text.strip():
        st.warning("Empty text provided for sentiment analysis.")
        return "NEUTRAL"

    sentiment_pipeline = get_sentiment_pipeline(
        model_name=config['sentiment_analysis']['model'],
        device=device
    )

    if sentiment_pipeline is None:
        st.error("Sentiment analysis pipeline not available.")
        return "NEUTRAL"

    try:
        results = sentiment_pipeline(text)
        if not results:
            st.warning("No results returned from sentiment analysis.")
            return "NEUTRAL"
        sentiment = results[0]['label']
        return sentiment
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return "NEUTRAL"

def run_sentiment_analysis_interface():
    """
    Runs the sentiment analysis interface using Streamlit.
    """
    st.header("📈 Sentiment Analysis")
    
    # Parameters with tooltips
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Language",
            ["Portuguese", "English", "Spanish"],
            index=0,
            help="Select the language of the text for sentiment analysis."
        )
    with col2:
        device_choice = st.selectbox(
            "Device",
            ["CPU", "GPU"],
            index=0,
            help="Choose the device to run the sentiment analysis."
        )
    
    # Determine the device
    device = 0 if device_choice == "GPU" and torch.cuda.is_available() else -1
    
    # User input
    user_input = st.text_area("Text for Sentiment Analysis:", height=150)
    
    # Button to perform the analysis
    analyze_button = st.button("Analyze Sentiment")
    
    # Button to clear the input text
    clear_button = st.button("Clear Text")
    
    if clear_button:
        st.session_state['sentiment_input'] = ""
        st.experimental_rerun()
    
    if analyze_button:
        if not user_input.strip():
            st.warning("Please enter text for sentiment analysis.")
        else:
            try:
                with st.spinner("Analyzing sentiment..."):
                    normalized_text = normalize_text(user_input)
                    sentiment = analyze_sentiment(normalized_text, device=device)
                    st.success(f"Detected Sentiment: **{sentiment}**")
            except Exception as e:
                st.error(f"An error occurred during sentiment analysis: {e}")

if __name__ == "__main__":
    st.warning("This module should not be run directly.")

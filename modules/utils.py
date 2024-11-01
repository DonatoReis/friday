# modules/utils.py

import os
import yaml
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Tuple, Optional

import streamlit as st

def setup_logging():
    """
    Configures the logging system for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log")
        ]
    )

def load_config(config_path: str) -> Optional[dict]:
    """
    Loads configurations from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        Optional[dict]: Configuration dictionary or None if it fails.
    """
    if not os.path.exists(config_path):
        st.error(f"Configuration file '{config_path}' not found.")
        logging.error(f"Configuration file '{config_path}' not found.")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from '{config_path}'.")
            return config
    except yaml.YAMLError as e:
        st.error(f"Error reading configuration file: {e}")
        logging.error(f"Error reading configuration file: {e}")
        return None
    except UnicodeDecodeError as e:
        st.error(f"Decoding error reading configuration file: {e}")
        logging.error(f"Decoding error reading configuration file: {e}")
        return None

def load_model_and_tokenizer(model_path: str) -> Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]:
    """
    Loads the GPT-2 model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the trained model directory.
    
    Returns:
        Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]: Loaded model and tokenizer or None if it fails.
    """
    if not os.path.exists(model_path):
        st.error(f"Model path '{model_path}' not found.")
        logging.error(f"Model path '{model_path}' not found.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        logging.info(f"Model and tokenizer loaded from '{model_path}'.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        logging.error(f"Error loading model or tokenizer: {e}")
        return None, None

def normalize_text(text: str) -> str:
    """
    Normalizes text by removing unnecessary spaces and special characters.
    
    Args:
        text (str): Text to be normalized.
    
    Returns:
        str: Normalized text.
    """
    normalized = ' '.join(text.strip().split())
    logging.debug(f"Normalized text: '{normalized}'")
    return normalized

def save_feedback(feedback: list, feedback_file: str = "feedback_data.txt") -> bool:
    """
    Saves the collected feedback to a text file for continuous learning.
    
    Args:
        feedback (list): List of dictionaries containing feedback.
        feedback_file (str, optional): Path to the feedback file. Defaults to "feedback_data.txt".
    
    Returns:
        bool: True if saving is successful, False otherwise.
    """
    try:
        with open(feedback_file, 'a', encoding='utf-8') as f:
            for entry in feedback:
                f.write(f"{entry}\n")
        logging.info(f"Feedback saved to '{feedback_file}'.")
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        logging.error(f"Error saving feedback: {e}")
        return False

def ensure_directory(path: str) -> bool:
    """
    Ensures that a directory exists. If it does not, attempts to create it.
    
    Args:
        path (str): Directory path.
    
    Returns:
        bool: True if the directory exists or is created successfully, False otherwise.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory ensured: '{path}'.")
        return True
    except Exception as e:
        st.error(f"Error creating directory '{path}': {e}")
        logging.error(f"Error creating directory '{path}': {e}")
        return False

def validate_model_files(model_path: str, required_files: list) -> bool:
    """
    Validates that all necessary model files are present.
    
    Args:
        model_path (str): Path to the model directory.
        required_files (list): List of required file names.
    
    Returns:
        bool: True if all files are present, False otherwise.
    """
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"Missing files in model '{model_path}': {missing_files}")
        logging.error(f"Missing files in model '{model_path}': {missing_files}")
        return False
    logging.info(f"All required files are present in '{model_path}'.")
    return True

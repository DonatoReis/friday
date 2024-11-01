# main.py

import streamlit as st
import sys
import logging
from modules.interface import run_interface
from modules.utils import load_config
from dotenv import load_dotenv

def setup_logging():
    """
    Sets up the logging system for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log")
        ]
    )

def validate_config(config):
    """
    Validates if all the necessary configurations are present.
    
    Args:
        config (dict): Loaded configuration dictionary.
    
    Returns:
        bool: True if all configurations are present, False otherwise.
    """
    required_sections = [
        'general',
        'training',
        'optimization',
        'chatbot',
        'prediction',
        'sentiment_analysis',
        'huggingface',
        'quantization',
        'feedback',
        'other'
    ]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        st.error(f"Missing configuration sections: {missing_sections}")
        logging.error(f"Missing configuration sections: {missing_sections}")
        return False
    return True

def main():
    """
    Main function that runs the application interface.
    """
    try:
        # Set up logging
        setup_logging()
        logging.info("Application started.")
        
        # Load environment variables from .env file, if it exists
        load_dotenv()
        logging.info("Environment variables loaded.")
        
        # Load configurations from config.yaml
        config = load_config('config/config.yaml')
        if config is None:
            st.error("Failed to load configurations. Check the 'config/config.yaml' file.")
            logging.error("Failed to load configurations.")
            sys.exit(1)
        
        # Validate configurations
        if not validate_config(config):
            logging.error("Invalid configurations. Shutting down the application.")
            sys.exit(1)

        # Run the interface
        run_interface()
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("Unexpected error in the application.")
        sys.exit(1)
    finally:
        logging.info("Application stopped.")

if __name__ == "__main__":
    main()

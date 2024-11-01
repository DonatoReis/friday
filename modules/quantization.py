# modules/quantization.py

import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import streamlit as st
from typing import Optional, Dict

from .utils import load_config, load_model_and_tokenizer
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# Load configuration from config.yaml
config = load_config('config/config.yaml')

# Function to load and cache the model and tokenizer
@st.cache_resource(show_spinner=False)
def get_model_and_tokenizer(model_path: str) -> Optional[tuple]:
    """
    Loads the model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the directory of the trained model.
    
    Returns:
        Optional[tuple]: Loaded model and tokenizer or None if loading fails.
    """
    return load_model_and_tokenizer(model_path)

def quantize_model(
    model_path: str,
    quantized_output_dir: str,
    dtype: str = "torch.qint8",
    strategy: str = "dynamic",
    calibration_data: Optional[str] = None
) -> bool:
    """
    Applies quantization to the trained GPT-2 model.
    
    Args:
        model_path (str): Path to the directory of the trained model.
        quantized_output_dir (str): Path to save the quantized model.
        dtype (str, optional): Data type for quantization. Defaults to "torch.qint8".
        strategy (str, optional): Quantization strategy ("dynamic", "static", "hybrid"). Defaults to "dynamic".
        calibration_data (Optional[str], optional): Path to calibration data if the strategy is static. Defaults to None.
    
    Returns:
        bool: True if quantization is successful, False otherwise.
    """
    try:
        model, tokenizer = get_model_and_tokenizer(model_path)
        if model is None or tokenizer is None:
            st.error("Model or tokenizer not loaded correctly.")
            return False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        st.info(f"Starting model quantization with dtype={dtype} and strategy={strategy}.")

        if strategy == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=getattr(torch, dtype)
            )
        elif strategy == "static":
            if calibration_data is None:
                st.error("Calibration data is required for static quantization.")
                return False
            
            # Load calibration data
            from datasets import load_dataset
            dataset = load_dataset('json', data_files={'calibration': calibration_data}, split='calibration')
            
            def calibrate(model, dataset, tokenizer, device, max_length=128):
                model.eval()
                with torch.no_grad():
                    for example in dataset:
                        inputs = tokenizer.encode(example['text'], return_tensors='pt', max_length=max_length, truncation=True).to(device)
                        model(inputs)
            
            # Prepare the model for static quantization
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            calibrate(model, dataset, tokenizer, device)
            quantized_model = torch.quantization.convert(model, inplace=True)
        elif strategy == "hybrid":
            # Hybrid quantization implementation
            st.info("Hybrid quantization strategy is not yet implemented.")
            return False
        else:
            st.error(f"Quantization strategy '{strategy}' is not supported.")
            return False

        # Create output directory if it does not exist
        os.makedirs(quantized_output_dir, exist_ok=True)

        # Save the quantized model
        quantized_model.save_pretrained(quantized_output_dir)
        tokenizer.save_pretrained(quantized_output_dir)

        # Verify if all files were saved
        required_files = ['tokenizer.json', 'config.json', 'vocab.json', 'merges.txt']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(quantized_output_dir, f))]
        if missing_files:
            st.error(f"The quantized model in directory '{quantized_output_dir}' is incomplete. Missing files: {missing_files}")
            return False
        else:
            st.success(f"Quantization completed successfully and model saved in '{quantized_output_dir}'.")
            return True

    except Exception as e:
        st.error(f"An error occurred during quantization: {e}")
        return False

def run_quantization_interface():
    """
    Runs the quantization interface using Streamlit.
    """
    st.header("⚙️ GPT-2 Model Quantization")
    
    # Parameters with tooltips
    col1, col2 = st.columns(2)
    with col1:
        dtype = st.selectbox(
            "Data Type for Quantization",
            options=["torch.qint8", "torch.float16"],
            index=["torch.qint8", "torch.float16"].index(config['quantization']['dtype']),
            help="Choose the data type for model quantization."
        )
    with col2:
        strategy = st.selectbox(
            "Quantization Strategy",
            options=["dynamic", "static", "hybrid"],
            index=["dynamic", "static", "hybrid"].index(config['quantization']['strategy']),
            help="Choose the quantization strategy to be used."
        )
    
    # Upload calibration data if the strategy is static
    calibration_file = None
    if strategy == "static":
        calibration_file = st.file_uploader(
            "Select calibration file (.jsonl)",
            type=["jsonl"],
            help="Upload a .jsonl file containing calibration examples."
        )
    
    # Output directory for the quantized model
    quantized_output_dir = st.text_input(
        "Directory to save the quantized model",
        value=os.path.join(config['training']['output_dir'], "quantized"),
        help="Enter the path to save the quantized model."
    )
    
    # Button to start quantization
    if st.button("Start Quantization"):
        model_path = config['training']['output_dir']
        
        # Validate model presence
        required_files = ['tokenizer.json', 'config.json', 'vocab.json', 'merges.txt']
        if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
            st.error(f"The model in directory '{model_path}' is incomplete or not trained.")
            return
        
        # Save the calibration file temporarily if needed
        calibration_path = None
        if strategy == "static" and calibration_file is not None:
            with open("calibration_data.jsonl", "wb") as f:
                f.write(calibration_file.getbuffer())
            calibration_path = "calibration_data.jsonl"
        
        # Start quantization with error handling
        success = quantize_model(
            model_path=model_path,
            quantized_output_dir=quantized_output_dir,
            dtype=dtype,
            strategy=strategy,
            calibration_data=calibration_path
        )
        
        # Remove temporary calibration file
        if calibration_path and os.path.exists(calibration_path):
            os.remove(calibration_path)
        
        if success and config['quantization']['enabled']:
            st.info("Quantization and model saving completed successfully.")
        elif not success:
            st.error("Failed to quantize the model. Check logs for more details.")
    

def run_quantization_module():
    """
    Runs the quantization module interface.
    """
    run_quantization_interface()

if __name__ == "__main__":
    st.warning("This module should not be run directly.")

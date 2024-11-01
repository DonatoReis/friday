# modules/interface.py

import streamlit as st
import yaml
import os
import tempfile
import torch
import logging
from datetime import datetime
from typing import Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from .train_transformers import train_with_transformers
from .train_optuna import train_with_optuna
from .chat_predict import chat_with_model, predict_next_word
from .quantization import quantize_model
from .utils import load_config, normalize_text, load_model_and_tokenizer
from .sentiment_analysis import analyze_sentiment

# Import for managing environment variables
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# Function to load the configuration file
config = load_config('config/config.yaml')

# Function to load and cache the model and tokenizer
@st.cache_resource(show_spinner=False)
def get_model_and_tokenizer(model_path: str) -> Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]:
    """
    Loads the GPT-2 model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the directory of the trained model.

    Returns:
        Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]: Loaded model and tokenizer or None if loading fails.
    """
    model, tokenizer = load_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded correctly.")
        return None, None
    return model, tokenizer

# Streamlit page configuration

def setup_page():
    st.set_page_config(
        page_title=config['general']['app_title'],
        page_icon=config['general']['page_icon'],
        layout=config['general']['layout'],
        initial_sidebar_state=config['general']['initial_sidebar_state'],
    )

# Function to apply custom CSS

def apply_custom_css():
    """
    Applies custom CSS styles to enhance the Streamlit interface.
    """
    custom_css = """
    <style>
    /* Local "Inter" font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #171717;
        color: #FFFFFF;
    }

    .sidebar .sidebar-content {
        background-color: #1A1A1A;
    }

    /* Updated CSS selectors for greater robustness */
    div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        background-color: #1A1A1A;
    }

    .css-2trqyj {
        background-color: #2b2b2b;
    }

    .css-1lcbmhc {
        color: #2B4FFF;
    }

    /* Buttons */
    .stButton>button {
        color: #FFFFFF;
        background-color: #2B4FFF;
    }

    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #2b2b2b;
        color: #FFFFFF;
    }

    .stFileUploader>div>div>label>div {
        background-color: #2b2b2b;
        color: #FFFFFF;
    }

    /* Tabs */
    .css-1kyxreq {
        background-color: #2b2b2b;
    }

    /* Chat bubble styling */
    .user_message {
        background-color: #3a3a3a;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        text-align: right;
    }

    .bot_message {
        background-color: #2b4fff;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        text-align: left;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: #2B4FFF;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #2b2b2b;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position above the icon */
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Avatars */
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        vertical-align: middle;
        margin-right: 10px;
    }

    /* Timestamps */
    .timestamp {
        font-size: 10px;
        color: #AAAAAA;
        margin-left: 5px;
    }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Main function for the Streamlit interface

def run_interface():
    # Configure the page
    setup_page()

    # Apply custom CSS
    apply_custom_css()

    # Load HuggingFace token from environment variable
    huggingface_token = os.getenv(config['huggingface']['token_env_var'], "")
    
    # Training Settings in the Sidebar
    st.sidebar.header("Training Settings")
    
    # User Guide
    with st.sidebar.expander("📖 User Guide"):
        st.markdown("""
        **GPT-2 Trainer & Chatbot**
        
        This application allows training a custom GPT-2 model and interacting with it via a chatbot.
        
        **Limitations:**
        - Long training sessions may take significant time, especially without a GPU.
        - The quality of responses depends on the quality and quantity of training data.
        
        **How to Use:**
        1. **Training:**
            - Upload a `.jsonl` or `.txt` file with training data.
            - Set the training parameters as needed.
            - If desired, enable hyperparameter tuning with Optuna.
            - Click "Start Training" and wait for it to complete.
        2. **Chatbot:**
            - After training, use the "🤖 Chatbot" tab to chat with the trained model.
        3. **Predict Next Word:**
            - Use the "🔮 Predict Next Word" tab to get word suggestions based on provided text.
        """)
    
    # Upload training file using tempfile for secure management
    uploaded_file = st.sidebar.file_uploader(
        "Select training file (.jsonl or .txt)", type=["jsonl", "txt"]
    )
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose GPT-2 model",
        config['training']['model_options'],
        index=config['training']['model_options'].index(config['training']['default_model'])
    )
    
    # Custom HuggingFace model input
    custom_model = st.sidebar.text_input(
        "Custom HuggingFace Model",
        value=config['huggingface']['custom_model'],
        help="Enter the path or name of the custom HuggingFace model."
    )
    
    # Output directory
    output_dir = st.sidebar.text_input(
        "Directory to save the trained model",
        value=config['training']['output_dir']
    )
    
    # Training parameters with validation
    num_epochs = st.sidebar.number_input(
        "Number of epochs",
        min_value=int(1),
        max_value=int(1000),
        value=int(config['training']['num_epochs']),
        step=int(1)
    )
    batch_size = st.sidebar.number_input(
        "Batch size",
        min_value=int(1),
        max_value=int(128),
        value=int(config['training']['batch_size']),
        step=int(1)
    )
    max_length = st.sidebar.number_input(
        "Maximum token length",
        min_value=int(16),
        max_value=int(2048),
        value=int(config['training']['max_length']),
        step=int(16)
    )
    learning_rate = st.sidebar.number_input(
        "Learning rate",
        min_value=float(1e-6),
        max_value=float(1e-3),
        value=float(config['training']['learning_rate']),
        step=float(1e-6),
        format="%.6f"
    )
    validation_split = st.sidebar.slider(
        "Validation split",
        0.0,
        0.5,
        config['training']['validation_split'],
        0.05
    )
    
    # Gradient Accumulation Steps
    gradient_accumulation_steps = st.sidebar.number_input(
        "Gradient Accumulation Steps",
        min_value=int(1),
        max_value=int(64),
        value=int(config['training']['gradient_accumulation_steps']),
        step=int(1),
        help="Number of steps for gradient accumulation before updating weights."
    )
    
    # Reduced precision (fp16) or standard precision (fp32)
    fp16_option = st.sidebar.selectbox(
        "Select model precision",
        ["auto", "fp16", "fp32"],
        index=["auto", "fp16", "fp32"].index(config['training']['fp16_option'])
    )
    if fp16_option == "fp16":
        fp16 = True
    elif fp16_option == "fp32":
        fp16 = False
    else:
        fp16 = config['training']['fp16_option'] == "auto"
    
    # Option for automatic hyperparameter tuning
    automatic_hp_tuning = st.sidebar.checkbox(
        "Enable hyperparameter tuning (Optuna)",
        value=config['training']['automatic_hp_tuning']
    )
    
    # Evaluation strategy option
    st.sidebar.header("Evaluation Strategy")
    use_eval = st.sidebar.checkbox(
        "Use evaluation strategy",
        value=config['training']['use_eval_strategy']
    )
    if use_eval:
        eval_strategy = st.sidebar.selectbox(
            "Evaluation strategy",
            ["steps", "epoch"]
        )
        eval_steps = None
        if eval_strategy == "steps":
            eval_steps = st.sidebar.number_input(
                "Number of steps between evaluations",
                min_value=int(100),
                max_value=int(10000),
                value=int(config['training']['eval_steps_hf']),
                step=int(100)
            )
        # Option to load the best model at the end of training
        load_best = st.sidebar.checkbox(
            "Load best model at the end of training",
            value=config['training']['load_best_model_at_end']
        )
    else:
        eval_strategy = None
        eval_steps = None
        load_best = False  # Do not load the best model if there is no evaluation
    
    # Button to start training
    if st.sidebar.button("Start Training"):
        if uploaded_file is None:
            st.sidebar.error("Please upload a training file.")
        else:
            # Validate input parameters
            if not (0 <= validation_split <= 0.5):
                st.sidebar.error("Validation split must be between 0 and 0.5.")
            elif not (1e-6 <= learning_rate <= 1e-3):
                st.sidebar.error("Learning rate must be between 1e-6 and 1e-3.")
            elif not (16 <= max_length <= 1024):
                st.sidebar.error("Maximum token length must be between 16 and 1024.")
            else:
                # Use tempfile for secure temporary file management
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    data_path = temp_file.name
                
                # Progress bar (only for HuggingFace Trainer)
                progress_bar = st.sidebar.progress(0) if use_eval and eval_strategy == "steps" else None
                status_text = st.sidebar.empty()
                
                # Start training with error handling
                try:
                    with st.spinner("Starting training..."):
                        if automatic_hp_tuning:
                            # Use PyTorch Lightning with Optuna
                            train_with_optuna(
                                data_path=data_path,
                                model_name=selected_model,
                                output_dir=output_dir,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                max_length=max_length,
                                learning_rate=learning_rate,
                                validation_split=validation_split,
                                fp16=fp16,
                                metric_for_best_model=config['training']['metric_for_best_model'],
                                automatic_hp_tuning=automatic_hp_tuning,
                                use_eval_strategy=use_eval,
                                eval_strategy=eval_strategy,
                                eval_steps=eval_steps,
                                load_best_model_at_end=load_best,
                                huggingface_token=huggingface_token,
                                custom_model=custom_model
                            )
                        else:
                            # Use HuggingFace Trainer
                            train_with_transformers(
                                data_path=data_path,
                                model_name=selected_model,
                                output_dir=output_dir,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                max_length=max_length,
                                learning_rate=learning_rate,
                                validation_split=validation_split,
                                save_steps=config['training']['save_steps'],
                                logging_steps=config['training']['logging_steps'],
                                save_total_limit=config['training']['save_total_limit'],
                                fp16=fp16,
                                load_best_model_at_end=use_eval,  # Important change
                                metric_for_best_model=config['training']['metric_for_best_model'],
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                use_eval_strategy=use_eval,
                                eval_strategy=eval_strategy,
                                eval_steps_hf=eval_steps if eval_strategy == "steps" else None,
                                progress_bar=progress_bar,
                                huggingface_token=huggingface_token,
                                custom_model=custom_model
                            )
                    
                    st.sidebar.success("Training successfully completed!")
                    
                    # Clear cache to reload the trained model
                    get_model_and_tokenizer.clear()

                    # Check if all model files were saved correctly
                    model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
                    if not all(os.path.exists(os.path.join(output_dir, f)) for f in model_files):
                        st.error(f"Model in directory '{output_dir}' is incomplete. Missing files: {[f for f in model_files if not os.path.exists(os.path.join(output_dir, f))]}")
                    else:
                        st.sidebar.success("Model saved correctly!")
                    
                    # Load the best model or the last saved model based on the evaluation strategy
                    if use_eval:
                        st.sidebar.info("Loading the best trained model based on the metric.")
                        # Logic to load the best model based on the metric
                    else:
                        st.sidebar.info("Loading the last trained model.")
                        # Logic to load the last saved model

                    # Restart the application to update the interface
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"An error occurred during training: {e}")
                finally:
                    # Remove the temporary file after training
                    if os.path.exists(data_path):
                        os.remove(data_path)
    
    # Tabs for Chat and Prediction
    tab1, tab2 = st.tabs(["🤖 Chatbot", "🔮 Predict Next Word"])
    
    with tab1:
        st.header("Chat with GPT-2 Model")
        model_path = output_dir if os.path.exists(output_dir) else config['training']['output_dir']
        
        # Check if the model is complete before proceeding
        model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
        if not all(os.path.exists(os.path.join(model_path, f)) for f in model_files):
            st.error(f"Model in directory '{model_path}' is incomplete or not trained.")
            st.stop()  # Stop execution of the tab if the model is not available
        else:
            # Load model and tokenizer with cache
            model, tokenizer = get_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                st.error("Failed to load model or tokenizer.")
                st.stop()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
    
                # Conversation history stored in Session State
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
    
                # Display message history
                for chat in st.session_state['chat_history']:
                    if chat['role'] == 'user':
                        st.markdown(f"""
                        <div>
                            <img src="https://i.imgur.com/7yUvePI.png" class="avatar">
                            <span class="user_message">{chat['content']}</span>
                            <span class="timestamp">{chat['timestamp']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div>
                            <img src="https://i.imgur.com/6RMhx.gif" class="avatar">
                            <span class="bot_message">{chat['content']}</span>
                            <span class="timestamp">{chat['timestamp']}</span>
                        </div>
                        """, unsafe_allow_html=True)
    
                # Parameters with tooltips
                col1, col2, col3, col4 = st.columns([1, 1, 1, 0.2])
                with col1:
                    temperature = st.slider("Temperature", 0.1, 1.5, config['chatbot']['temperature'], 0.1)
                with col2:
                    top_k = st.slider("Top K", 10, 100, config['chatbot']['top_k'], 10)
                with col3:
                    top_p = st.slider("Top P", 0.5, 1.0, config['chatbot']['top_p'], 0.05)
                with col4:
                    st.markdown("""
                    <div class="tooltip">?
                        <span class="tooltiptext">Controls the diversity of text generation. Higher values result in more varied responses.</span>
                    </div>
                    """, unsafe_allow_html=True)
    
                with col1:
                    penalty = st.slider("Repetition Penalty", 1.0, 2.0, config['chatbot']['penalty'], 0.1)
                with col2:
                    debug_mode = st.checkbox("Debug Mode", value=config['chatbot']['debug_mode'])
                with col3:
                    language = st.selectbox("Language", ["Portuguese", "English", "Spanish"], index=0)
                with col4:
                    pass  # Space for alignment
    
                # User input
                user_input = st.text_input("You:", key="chat_input")
    
                # Button disabled when the field is empty
                send_button = st.button("Send", disabled=not user_input.strip())
    
                # Button to clear the conversation history
                clear_button = st.button("Clear Conversation")
    
                if clear_button:
                    st.session_state['chat_history'] = []
                    # Do not use st.experimental_rerun to avoid loops
                    st.experimental_rerun()
    
                if send_button:
                    if not user_input.strip():
                        st.warning("Please enter text to chat.")
                    else:
                        try:
                            with st.spinner("The bot is thinking..."):
                                normalized_input = normalize_text(user_input)
    
                                # Sentiment analysis
                                sentiment = analyze_sentiment(normalized_input, device=0 if device.type == 'cuda' else -1)
    
                                # Adjust response based on sentiment
                                if sentiment == "NEGATIVE":
                                    prompt = f"{normalized_input}\nRespond in a comforting manner."
                                else:
                                    prompt = normalized_input
    
                                response, debug_info = chat_with_model(
                                    prompt,
                                    model_path=model_path,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                    penalty=penalty,
                                    debug_mode=debug_mode
                                )
    
                                # Add to history
                                st.session_state['chat_history'].append({
                                    'role': 'user',
                                    'content': normalized_input,
                                    'timestamp': datetime.now().strftime("%H:%M:%S")
                                })
                                st.session_state['chat_history'].append({
                                    'role': 'bot',
                                    'content': response,
                                    'timestamp': datetime.now().strftime("%H:%M:%S")
                                })
    
                                # Continuous learning feedback
                                if 'continuous_learning' not in st.session_state:
                                    st.session_state['continuous_learning'] = []
                                st.session_state['continuous_learning'].append({
                                    'input': normalized_input,
                                    'response': response
                                })
    
                                # Show debug info if enabled
                                if debug_info:
                                    st.write("**Debug Info:**")
                                    st.json(debug_info)
    
                        except Exception as e:
                            st.error(f"An error occurred while generating the response: {e}")

    with tab2:
        st.header("Predict Next Word")
        model_path = output_dir if os.path.exists(output_dir) else config['training']['output_dir']
        
        # Check if the model is complete before proceeding
        model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
        if not all(os.path.exists(os.path.join(model_path, f)) for f in model_files):
            st.error(f"Model in directory '{model_path}' is incomplete or not trained.")
            st.stop()
        else:
            # Load model and tokenizer with cache
            model, tokenizer = get_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                st.error("Failed to load model or tokenizer.")
                st.stop()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
    
                # User input
                input_text = st.text_input("Enter text to predict the next word:", key="predict_input")
    
                # Button to predict
                predict_button = st.button("Predict Next Word", disabled=not input_text.strip())
    
                if predict_button:
                    if not input_text.strip():
                        st.warning("Please enter text to predict the next word.")
                    else:
                        try:
                            with st.spinner("Predicting the next word..."):
                                next_word = predict_next_word(
                                    input_text,
                                    model_path=model_path,
                                    device=device
                                )
                                st.success(f"Predicted next word: **{next_word}**")
                        except Exception as e:
                            st.error(f"An error occurred while predicting the next word: {e}")

# Run the interface
run_interface()

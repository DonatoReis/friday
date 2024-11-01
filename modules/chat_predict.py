# modules/chat_predict.py

import os
import torch
from typing import Tuple, Optional, Dict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import streamlit as st
from datetime import datetime
from .utils import load_model_and_tokenizer, normalize_text
from .sentiment_analysis import analyze_sentiment
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# Load configurations from config.yaml
from .utils import load_config

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
    model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
    
    missing_files = [f for f in model_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"Model in directory '{model_path}' is incomplete. Missing files: {missing_files}")
        return None, None
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    if model is None or tokenizer is None:
        st.error("Failed to load model or tokenizer.")
        return None, None
    
    return model, tokenizer

def chat_with_model(
    prompt: str,
    model_path: str = './fine-tune-gpt2',
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    penalty: float = 1.0,
    debug_mode: bool = False
) -> Tuple[str, Optional[Dict]]:
    """
    Generates a response from the GPT-2 model based on the provided prompt.
    
    Args:
        prompt (str): Input text for the model.
        model_path (str, optional): Path to the trained model. Defaults to './fine-tune-gpt2'.
        temperature (float, optional): Temperature parameter for text generation. Defaults to 1.0.
        top_k (int, optional): Top K parameter for text generation. Defaults to 50.
        top_p (float, optional): Top P parameter for text generation. Defaults to 0.95.
        penalty (float, optional): Penalty parameter for repetition. Defaults to 1.0.
        debug_mode (bool, optional): If True, returns debugging information. Defaults to False.
    
    Returns:
        Tuple[str, Optional[Dict]]: Generated response and debug data if enabled.
    """
    model, tokenizer = get_model_and_tokenizer(model_path)
    
    if model is None or tokenizer is None:
        st.error("Model or tokenizer were not loaded correctly.")
        return "", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.size(1) + 50,  # Generate up to 50 additional tokens
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=penalty,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
    
        debug_info = None
        if debug_mode:
            logits = model(input_ids).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
            debug_info = {
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probabilities
            }
    
        return generated_text, debug_info

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "", None

def predict_next_word(
    text: str,
    model_path: str = './fine-tune-gpt2',
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    penalty: float = 1.0,
    debug_mode: bool = False
) -> Tuple[str, Optional[Dict]]:
    """
    Predicts the next word after the given text.
    
    Args:
        text (str): Input text.
        model_path (str, optional): Path to the trained model. Defaults to './fine-tune-gpt2'.
        temperature (float, optional): Temperature parameter for text generation. Defaults to 1.0.
        top_k (int, optional): Top K parameter for text generation. Defaults to 50.
        top_p (float, optional): Top P parameter for text generation. Defaults to 0.95.
        penalty (float, optional): Penalty parameter for repetition. Defaults to 1.0.
        debug_mode (bool, optional): If True, returns debugging information. Defaults to False.
    
    Returns:
        Tuple[str, Optional[Dict]]: Predicted next word and debug data if enabled.
    """
    model, tokenizer = get_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded correctly.")
        return "", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.size(1) + 1,  # Only one additional word
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=penalty,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        next_token_id = outputs[0, -1].unsqueeze(0)
        next_word = tokenizer.decode(next_token_id, skip_special_tokens=True).strip()
    
        debug_info = None
        if debug_mode:
            logits = model(input_ids).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
            debug_info = {
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probabilities
            }
    
        return next_word, debug_info

    except Exception as e:
        st.error(f"Error predicting next word: {e}")
        return "", None

def run_chatbot_interface():
    """
    Runs the chatbot interface using Streamlit.
    """
    st.header("🤖 GPT-2 Chatbot")
    
    # Load model and tokenizer with cache
    model_path = config['training']['output_dir']
    model, tokenizer = get_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded correctly.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Conversation history stored in Session State
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display conversation history
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            st.markdown(f"""
            <div style="text-align: right;">
                <span style="background-color: #3a3a3a; border-radius: 10px; padding: 10px; display: inline-block;">
                    <strong>You:</strong> {chat['content']}
                </span>
                <span style="font-size: 10px; color: #AAAAAA;">{chat['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: left;">
                <span style="background-color: #2b4fff; border-radius: 10px; padding: 10px; display: inline-block;">
                    <strong>Bot:</strong> {chat['content']}
                </span>
                <span style="font-size: 10px; color: #AAAAAA;">{chat['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Parameters with tooltips
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=config['chatbot']['temperature'],
            step=0.1,
            help="Controls the randomness of the text generation. Higher values result in more varied responses."
        )
    with col2:
        top_k = st.slider(
            "Top K",
            min_value=10,
            max_value=100,
            value=config['chatbot']['top_k'],
            step=10,
            help="Considers the top K most probable tokens for the next word."
        )
    with col3:
        top_p = st.slider(
            "Top P",
            min_value=0.5,
            max_value=1.0,
            value=config['chatbot']['top_p'],
            step=0.05,
            help="Considers the cumulative probability of tokens for the next word."
        )
    
    col4, col5 = st.columns(2)
    with col4:
        penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=config['chatbot']['penalty'],
            step=0.1,
            help="Penalizes repetition of already generated words."
        )
    with col5:
        debug_mode = st.checkbox(
            "Debug Mode",
            value=config['chatbot']['debug_mode'],
            help="Enables additional debugging information."
        )
    
    # User input
    user_input = st.text_input("You:", key="chat_input")
    
    # Button to send the message
    send_button = st.button("Send", disabled=not user_input.strip())
    
    # Button to clear the conversation history
    clear_button = st.button("Clear Conversation")
    
    if clear_button:
        st.session_state['chat_history'] = []
        st.experimental_rerun()
    
    if send_button:
        if not user_input.strip():
            st.warning("Please enter some text to chat.")
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
                        prompt=prompt,
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

def run_predict_interface():
    """
    Runs the next word prediction interface using Streamlit.
    """
    st.header("🔮 Predict Next Word")
    
    # Load model and tokenizer with cache
    model_path = config['training']['output_dir']
    model, tokenizer = get_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded correctly.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prediction history stored in Session State
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # Display prediction history
    for prediction in st.session_state['prediction_history']:
        st.markdown(f"""
        <div style="text-align: left;">
            <span style="background-color: #2b4fff; border-radius: 10px; padding: 10px; display: inline-block;">
                <strong>Text:</strong> {prediction['input']}
            </span>
            <span style="background-color: #3a3a3a; border-radius: 10px; padding: 10px; display: inline-block; margin-left: 10px;">
                <strong>Next Word:</strong> {prediction['next_word']}
            </span>
            <span style="font-size: 10px; color: #AAAAAA;">{prediction['timestamp']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Parameters with tooltips
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=config['prediction']['temperature'],
            step=0.1,
            help="Controls the randomness of the text generation. Higher values result in more varied responses."
        )
    with col2:
        top_k = st.slider(
            "Top K",
            min_value=10,
            max_value=100,
            value=config['prediction']['top_k'],
            step=10,
            help="Considers the top K most probable tokens for the next word."
        )
    with col3:
        top_p = st.slider(
            "Top P",
            min_value=0.5,
            max_value=1.0,
            value=config['prediction']['top_p'],
            step=0.05,
            help="Considers the cumulative probability of tokens for the next word."
        )
    
    col4, col5 = st.columns(2)
    with col4:
        penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=config['prediction']['penalty'],
            step=0.1,
            help="Penalizes repetition of already generated words."
        )
    with col5:
        debug_mode = st.checkbox(
            "Debug Mode",
            value=config['prediction']['debug_mode'],
            help="Enables additional debugging information."
        )
    
    # User input
    user_input = st.text_input("Text:", key="prediction_input")
    
    # Button to predict the next word
    predict_button = st.button("Predict", disabled=not user_input.strip())
    
    # Button to clear prediction history
    clear_button = st.button("Clear History")
    
    if clear_button:
        st.session_state['prediction_history'] = []
        st.experimental_rerun()
    
    if predict_button:
        if not user_input.strip():
            st.warning("Please enter some text to predict the next word.")
        else:
            try:
                with st.spinner("Generating the next word..."):
                    normalized_input = normalize_text(user_input)
    
                    next_word, debug_info = predict_next_word(
                        text=normalized_input,
                        model_path=model_path,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        penalty=penalty,
                        debug_mode=debug_mode
                    )
    
                    # Add to history
                    st.session_state['prediction_history'].append({
                        'input': normalized_input,
                        'next_word': next_word,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
    
                    # Continuous learning feedback
                    if 'continuous_learning' not in st.session_state:
                        st.session_state['continuous_learning'] = []
                    st.session_state['continuous_learning'].append({
                        'input': normalized_input,
                        'response': next_word
                    })
    
                    # Show debug info if enabled
                    if debug_info:
                        st.write("**Debug Info:**")
                        st.json(debug_info)
    
            except Exception as e:
                st.error(f"An error occurred while predicting the next word: {e}")

def run_chat_predict_interface():
    """
    Runs the Chatbot and Next Word Prediction interfaces using Streamlit.
    """
    tab1, tab2 = st.tabs(["🤖 Chatbot", "🔮 Predict Next Word"])
    
    with tab1:
        run_chatbot_interface()
    
    with tab2:
        run_predict_interface()

if __name__ == "__main__":
    st.warning("This module should not be run directly.")

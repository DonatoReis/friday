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

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Carregar configurações do config.yaml
from .utils import load_config

config = load_config('config/config.yaml')

# Função para carregar e armazenar o modelo e tokenizador usando cache
@st.cache_resource(show_spinner=False)
def get_model_and_tokenizer(model_path: str) -> Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]:
    """
    Carrega o modelo GPT-2 e o tokenizador a partir do caminho especificado.
    
    Args:
        model_path (str): Caminho para o diretório do modelo treinado.
    
    Returns:
        Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]: Modelo e tokenizador carregados ou None se falhar.
    """
    model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
    
    missing_files = [f for f in model_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"Modelo no diretório '{model_path}' está incompleto. Arquivos faltantes: {missing_files}")
        return None, None
    
    # Carregar o modelo e tokenizador
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    if model is None or tokenizer is None:
        st.error("Falha ao carregar o modelo ou tokenizador.")
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
    Gera uma resposta do modelo GPT-2 com base no prompt fornecido.
    
    Args:
        prompt (str): Texto de entrada para o modelo.
        model_path (str, optional): Caminho para o modelo treinado. Defaults to './fine-tune-gpt2'.
        temperature (float, optional): Parâmetro de temperatura para geração. Defaults to 1.0.
        top_k (int, optional): Parâmetro top_k para geração. Defaults to 50.
        top_p (float, optional): Parâmetro top_p para geração. Defaults to 0.95.
        penalty (float, optional): Parâmetro de penalização para repetição. Defaults to 1.0.
        debug_mode (bool, optional): Se True, retorna informações de depuração. Defaults to False.
    
    Returns:
        Tuple[str, Optional[Dict]]: Resposta gerada e dados de depuração se ativado.
    """
    model, tokenizer = get_model_and_tokenizer(model_path)
    
    if model is None or tokenizer is None:
        st.error("Modelo ou tokenizador não foram carregados corretamente.")
        return "", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.size(1) + 50,  # Gera até 50 tokens adicionais
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
        st.error(f"Erro ao gerar a resposta: {e}")
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
    Prevé a próxima palavra após o texto fornecido.
    
    Args:
        text (str): Texto de entrada.
        model_path (str, optional): Caminho para o modelo treinado. Defaults to './fine-tune-gpt2'.
        temperature (float, optional): Parâmetro de temperatura para geração. Defaults to 1.0.
        top_k (int, optional): Parâmetro top_k para geração. Defaults to 50.
        top_p (float, optional): Parâmetro top_p para geração. Defaults to 0.95.
        penalty (float, optional): Parâmetro de penalização para repetição. Defaults to 1.0.
        debug_mode (bool, optional): Se True, retorna informações de depuração. Defaults to False.
    
    Returns:
        Tuple[str, Optional[Dict]]: Próxima palavra prevista e dados de depuração se ativado.
    """
    model, tokenizer = get_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Modelo ou tokenizador não carregado corretamente.")
        return "", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=input_ids.size(1) + 1,  # Apenas uma palavra a mais
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
        st.error(f"Erro ao prever a próxima palavra: {e}")
        return "", None

def run_chatbot_interface():
    """
    Executa a interface do chatbot utilizando Streamlit.
    """
    st.header("🤖 Chatbot GPT-2")
    
    # Carregar modelo e tokenizador com cache
    model_path = config['training']['output_dir']
    model, tokenizer = get_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Modelo ou tokenizador não carregado corretamente.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Histórico de conversas armazenado no Session State
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Exibir histórico de mensagens
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            st.markdown(f"""
            <div style="text-align: right;">
                <span style="background-color: #3a3a3a; border-radius: 10px; padding: 10px; display: inline-block;">
                    <strong>Você:</strong> {chat['content']}
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
    
    # Parâmetros com tooltips
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=config['chatbot']['temperature'],
            step=0.1,
            help="Controla a aleatoriedade da geração de texto. Valores mais altos resultam em respostas mais variadas."
        )
    with col2:
        top_k = st.slider(
            "Top K",
            min_value=10,
            max_value=100,
            value=config['chatbot']['top_k'],
            step=10,
            help="Considera os top K tokens mais prováveis para a próxima palavra."
        )
    with col3:
        top_p = st.slider(
            "Top P",
            min_value=0.5,
            max_value=1.0,
            value=config['chatbot']['top_p'],
            step=0.05,
            help="Considera a probabilidade cumulativa dos tokens para a próxima palavra."
        )
    
    col4, col5 = st.columns(2)
    with col4:
        penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=config['chatbot']['penalty'],
            step=0.1,
            help="Penaliza a repetição de palavras já geradas."
        )
    with col5:
        debug_mode = st.checkbox(
            "Modo de Depuração",
            value=config['chatbot']['debug_mode'],
            help="Ativa informações de depuração adicionais."
        )
    
    # Entrada do usuário
    user_input = st.text_input("Você:", key="chat_input")
    
    # Botão para enviar a mensagem
    send_button = st.button("Enviar", disabled=not user_input.strip())
    
    # Botão para limpar o histórico da conversa
    clear_button = st.button("Limpar Conversa")
    
    if clear_button:
        st.session_state['chat_history'] = []
        st.experimental_rerun()
    
    if send_button:
        if not user_input.strip():
            st.warning("Por favor, insira um texto para conversar.")
        else:
            try:
                with st.spinner("O bot está pensando..."):
                    normalized_input = normalize_text(user_input)
    
                    # Análise de sentimentos
                    sentiment = analyze_sentiment(normalized_input, device=0 if device.type == 'cuda' else -1)
    
                    # Ajuste da resposta com base no sentimento
                    if sentiment == "NEGATIVE":
                        prompt = f"{normalized_input}\nResponda de forma reconfortante."
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
    
                    # Adicionar ao histórico
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
    
                    # Feedback para aprendizado contínuo
                    if 'continuous_learning' not in st.session_state:
                        st.session_state['continuous_learning'] = []
                    st.session_state['continuous_learning'].append({
                        'input': normalized_input,
                        'response': response
                    })
    
                    # Mostrar debug info se ativado
                    if debug_info:
                        st.write("**Debug Info:**")
                        st.json(debug_info)
    
            except Exception as e:
                st.error(f"Ocorreu um erro ao gerar a resposta: {e}")

def run_predict_interface():
    """
    Executa a interface de previsão de próxima palavra utilizando Streamlit.
    """
    st.header("🔮 Prever Próxima Palavra")
    
    # Carregar modelo e tokenizador com cache
    model_path = config['training']['output_dir']
    model, tokenizer = get_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Modelo ou tokenizador não carregado corretamente.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Histórico de previsões armazenado no Session State
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # Exibir histórico de previsões
    for prediction in st.session_state['prediction_history']:
        st.markdown(f"""
        <div style="text-align: left;">
            <span style="background-color: #2b4fff; border-radius: 10px; padding: 10px; display: inline-block;">
                <strong>Texto:</strong> {prediction['input']}
            </span>
            <span style="background-color: #3a3a3a; border-radius: 10px; padding: 10px; display: inline-block; margin-left: 10px;">
                <strong>Próxima Palavra:</strong> {prediction['next_word']}
            </span>
            <span style="font-size: 10px; color: #AAAAAA;">{prediction['timestamp']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Parâmetros com tooltips
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=config['prediction']['temperature'],
            step=0.1,
            help="Controla a aleatoriedade da geração de texto. Valores mais altos resultam em respostas mais variadas."
        )
    with col2:
        top_k = st.slider(
            "Top K",
            min_value=10,
            max_value=100,
            value=config['prediction']['top_k'],
            step=10,
            help="Considera os top K tokens mais prováveis para a próxima palavra."
        )
    with col3:
        top_p = st.slider(
            "Top P",
            min_value=0.5,
            max_value=1.0,
            value=config['prediction']['top_p'],
            step=0.05,
            help="Considera a probabilidade cumulativa dos tokens para a próxima palavra."
        )
    
    col4, col5 = st.columns(2)
    with col4:
        penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=config['prediction']['penalty'],
            step=0.1,
            help="Penaliza a repetição de palavras já geradas."
        )
    with col5:
        debug_mode = st.checkbox(
            "Modo de Depuração",
            value=config['prediction']['debug_mode'],
            help="Ativa informações de depuração adicionais."
        )
    
    # Entrada do usuário
    user_input = st.text_input("Texto:", key="prediction_input")
    
    # Botão para prever a próxima palavra
    predict_button = st.button("Prever", disabled=not user_input.strip())
    
    # Botão para limpar o histórico de previsões
    clear_button = st.button("Limpar Histórico")
    
    if clear_button:
        st.session_state['prediction_history'] = []
        st.experimental_rerun()
    
    if predict_button:
        if not user_input.strip():
            st.warning("Por favor, insira um texto para prever a próxima palavra.")
        else:
            try:
                with st.spinner("Gerando a próxima palavra..."):
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
    
                    # Adicionar ao histórico
                    st.session_state['prediction_history'].append({
                        'input': normalized_input,
                        'next_word': next_word,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
    
                    # Feedback para aprendizado contínuo
                    if 'continuous_learning' not in st.session_state:
                        st.session_state['continuous_learning'] = []
                    st.session_state['continuous_learning'].append({
                        'input': normalized_input,
                        'response': next_word
                    })
    
                    # Mostrar debug info se ativado
                    if debug_info:
                        st.write("**Debug Info:**")
                        st.json(debug_info)
    
            except Exception as e:
                st.error(f"Ocorreu um erro ao prever a próxima palavra: {e}")

def run_chat_predict_interface():
    """
    Executa as interfaces de Chatbot e Previsão de Próxima Palavra utilizando Streamlit.
    """
    tab1, tab2 = st.tabs(["🤖 Chatbot", "🔮 Prever Próxima Palavra"])
    
    with tab1:
        run_chatbot_interface()
    
    with tab2:
        run_predict_interface()

if __name__ == "__main__":
    st.warning("Este módulo não deve ser executado diretamente.")

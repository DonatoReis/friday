# modules/sentiment_analysis.py

import os
import torch
from typing import Optional, Dict

from transformers import pipeline
import streamlit as st
from dotenv import load_dotenv

from .utils import load_config, normalize_text

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Carregar configurações do config.yaml
config = load_config('config/config.yaml')

# Função para carregar e armazenar a pipeline de análise de sentimentos usando cache
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline(model_name: str, device: int) -> Optional[pipeline]:
    """
    Carrega a pipeline de análise de sentimentos do HuggingFace.

    Args:
        model_name (str): Nome ou caminho do modelo de análise de sentimentos.
        device (int): Dispositivo a ser utilizado (-1 para CPU, 0 para primeira GPU, etc.).

    Returns:
        Optional[pipeline]: Pipeline de análise de sentimentos ou None se falhar.
    """
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Erro ao carregar a pipeline de análise de sentimentos: {e}")
        return None

def analyze_sentiment(
    text: str,
    device: int = -1
) -> str:
    """
    Analisa o sentimento do texto fornecido.

    Args:
        text (str): Texto a ser analisado.
        device (int, optional): Dispositivo a ser utilizado (-1 para CPU, 0 para primeira GPU, etc.). Defaults to -1.

    Returns:
        str: Rótulo do sentimento ("POSITIVE", "NEGATIVE", ou "NEUTRAL").
    """
    if not text.strip():
        st.warning("Texto vazio fornecido para análise de sentimentos.")
        return "NEUTRAL"

    sentiment_pipeline = get_sentiment_pipeline(
        model_name=config['sentiment_analysis']['model'],
        device=device
    )

    if sentiment_pipeline is None:
        st.error("Pipeline de análise de sentimentos não disponível.")
        return "NEUTRAL"

    try:
        results = sentiment_pipeline(text)
        if not results:
            st.warning("Nenhum resultado retornado pela análise de sentimentos.")
            return "NEUTRAL"
        sentiment = results[0]['label']
        return sentiment
    except Exception as e:
        st.error(f"Erro durante a análise de sentimentos: {e}")
        return "NEUTRAL"

def run_sentiment_analysis_interface():
    """
    Executa a interface de análise de sentimentos utilizando Streamlit.
    """
    st.header("📈 Análise de Sentimentos")
    
    # Parâmetros com tooltips
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Idioma",
            ["Português", "Inglês", "Espanhol"],
            index=0,
            help="Selecione o idioma do texto para análise de sentimentos."
        )
    with col2:
        device_choice = st.selectbox(
            "Dispositivo",
            ["CPU", "GPU"],
            index=0,
            help="Escolha o dispositivo para executar a análise de sentimentos."
        )
    
    # Determinar o dispositivo
    device = 0 if device_choice == "GPU" and torch.cuda.is_available() else -1
    
    # Entrada do usuário
    user_input = st.text_area("Texto para Análise de Sentimentos:", height=150)
    
    # Botão para realizar a análise
    analyze_button = st.button("Analisar Sentimento")
    
    # Botão para limpar o texto de entrada
    clear_button = st.button("Limpar Texto")
    
    if clear_button:
        st.session_state['sentiment_input'] = ""
        st.experimental_rerun()
    
    if analyze_button:
        if not user_input.strip():
            st.warning("Por favor, insira um texto para análise de sentimentos.")
        else:
            try:
                with st.spinner("Analisando sentimento..."):
                    normalized_text = normalize_text(user_input)
                    sentiment = analyze_sentiment(normalized_text, device=device)
                    st.success(f"Sentimento Detectado: **{sentiment}**")
            except Exception as e:
                st.error(f"Ocorreu um erro durante a análise de sentimentos: {e}")

if __name__ == "__main__":
    st.warning("Este módulo não deve ser executado diretamente.")

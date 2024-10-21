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
    Configura o sistema de logging para a aplicação.
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
    Carrega as configurações a partir de um arquivo YAML.
    
    Args:
        config_path (str): Caminho para o arquivo de configuração YAML.
    
    Returns:
        Optional[dict]: Dicionário de configurações ou None se falhar.
    """
    if not os.path.exists(config_path):
        st.error(f"Arquivo de configuração '{config_path}' não encontrado.")
        logging.error(f"Arquivo de configuração '{config_path}' não encontrado.")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuração carregada a partir de '{config_path}'.")
            return config
    except yaml.YAMLError as e:
        st.error(f"Erro ao ler o arquivo de configuração: {e}")
        logging.error(f"Erro ao ler o arquivo de configuração: {e}")
        return None
    except UnicodeDecodeError as e:
        st.error(f"Erro de decodificação ao ler o arquivo de configuração: {e}")
        logging.error(f"Erro de decodificação ao ler o arquivo de configuração: {e}")
        return None

def load_model_and_tokenizer(model_path: str) -> Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]:
    """
    Carrega o modelo GPT-2 e o tokenizador a partir do caminho especificado.
    
    Args:
        model_path (str): Caminho para o diretório do modelo treinado.
    
    Returns:
        Tuple[Optional[GPT2LMHeadModel], Optional[GPT2TokenizerFast]]: Modelo e tokenizador carregados ou None se falhar.
    """
    if not os.path.exists(model_path):
        st.error(f"Caminho do modelo '{model_path}' não encontrado.")
        logging.error(f"Caminho do modelo '{model_path}' não encontrado.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        logging.info(f"Modelo e tokenizador carregados a partir de '{model_path}'.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou tokenizador: {e}")
        logging.error(f"Erro ao carregar o modelo ou tokenizador: {e}")
        return None, None

def normalize_text(text: str) -> str:
    """
    Normaliza o texto removendo espaços desnecessários e caracteres especiais.
    
    Args:
        text (str): Texto a ser normalizado.
    
    Returns:
        str: Texto normalizado.
    """
    normalized = ' '.join(text.strip().split())
    logging.debug(f"Texto normalizado: '{normalized}'")
    return normalized

def save_feedback(feedback: list, feedback_file: str = "feedback_data.txt") -> bool:
    """
    Salva o feedback coletado em um arquivo de texto para aprendizado contínuo.
    
    Args:
        feedback (list): Lista de dicionários contendo feedback.
        feedback_file (str, optional): Caminho para o arquivo de feedback. Defaults to "feedback_data.txt".
    
    Returns:
        bool: True se o salvamento for bem-sucedido, False caso contrário.
    """
    try:
        with open(feedback_file, 'a', encoding='utf-8') as f:
            for entry in feedback:
                f.write(f"{entry}\n")
        logging.info(f"Feedback salvo em '{feedback_file}'.")
        return True
    except Exception as e:
        st.error(f"Erro ao salvar o feedback: {e}")
        logging.error(f"Erro ao salvar o feedback: {e}")
        return False

def ensure_directory(path: str) -> bool:
    """
    Garante que um diretório exista. Se não existir, tenta criá-lo.
    
    Args:
        path (str): Caminho do diretório.
    
    Returns:
        bool: True se o diretório existir ou for criado com sucesso, False caso contrário.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Diretório garantido: '{path}'.")
        return True
    except Exception as e:
        st.error(f"Erro ao criar o diretório '{path}': {e}")
        logging.error(f"Erro ao criar o diretório '{path}': {e}")
        return False

def validate_model_files(model_path: str, required_files: list) -> bool:
    """
    Valida se todos os arquivos necessários do modelo estão presentes.
    
    Args:
        model_path (str): Caminho para o diretório do modelo.
        required_files (list): Lista de nomes de arquivos obrigatórios.
    
    Returns:
        bool: True se todos os arquivos estiverem presentes, False caso contrário.
    """
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"Arquivos faltantes no modelo '{model_path}': {missing_files}")
        logging.error(f"Arquivos faltantes no modelo '{model_path}': {missing_files}")
        return False
    logging.info(f"Todos os arquivos necessários estão presentes em '{model_path}'.")
    return True

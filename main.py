# main.py

import streamlit as st
import sys
import logging
from modules.interface import run_interface
from modules.utils import load_config
from dotenv import load_dotenv

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

def validate_config(config):
    """
    Valida se todas as configurações necessárias estão presentes.
    
    Args:
        config (dict): Dicionário de configurações carregadas.
    
    Returns:
        bool: True se todas as configurações estiverem presentes, False caso contrário.
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
        st.error(f"Seções de configuração faltantes: {missing_sections}")
        logging.error(f"Seções de configuração faltantes: {missing_sections}")
        return False
    return True

def main():
    """
    Função principal que executa a interface da aplicação.
    """
    try:
        # Configurar logging
        setup_logging()
        logging.info("Aplicação iniciada.")
        
        # Carregar variáveis de ambiente do arquivo .env, se existir
        load_dotenv()
        logging.info("Variáveis de ambiente carregadas.")
        
        # Carregar configurações do config.yaml
        config = load_config('config/config.yaml')
        if config is None:
            st.error("Falha ao carregar as configurações. Verifique o arquivo 'config/config.yaml'.")
            logging.error("Falha ao carregar as configurações.")
            sys.exit(1)
        
        # Validar configurações
        if not validate_config(config):
            logging.error("Configurações inválidas. Encerrando a aplicação.")
            sys.exit(1)

        # Executar a interface
        run_interface()
        
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
        logging.exception("Erro inesperado na aplicação.")
        sys.exit(1)
    finally:
        logging.info("Aplicação encerrada.")

if __name__ == "__main__":
    main()

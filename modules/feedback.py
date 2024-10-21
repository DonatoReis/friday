# modules/feedback.py

import os
import json
import logging
from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime

from .utils import load_config, ensure_directory, save_feedback, normalize_text
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Carregar configurações do config.yaml
config = load_config('config/config.yaml')

def collect_feedback(
    input_text: str,
    response_text: str,
    feedback: str,
    timestamp: Optional[str] = None
) -> Dict:
    """
    Coleta o feedback fornecido pelo usuário.
    
    Args:
        input_text (str): Texto de entrada fornecido pelo usuário.
        response_text (str): Resposta gerada pelo modelo.
        feedback (str): Feedback fornecido pelo usuário (e.g., "útil", "não útil", "sugestão").
        timestamp (Optional[str], optional): Timestamp da interação. Defaults to None.
    
    Returns:
        Dict: Dicionário contendo os dados de feedback.
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    feedback_entry = {
        "timestamp": timestamp,
        "input": normalize_text(input_text),
        "response": normalize_text(response_text),
        "feedback": feedback.lower().strip()
    }
    
    logging.debug(f"Feedback coletado: {feedback_entry}")
    return feedback_entry

def save_user_feedback(feedback_entry: Dict, feedback_file: str = "feedback_data.jsonl") -> bool:
    """
    Salva o feedback do usuário em um arquivo JSON Lines.
    
    Args:
        feedback_entry (Dict): Dicionário contendo os dados de feedback.
        feedback_file (str, optional): Caminho para o arquivo de feedback. Defaults to "feedback_data.jsonl".
    
    Returns:
        bool: True se o salvamento for bem-sucedido, False caso contrário.
    """
    try:
        with open(feedback_file, 'a', encoding='utf-8') as f:
            json.dump(feedback_entry, f, ensure_ascii=False)
            f.write('\n')
        logging.info(f"Feedback salvo em '{feedback_file}'.")
        return True
    except Exception as e:
        st.error(f"Erro ao salvar o feedback: {e}")
        logging.error(f"Erro ao salvar o feedback: {e}")
        return False

def load_feedback(feedback_file: str = "feedback_data.jsonl") -> List[Dict]:
    """
    Carrega todo o feedback armazenado.
    
    Args:
        feedback_file (str, optional): Caminho para o arquivo de feedback. Defaults to "feedback_data.jsonl".
    
    Returns:
        List[Dict]: Lista de dicionários contendo os dados de feedback.
    """
    feedback_list = []
    if not os.path.exists(feedback_file):
        logging.warning(f"Arquivo de feedback '{feedback_file}' não encontrado.")
        return feedback_list
    try:
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                feedback_list.append(json.loads(line))
        logging.info(f"Carregados {len(feedback_list)} registros de feedback de '{feedback_file}'.")
    except Exception as e:
        st.error(f"Erro ao carregar o feedback: {e}")
        logging.error(f"Erro ao carregar o feedback: {e}")
    return feedback_list

def process_feedback():
    """
    Interface para coletar e processar feedback dos usuários.
    """
    st.header("💬 Feedback do Usuário")
    
    # Entrada do usuário para fornecer feedback
    with st.form("feedback_form"):
        st.write("Por favor, forneça seu feedback sobre a resposta do modelo.")
        feedback_options = ["Útil", "Não Útil", "Sugestão"]
        selected_feedback = st.selectbox("Selecione o tipo de feedback:", feedback_options)
        suggestion = st.text_area("Se você tiver uma sugestão, por favor, descreva aqui:")
        submit_button = st.form_submit_button("Enviar Feedback")
    
    if submit_button:
        # Obter informações do histórico de chat
        if 'chat_history' not in st.session_state or not st.session_state['chat_history']:
            st.warning("Nenhuma interação disponível para fornecer feedback.")
            return
        
        last_interaction = st.session_state['chat_history'][-1]
        if last_interaction['role'] != 'bot':
            st.warning("Forneça feedback apenas para as respostas do bot.")
            return
        
        input_text = st.session_state['chat_history'][-2]['content'] if len(st.session_state['chat_history']) >=2 else ""
        response_text = last_interaction['content']
        
        # Incorporar a sugestão no feedback, se fornecida
        final_feedback = selected_feedback
        if suggestion.strip():
            final_feedback += f" - Sugestão: {normalize_text(suggestion)}"
        
        # Coletar o feedback
        feedback_entry = collect_feedback(
            input_text=input_text,
            response_text=response_text,
            feedback=final_feedback
        )
        
        # Garantir que o diretório para feedback exista
        feedback_dir = os.path.dirname(config['feedback']['feedback_file'])
        if not ensure_directory(feedback_dir):
            st.error("Não foi possível garantir o diretório para salvar o feedback.")
            return
        
        # Salvar o feedback
        if save_user_feedback(feedback_entry, feedback_file=config['feedback']['feedback_file']):
            st.success("Obrigado pelo seu feedback!")
            # Limpar o campo de sugestão após o envio
            if suggestion.strip():
                st.session_state['feedback_suggestion'] = ""
        else:
            st.error("Falha ao salvar o feedback. Por favor, tente novamente mais tarde.")

def analyze_feedback_statistics():
    """
    Analisa e exibe estatísticas básicas do feedback coletado.
    """
    st.header("📊 Estatísticas de Feedback")
    
    feedback_list = load_feedback(feedback_file=config['feedback']['feedback_file'])
    if not feedback_list:
        st.info("Nenhum feedback coletado ainda.")
        return
    
    # Contagem de tipos de feedback
    feedback_counts = {}
    for entry in feedback_list:
        label = entry['feedback'].split(" - ")[0].capitalize()
        feedback_counts[label] = feedback_counts.get(label, 0) + 1
    
    st.subheader("Distribuição de Feedback")
    st.bar_chart(feedback_counts)
    
    # Exibir sugestões
    suggestions = [entry['feedback'].split(" - ")[1] for entry in feedback_list if " - Sugestão:" in entry['feedback']]
    if suggestions:
        st.subheader("Sugestões dos Usuários")
        for idx, suggestion in enumerate(suggestions, 1):
            st.write(f"{idx}. {suggestion}")
    else:
        st.info("Nenhuma sugestão fornecida ainda.")

def run_feedback_interface():
    """
    Executa a interface de feedback utilizando Streamlit.
    """
    tab1, tab2 = st.tabs(["💬 Fornecer Feedback", "📊 Ver Estatísticas"])
    
    with tab1:
        process_feedback()
    
    with tab2:
        analyze_feedback_statistics()

def run_feedback_module():
    """
    Executa o módulo de feedback.
    """
    run_feedback_interface()

if __name__ == "__main__":
    st.warning("Este módulo não deve ser executado diretamente.")

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

# Importar para manipulação de variáveis de ambiente
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Função para carregar o arquivo de configuração
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
    model, tokenizer = load_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        st.error("Modelo ou tokenizador não carregado corretamente.")
        return None, None
    return model, tokenizer

# Configuração da página do Streamlit
def setup_page():
    st.set_page_config(
        page_title=config['general']['app_title'],
        page_icon=config['general']['page_icon'],
        layout=config['general']['layout'],
        initial_sidebar_state=config['general']['initial_sidebar_state'],
    )

# Função para aplicar CSS personalizado
def apply_custom_css():
    """
    Aplica estilos CSS personalizados para melhorar a interface do Streamlit.
    """
    custom_css = """
    <style>
    /* Fonte local "Inter" */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #171717;
        color: #FFFFFF;
    }

    .sidebar .sidebar-content {
        background-color: #1A1A1A;
    }

    /* Atualização de seletores CSS para maior robustez */
    div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        background-color: #1A1A1A;
    }

    .css-2trqyj {
        background-color: #2b2b2b;
    }

    .css-1lcbmhc {
        color: #2B4FFF;
    }

    /* Botões */
    .stButton>button {
        color: #FFFFFF;
        background-color: #2B4FFF;
    }

    /* Campos de Input */
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

    /* Estilização de balões de chat */
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

    /* Avatares */
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

# Função principal para a interface do Streamlit
def run_interface():
    # Configurar a página
    setup_page()

    # Aplicar CSS personalizado
    apply_custom_css()

    # Carregar token do HuggingFace de variável de ambiente
    huggingface_token = os.getenv(config['huggingface']['token_env_var'], "")
    
    # Configurações de Treinamento no Sidebar
    st.sidebar.header("Configurações de Treinamento")
    
    # Guia de Uso
    with st.sidebar.expander("📖 Guia de Uso"):
        st.markdown("""
        **GPT-2 Trainer & Chatbot**
        
        Este aplicativo permite treinar um modelo GPT-2 personalizado e interagir com ele através de um chatbot.
        
        **Limitações:**
        - Treinamentos longos podem levar tempo significativo, especialmente sem GPU.
        - A qualidade das respostas depende da qualidade e quantidade dos dados de treinamento.
        
        **Como Usar:**
        1. **Treinamento:**
            - Faça upload de um arquivo `.jsonl` ou `.txt` com os dados de treinamento.
            - Configure os parâmetros de treinamento conforme necessário.
            - Se desejar, ative o ajuste automático de hiperparâmetros com Optuna.
            - Clique em "Iniciar Treinamento" e aguarde a conclusão.
        2. **Chatbot:**
            - Após o treinamento, utilize a aba "🤖 Chatbot" para conversar com o modelo treinado.
        3. **Prever Próxima Palavra:**
            - Utilize a aba "🔮 Prever Próxima Palavra" para obter sugestões de palavras baseadas no texto fornecido.
        """)
    
    # Upload de arquivo de treinamento utilizando tempfile para gestão segura
    uploaded_file = st.sidebar.file_uploader(
        "Escolha o arquivo de treinamento (.jsonl ou .txt)", type=["jsonl", "txt"]
    )
    
    # Seleção do modelo
    selected_model = st.sidebar.selectbox(
        "Escolha o modelo GPT-2",
        config['training']['model_options'],
        index=config['training']['model_options'].index(config['training']['default_model'])
    )
    
    # Inserção do modelo personalizado do HuggingFace
    custom_model = st.sidebar.text_input(
        "Modelo Personalizado do HuggingFace",
        value=config['huggingface']['custom_model'],
        help="Insira o caminho ou nome do modelo personalizado do HuggingFace."
    )
    
    # Diretório de saída
    output_dir = st.sidebar.text_input(
        "Diretório para salvar o modelo treinado",
        value=config['training']['output_dir']
    )
    
    # Parâmetros de treinamento com validação
    num_epochs = st.sidebar.number_input(
        "Número de épocas",
        min_value=int(1),
        max_value=int(1000),
        value=int(config['training']['num_epochs']),
        step=int(1)
    )
    logging.info(f"num_epochs type: {type(config['training']['num_epochs'])}, value: {config['training']['num_epochs']}")
    batch_size = st.sidebar.number_input(
        "Tamanho do batch",
        min_value=int(1),
        max_value=int(128),
        value=int(config['training']['batch_size']),
        step=int(1)
    )
    logging.info(f"batch_size type: {type(config['training']['batch_size'])}, value: {config['training']['batch_size']}")
    max_length = st.sidebar.number_input(
        "Comprimento máximo dos tokens",
        min_value=int(16),
        max_value=int(2048),
        value=int(config['training']['max_length']),
        step=int(16)
    )
    logging.info(f"max_length type: {type(config['training']['max_length'])}, value: {config['training']['max_length']}")
    learning_rate = st.sidebar.number_input(
        "Taxa de aprendizado",
        min_value=float(1e-6),
        max_value=float(1e-3),
        value=float(config['training']['learning_rate']),
        step=float(1e-6),
        format="%.6f"
    )
    logging.info(f"learning_rate type: {type(config['training']['learning_rate'])}, value: {config['training']['learning_rate']}")
    validation_split = st.sidebar.slider(
        "Proporção de validação",
        0.0,
        0.5,
        config['training']['validation_split'],
        0.05
    )
    logging.info(f"validation_split type: {type(config['training']['validation_split'])}, value: {config['training']['validation_split']}")
    
    # Gradient Accumulation Steps
    gradient_accumulation_steps = st.sidebar.number_input(
        "Gradient Accumulation Steps",
        min_value=int(1),
        max_value=int(64),
        value=int(config['training']['gradient_accumulation_steps']),
        step=int(1),
        help="Número de passos para acumulação de gradientes antes de atualizar os pesos."
    )
    
    # Precisão reduzida (fp16) ou padrão (fp32)
    fp16_option = st.sidebar.selectbox(
        "Escolha a precisão do modelo",
        ["auto", "fp16", "fp32"],
        index=["auto", "fp16", "fp32"].index(config['training']['fp16_option'])
    )
    if fp16_option == "fp16":
        fp16 = True
    elif fp16_option == "fp32":
        fp16 = False
    else:
        fp16 = config['training']['fp16_option'] == "auto"
    
    # Opção de ajuste automático de hiperparâmetros
    automatic_hp_tuning = st.sidebar.checkbox(
        "Ativar ajuste automático de hiperparâmetros (Optuna)",
        value=config['training']['automatic_hp_tuning']
    )
    
    # Opção de estratégia de avaliação
    st.sidebar.header("Estratégia de Avaliação")
    use_eval = st.sidebar.checkbox(
        "Usar estratégia de avaliação",
        value=config['training']['use_eval_strategy']
    )
    if use_eval:
        eval_strategy = st.sidebar.selectbox(
            "Estratégia de avaliação",
            ["steps", "epoch"]
        )
        eval_steps = None
        if eval_strategy == "steps":
            eval_steps = st.sidebar.number_input(
                "Número de steps entre avaliações",
                min_value=int(100),
                max_value=int(10000),
                value=int(config['training']['eval_steps_hf']),
                step=int(100)
            )
        # Opção para carregar o melhor modelo ao final do treinamento
        load_best = st.sidebar.checkbox(
            "Carregar melhor modelo ao final do treinamento",
            value=config['training']['load_best_model_at_end']
        )
    else:
        eval_strategy = None
        eval_steps = None
        load_best = False  # Não carregar o melhor modelo se não houver avaliação
    
    # Botão para iniciar o treinamento
    if st.sidebar.button("Iniciar Treinamento"):
        if uploaded_file is None:
            st.sidebar.error("Por favor, faça upload de um arquivo de treinamento.")
        else:
            # Validar parâmetros de entrada
            if not (0 <= validation_split <= 0.5):
                st.sidebar.error("A proporção de validação deve estar entre 0 e 0.5.")
            elif not (1e-6 <= learning_rate <= 1e-3):
                st.sidebar.error("A taxa de aprendizado deve estar entre 1e-6 e 1e-3.")
            elif not (16 <= max_length <= 1024):
                st.sidebar.error("O comprimento máximo dos tokens deve estar entre 16 e 1024.")
            else:
                # Utilizar tempfile para gestão segura de arquivos temporários
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    data_path = temp_file.name
                
                # Barra de progresso (apenas para HuggingFace Trainer)
                progress_bar = st.sidebar.progress(0) if use_eval and eval_strategy == "steps" else None
                status_text = st.sidebar.empty()
                
                # Iniciar o treinamento com tratamento de erros
                try:
                    with st.spinner("Iniciando o treinamento..."):
                        if automatic_hp_tuning:
                            # Usar PyTorch Lightning com Optuna
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
                            # Usar HuggingFace Trainer
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
                                load_best_model_at_end=use_eval,  # Alteração importante
                                metric_for_best_model=config['training']['metric_for_best_model'],
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                use_eval_strategy=use_eval,
                                eval_strategy=eval_strategy,
                                eval_steps_hf=eval_steps if eval_strategy == "steps" else None,
                                progress_bar=progress_bar,
                                huggingface_token=huggingface_token,
                                custom_model=custom_model
                            )
                    
                    st.sidebar.success("Treinamento concluído com sucesso!")
                    
                    # Limpar o cache para recarregar o modelo treinado
                    get_model_and_tokenizer.clear()

                    # Verificar se todos os arquivos do modelo foram salvos corretamente
                    model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
                    if not all(os.path.exists(os.path.join(output_dir, f)) for f in model_files):
                        st.error(f"O modelo no diretório '{output_dir}' está incompleto. Arquivos faltantes: {[f for f in model_files if not os.path.exists(os.path.join(output_dir, f))]}")
                    else:
                        st.sidebar.success("Modelo salvo corretamente!")
                    
                    # Carregar o melhor modelo ou o último modelo salvo com base na estratégia de avaliação
                    if use_eval:
                        st.sidebar.info("Carregando o melhor modelo treinado com base na métrica.")
                        # Lógica para carregar o melhor modelo com base na métrica
                    else:
                        st.sidebar.info("Carregando o último modelo treinado.")
                        # Lógica para carregar o último modelo salvo

                    # Reiniciar a aplicação para atualizar a interface
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Ocorreu um erro durante o treinamento: {e}")
                finally:
                    # Remover o arquivo temporário após o treinamento
                    if os.path.exists(data_path):
                        os.remove(data_path)
    
    # Abas para Chat e Previsão
    tab1, tab2 = st.tabs(["🤖 Chatbot", "🔮 Prever Próxima Palavra"])
    
    with tab1:
        st.header("Chat com o Modelo GPT-2")
        model_path = output_dir if os.path.exists(output_dir) else config['training']['output_dir']
        
        # Verificar se o modelo está completo antes de prosseguir
        model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
        if not all(os.path.exists(os.path.join(model_path, f)) for f in model_files):
            st.error(f"O modelo no diretório '{model_path}' não está completo ou não foi treinado.")
            st.stop()  # Parar a execução da aba se o modelo não estiver disponível
        else:
            # Carregar modelo e tokenizador com cache
            model, tokenizer = get_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                st.error("Falha ao carregar o modelo ou tokenizador.")
                st.stop()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
    
                # Histórico de conversas armazenado no Session State
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
    
                # Exibir histórico de mensagens
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
    
                # Parâmetros com tooltips
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
                        <span class="tooltiptext">Controla a diversidade da geração. Valores mais altos resultam em respostas mais variadas.</span>
                    </div>
                    """, unsafe_allow_html=True)
    
                with col1:
                    penalty = st.slider("Repetition Penalty", 1.0, 2.0, config['chatbot']['penalty'], 0.1)
                with col2:
                    debug_mode = st.checkbox("Modo de Depuração", value=config['chatbot']['debug_mode'])
                with col3:
                    language = st.selectbox("Idioma", ["Português", "Inglês", "Espanhol"], index=0)
                with col4:
                    pass  # Espaço para alinhamento
    
                # Entrada do usuário
                user_input = st.text_input("Você:", key="chat_input")
    
                # Botão desabilitado quando o campo está vazio
                send_button = st.button("Enviar", disabled=not user_input.strip())
    
                # Botão para limpar o histórico da conversa
                clear_button = st.button("Limpar Conversa")
    
                if clear_button:
                    st.session_state['chat_history'] = []
                    # Não usar st.experimental_rerun para evitar loops
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
                                    prompt,
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

    with tab2:
        st.header("Prever Próxima Palavra")
        model_path = output_dir if os.path.exists(output_dir) else config['training']['output_dir']
        
        # Verificar se o modelo está completo antes de prosseguir
        model_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
        if not all(os.path.exists(os.path.join(model_path, f)) for f in model_files):
            st.error(f"O modelo no diretório '{model_path}' não está completo ou não foi treinado.")
            st.stop()
        else:
            # Carregar modelo e tokenizador com cache
            model, tokenizer = get_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                st.error("Falha ao carregar o modelo ou tokenizador.")
                st.stop()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
    
                # Entrada do usuário
                input_text = st.text_input("Insira o texto para prever a próxima palavra:", key="predict_input")
    
                # Botão para prever
                predict_button = st.button("Prever Próxima Palavra", disabled=not input_text.strip())
    
                if predict_button:
                    if not input_text.strip():
                        st.warning("Por favor, insira um texto para prever a próxima palavra.")
                    else:
                        try:
                            with st.spinner("Prevendo a próxima palavra..."):
                                next_word = predict_next_word(
                                    input_text,
                                    model_path=model_path,
                                    device=device
                                )
                                st.success(f"Próxima palavra prevista: **{next_word}**")
                        except Exception as e:
                            st.error(f"Ocorreu um erro ao prever a próxima palavra: {e}")

# Executar a interface
run_interface()

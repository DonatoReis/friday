# modules/quantization.py

import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import streamlit as st
from typing import Optional, Dict

from .utils import load_config, load_model_and_tokenizer
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Carregar configurações do config.yaml
config = load_config('config/config.yaml')

# Função para carregar e armazenar o modelo e tokenizador usando cache
@st.cache_resource(show_spinner=False)
def get_model_and_tokenizer(model_path: str) -> Optional[tuple]:
    """
    Carrega o modelo e o tokenizador a partir do caminho especificado.
    
    Args:
        model_path (str): Caminho para o diretório do modelo treinado.
    
    Returns:
        Optional[tuple]: Modelo e tokenizador carregados ou None em caso de falha.
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
    Aplica quantização ao modelo GPT-2 treinado.
    
    Args:
        model_path (str): Caminho para o diretório do modelo treinado.
        quantized_output_dir (str): Caminho para salvar o modelo quantizado.
        dtype (str, optional): Tipo de dado para quantização. Defaults to "torch.qint8".
        strategy (str, optional): Estratégia de quantização ("dynamic", "static", "hybrid"). Defaults to "dynamic".
        calibration_data (Optional[str], optional): Caminho para os dados de calibração se a estratégia for estática. Defaults to None.
    
    Returns:
        bool: True se a quantização foi bem-sucedida, False caso contrário.
    """
    try:
        model, tokenizer = get_model_and_tokenizer(model_path)
        if model is None or tokenizer is None:
            st.error("Modelo ou tokenizador não carregado corretamente.")
            return False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        st.info(f"Iniciando a quantização do modelo com dtype={dtype} e strategy={strategy}.")

        if strategy == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=getattr(torch, dtype)
            )
        elif strategy == "static":
            if calibration_data is None:
                st.error("Dados de calibração são necessários para a quantização estática.")
                return False
            
            # Carregar dados de calibração
            from datasets import load_dataset
            dataset = load_dataset('json', data_files={'calibration': calibration_data}, split='calibration')
            
            def calibrate(model, dataset, tokenizer, device, max_length=128):
                model.eval()
                with torch.no_grad():
                    for example in dataset:
                        inputs = tokenizer.encode(example['text'], return_tensors='pt', max_length=max_length, truncation=True).to(device)
                        model(inputs)
            
            # Preparar o modelo para quantização estática
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            calibrate(model, dataset, tokenizer, device)
            quantized_model = torch.quantization.convert(model, inplace=True)
        elif strategy == "hybrid":
            # Implementação de quantização híbrida
            st.info("Estratégia de quantização híbrida ainda não implementada.")
            return False
        else:
            st.error(f"Estratégia de quantização '{strategy}' não suportada.")
            return False

        # Criar o diretório de saída se não existir
        os.makedirs(quantized_output_dir, exist_ok=True)

        # Salvar o modelo quantizado
        quantized_model.save_pretrained(quantized_output_dir)
        tokenizer.save_pretrained(quantized_output_dir)

        # Verificar se todos os arquivos foram salvos
        required_files = ['tokenizer.json', 'config.json', 'vocab.json', 'merges.txt']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(quantized_output_dir, f))]
        if missing_files:
            st.error(f"O modelo quantizado no diretório '{quantized_output_dir}' está incompleto. Arquivos faltantes: {missing_files}")
            return False
        else:
            st.success(f"Quantização concluída com sucesso e modelo salvo em '{quantized_output_dir}'.")
            return True

    except Exception as e:
        st.error(f"Ocorreu um erro durante a quantização: {e}")
        return False

def run_quantization_interface():
    """
    Executa a interface de quantização utilizando Streamlit.
    """
    st.header("⚙️ Quantização do Modelo GPT-2")
    
    # Parâmetros com tooltips
    col1, col2 = st.columns(2)
    with col1:
        dtype = st.selectbox(
            "Tipo de Dado para Quantização",
            options=["torch.qint8", "torch.float16"],
            index=["torch.qint8", "torch.float16"].index(config['quantization']['dtype']),
            help="Escolha o tipo de dado para a quantização do modelo."
        )
    with col2:
        strategy = st.selectbox(
            "Estratégia de Quantização",
            options=["dynamic", "static", "hybrid"],
            index=["dynamic", "static", "hybrid"].index(config['quantization']['strategy']),
            help="Escolha a estratégia de quantização a ser utilizada."
        )
    
    # Upload de dados de calibração se a estratégia for estática
    calibration_file = None
    if strategy == "static":
        calibration_file = st.file_uploader(
            "Escolha o arquivo de calibração (.jsonl)",
            type=["jsonl"],
            help="Faça upload de um arquivo .jsonl contendo exemplos para calibração."
        )
    
    # Diretório de saída para o modelo quantizado
    quantized_output_dir = st.text_input(
        "Diretório para salvar o modelo quantizado",
        value=os.path.join(config['training']['output_dir'], "quantized"),
        help="Insira o caminho para salvar o modelo quantizado."
    )
    
    # Botão para iniciar a quantização
    if st.button("Iniciar Quantização"):
        model_path = config['training']['output_dir']
        
        # Validar presença do modelo
        required_files = ['tokenizer.json', 'config.json', 'vocab.json', 'merges.txt']
        if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
            st.error(f"O modelo no diretório '{model_path}' está incompleto ou não foi treinado.")
            return
        
        # Salvar o arquivo de calibração temporariamente se necessário
        calibration_path = None
        if strategy == "static" and calibration_file is not None:
            with open("calibration_data.jsonl", "wb") as f:
                f.write(calibration_file.getbuffer())
            calibration_path = "calibration_data.jsonl"
        
        # Iniciar quantização com tratamento de erros
        success = quantize_model(
            model_path=model_path,
            quantized_output_dir=quantized_output_dir,
            dtype=dtype,
            strategy=strategy,
            calibration_data=calibration_path
        )
        
        # Remover arquivo temporário de calibração
        if calibration_path and os.path.exists(calibration_path):
            os.remove(calibration_path)
        
        if success and config['quantization']['enabled']:
            st.info("Quantização e salvamento do modelo concluídos com sucesso.")
        elif not success:
            st.error("Falha na quantização do modelo. Verifique os logs para mais detalhes.")
    
def run_quantization_module():
    """
    Executa a interface de quantização.
    """
    run_quantization_interface()

if __name__ == "__main__":
    st.warning("Este módulo não deve ser executado diretamente.")

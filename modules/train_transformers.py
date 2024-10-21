# modules/train_transformers.py

import os
import time
from datetime import timedelta
from typing import Optional

import torch
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    logging as transformers_logging
)
from datasets import load_dataset

import streamlit as st
from .utils import normalize_text, load_model_and_tokenizer
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Callback personalizado para atualizar a barra de progresso do Streamlit durante o treinamento
class StreamlitProgressCallback(TrainerCallback):
    """
    Callback personalizado para atualizar a barra de progresso do Streamlit durante o treinamento com HuggingFace Trainer.
    """

    def __init__(self, progress_bar: Optional[st.delta_generator.DeltaGenerator], status_text: Optional[st.delta_generator.DeltaGenerator]):
        """
        Inicializa o callback com a barra de progresso e o texto de status.

        Args:
            progress_bar (Optional[st.delta_generator.DeltaGenerator]): Barra de progresso do Streamlit.
            status_text (Optional[st.delta_generator.DeltaGenerator]): Texto de status do Streamlit.
        """
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if self.progress_bar:
            self.progress_bar.progress(0)
        if self.status_text:
            self.status_text.text("Iniciando o treinamento...")

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar and args.num_train_epochs > 0:
            total_steps = args.num_train_epochs * (state.max_steps if state.max_steps > 0 else 1000)
            progress = state.global_step / total_steps
            self.progress_bar.progress(min(progress, 1.0))
            elapsed = time.time() - self.start_time
            if progress > 0:
                est_total = elapsed / progress
                est_remaining = est_total - elapsed
                est_time_str = str(timedelta(seconds=int(est_remaining)))
                if self.status_text:
                    self.status_text.text(f"Tempo restante estimado: {est_time_str}")
        return

def validate_parameters(
    data_path: str,
    model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    validation_split: float,
    save_steps: int,
    logging_steps: int,
    save_total_limit: int,
    fp16: bool,
    gradient_accumulation_steps: int,
    use_eval_strategy: bool,
    eval_strategy: Optional[str],
    eval_steps_hf: Optional[int],
    huggingface_token: str,
    custom_model: str
) -> bool:
    """
    Valida os parâmetros de entrada antes de iniciar o treinamento.

    Args:
        Todos os parâmetros necessários para o treinamento.

    Returns:
        bool: True se todos os parâmetros forem válidos, False caso contrário.
    """
    errors = []
    if not os.path.exists(data_path):
        errors.append(f"O arquivo de dados '{data_path}' não foi encontrado.")
    if num_epochs < 1:
        errors.append("O número de épocas deve ser pelo menos 1.")
    if batch_size < 1:
        errors.append("O tamanho do batch deve ser pelo menos 1.")
    if not (16 <= max_length <= 1024):
        errors.append("O comprimento máximo dos tokens deve estar entre 16 e 1024.")
    if not (1e-6 <= learning_rate <= 1e-3):
        errors.append("A taxa de aprendizado deve estar entre 1e-6 e 1e-3.")
    if not (0.0 <= validation_split <= 0.5):
        errors.append("A proporção de validação deve estar entre 0 e 0.5.")
    if save_steps < 100:
        errors.append("O número de steps para salvar deve ser pelo menos 100.")
    if logging_steps < 10:
        errors.append("O número de steps para log deve ser pelo menos 10.")
    if save_total_limit < 1:
        errors.append("O limite total de salvamentos deve ser pelo menos 1.")
    if gradient_accumulation_steps < 1:
        errors.append("O número de gradient accumulation steps deve ser pelo menos 1.")
    if use_eval_strategy:
        if eval_strategy not in ["steps", "epoch"]:
            errors.append("A estratégia de avaliação deve ser 'steps' ou 'epoch'.")
        if eval_strategy == "steps" and (eval_steps_hf is None or eval_steps_hf < 100):
            errors.append("O número de steps para avaliação deve ser pelo menos 100.")
    if huggingface_token and not isinstance(huggingface_token, str):
        errors.append("O token do HuggingFace deve ser uma string.")
    if custom_model and not isinstance(custom_model, str):
        errors.append("O caminho ou nome do modelo personalizado deve ser uma string.")

    if errors:
        for error in errors:
            st.error(error)
        return False
    return True

def train_with_transformers(
    data_path: str,
    model_name: str = 'gpt2',
    output_dir: str = './fine-tune-gpt2',
    num_epochs: int = 3,
    batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    validation_split: float = 0.1,
    save_steps: int = 500,
    logging_steps: int = 100,
    save_total_limit: int = 2,
    fp16: bool = False,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval_loss",
    gradient_accumulation_steps: int = 1,
    use_eval_strategy: bool = False,
    eval_strategy: Optional[str] = None,
    eval_steps_hf: Optional[int] = None,
    progress_bar: Optional[st.delta_generator.DeltaGenerator] = None,
    status_text: Optional[st.delta_generator.DeltaGenerator] = None,
    huggingface_token: str = "",
    custom_model: str = ""
):
    """
    Treina o modelo utilizando o Trainer da HuggingFace.

    Args:
        data_path (str): Caminho para o arquivo de dados de treinamento.
        model_name (str, optional): Nome do modelo pré-treinado ou caminho para modelo personalizado. Defaults to 'gpt2'.
        output_dir (str, optional): Diretório para salvar o modelo treinado. Defaults to './fine-tune-gpt2'.
        num_epochs (int, optional): Número de épocas. Defaults to 3.
        batch_size (int, optional): Tamanho do batch. Defaults to 32.
        max_length (int, optional): Comprimento máximo dos tokens. Defaults to 128.
        learning_rate (float, optional): Taxa de aprendizado. Defaults to 5e-5.
        validation_split (float, optional): Proporção de validação. Defaults to 0.1.
        save_steps (int, optional): Número de steps entre salvamentos. Defaults to 500.
        logging_steps (int, optional): Número de steps entre logs. Defaults to 100.
        save_total_limit (int, optional): Limite total de salvamentos. Defaults to 2.
        fp16 (bool, optional): Uso de precisão reduzida. Defaults to False.
        load_best_model_at_end (bool, optional): Carregar o melhor modelo no final. Defaults to False.
        metric_for_best_model (str, optional): Métrica para o melhor modelo. Defaults to "eval_loss".
        gradient_accumulation_steps (int, optional): Número de passos para acumulação de gradientes antes de atualizar os pesos. Defaults to 1.
        use_eval_strategy (bool, optional): Se True, usa estratégia de avaliação. Defaults to False.
        eval_strategy (Optional[str], optional): Estratégia de avaliação ('steps' ou 'epoch'). Defaults to None.
        eval_steps_hf (Optional[int], optional): Número de steps entre avaliações. Defaults to None.
        progress_bar (Optional[st.delta_generator.DeltaGenerator], optional): Barra de progresso. Defaults to None.
        status_text (Optional[st.delta_generator.DeltaGenerator], optional): Texto de status. Defaults to None.
        huggingface_token (str, optional): Token de autenticação do HuggingFace. Defaults to "".
        custom_model (str, optional): Caminho ou nome do modelo personalizado do HuggingFace. Defaults to "".
    """
    st.info("Iniciando o treinamento com HuggingFace Trainer...")
    
    start_time = time.time()

    # Validação de parâmetros
    if not validate_parameters(
        data_path, model_name, output_dir, num_epochs, batch_size, max_length,
        learning_rate, validation_split, save_steps, logging_steps,
        save_total_limit, fp16, gradient_accumulation_steps,
        use_eval_strategy, eval_strategy, eval_steps_hf,
        huggingface_token, custom_model
    ):
        st.error("Parâmetros de treinamento inválidos. Por favor, corrija os erros acima e tente novamente.")
        return

    try:
        # Configurar autenticação do HuggingFace se o token for fornecido
        if huggingface_token:
            os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
            from huggingface_hub import login
            login(token=huggingface_token)
            st.success("Autenticação no HuggingFace realizada com sucesso.")

        # Determinar o tipo de dataset
        if data_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files={'train': data_path}, split='train')
        elif data_path.endswith('.txt'):
            dataset = load_dataset('text', data_files={'train': data_path}, split='train')
        else:
            st.error("Formato de arquivo não suportado. Use '.jsonl' ou '.txt'.")
            return

        # Dividir o dataset em treinamento e validação
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None

        # Carregar tokenizador e modelo
        tokenizer = GPT2TokenizerFast.from_pretrained(custom_model if custom_model else model_name)
        model = GPT2LMHeadModel.from_pretrained(custom_model if custom_model else model_name)

        # Adicionar token de padding se necessário
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        # Função de tokenização
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        # Aplicar tokenização
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names if data_path.endswith('.jsonl') else ["text"]
        )
        if eval_dataset:
            tokenized_eval = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names if data_path.endswith('.jsonl') else ["text"]
            )
        else:
            tokenized_eval = None

        # Configurar o formato do dataset para PyTorch
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if tokenized_eval:
            tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Configurações de treinamento
        training_args_dict = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "logging_steps": logging_steps,
            "learning_rate": learning_rate,
            "fp16": fp16,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "eval_strategy": eval_strategy if use_eval_strategy else "no",
            "eval_steps": eval_steps_hf if use_eval_strategy and eval_strategy == "steps" else None,
            "report_to": "none",  # Desativa reportes para evitar conflitos com Streamlit
            #"use_cpu": not torch.cuda.is_available() if fp16 is None else not fp16,
            "no_cuda": not torch.cuda.is_available(),  # Substituição correta
        }

        training_args = TrainingArguments(**training_args_dict)

        # Inicializar Trainer com callbacks para atualizar a barra de progresso
        callbacks = []
        if use_eval_strategy and eval_strategy == "steps" and progress_bar:
            callbacks.append(StreamlitProgressCallback(progress_bar, status_text))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval if tokenized_eval else None,
            tokenizer=tokenizer,
            callbacks=callbacks
        )

        # Iniciar treinamento
        trainer.train()

        # Avaliar o modelo se houver conjunto de validação
        if eval_dataset:
            eval_result = trainer.evaluate()
            st.write(f"**Avaliação Final:** {eval_result}")

        # Salvar modelo e tokenizador
        st.info("Salvando o modelo e o tokenizador...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        st.success("Modelo e tokenizador salvos com sucesso.")

        # Verificar se todos os arquivos foram salvos
        required_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]

        if missing_files:
            st.error(f"O modelo no diretório '{output_dir}' está incompleto. Arquivos faltantes: {missing_files}")
        else:
            end_time = time.time()
            total_time = end_time - start_time
            st.success(f"Treinamento concluído e modelo salvo em '{output_dir}'. Tempo total: {str(timedelta(seconds=int(total_time)))}.")
      
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Verifique o caminho e tente novamente.")
    except ValueError as ve:
        st.error(f"Valor inválido: {ve}")
    except Exception as e:
        st.error(f"Ocorreu um erro durante o treinamento: {e}")

    if __name__ == "__main__":
        st.warning("Este módulo não deve ser executado diretamente.")
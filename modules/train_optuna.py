# modules/train_optuna.py

import os
import time
from datetime import timedelta
from typing import Tuple, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import streamlit as st
from .utils import normalize_text, load_config, load_model_and_tokenizer
from .sentiment_analysis import analyze_sentiment
from .quantization import quantize_model
from .train_transformers import train_with_transformers
from dotenv import load_dotenv
import optuna
import yaml

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Carregar configurações do config.yaml
config = load_config('config/config.yaml')

# Callback para atualizar a barra de progresso do Streamlit
class StreamlitProgressCallback(pl.Callback):
    """
    Callback personalizado para atualizar a barra de progresso do Streamlit durante o treinamento com PyTorch Lightning.
    """
    def __init__(self, progress_bar: Optional[st.delta_generator.DeltaGenerator], status_text: Optional[st.delta_generator.DeltaGenerator]):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        if self.progress_bar:
            self.progress_bar.progress(0)
        if self.status_text:
            self.status_text.text("Iniciando o treinamento...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.progress_bar:
            total_steps = config['training']['num_epochs'] * len(trainer.train_dataloader())
            current_step = trainer.global_step
            progress = current_step / total_steps
            self.progress_bar.progress(min(progress, 1.0))
            elapsed = time.time() - self.start_time
            if progress > 0:
                est_total = elapsed / progress
                est_remaining = est_total - elapsed
                est_time_str = str(timedelta(seconds=int(est_remaining)))
                if self.status_text:
                    self.status_text.text(f"Tempo restante estimado: {est_time_str}")

def validate_parameters(
    data_path: str,
    model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    validation_split: float,
    fp16: bool,
    custom_model: str
) -> bool:
    """
    Valida os parâmetros de entrada antes de iniciar o treinamento.

    Args:
        data_path (str): Caminho para o arquivo de dados de treinamento.
        model_name (str): Nome do modelo pré-treinado ou caminho para modelo personalizado.
        output_dir (str): Diretório para salvar o modelo treinado.
        num_epochs (int): Número de épocas para o treinamento.
        batch_size (int): Tamanho do batch durante o treinamento.
        max_length (int): Comprimento máximo dos tokens.
        learning_rate (float): Taxa de aprendizado.
        validation_split (float): Proporção de validação.
        fp16 (bool): Uso de precisão reduzida.
        custom_model (str): Caminho ou nome do modelo personalizado do HuggingFace.

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
    if custom_model and not isinstance(custom_model, str):
        errors.append("O caminho ou nome do modelo personalizado deve ser uma string.")
    if huggingface_token := os.getenv(config['huggingface']['token_env_var'], ""):
        if not isinstance(huggingface_token, str):
            errors.append("O token do HuggingFace deve ser uma string.")

    if errors:
        for error in errors:
            st.error(error)
        return False
    return True

# Definição do LightningModule personalizado
class GPT2LightningModule(pl.LightningModule):
    def __init__(self, model_name: str, learning_rate: float, max_length: int, batch_size: int):
        super(GPT2LightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        test_loss = outputs.loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def prepare_data(self, data_path: str, validation_split: float):
        """
        Prepara os dados para treinamento e validação.

        Args:
            data_path (str): Caminho para o arquivo de dados de treinamento.
            validation_split (float): Proporção de validação.
        """
        if data_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files={'train': data_path}, split='train')
        elif data_path.endswith('.txt'):
            dataset = load_dataset('text', data_files={'train': data_path}, split='train')
        else:
            raise ValueError("Formato de arquivo não suportado. Use '.jsonl' ou '.txt'.")

        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split)
            self.train_dataset = split_dataset['train']
            self.val_dataset = split_dataset['test']
        else:
            self.train_dataset = dataset
            self.val_dataset = None

        # Função de tokenização
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        # Aplicar tokenização
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names if data_path.endswith('.jsonl') else ["text"]
        )
        if self.val_dataset:
            self.val_dataset = self.val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names if data_path.endswith('.jsonl') else ["text"]
            )

        # Configurar o formato do dataset para PyTorch
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if self.val_dataset:
            self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset:
            return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
        else:
            return None

def train_with_optuna(
    data_path: str,
    model_name: str = 'gpt2',
    output_dir: str = './fine-tune-gpt2',
    num_epochs: int = 3,
    batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    validation_split: float = 0.1,
    fp16: bool = False,
    metric_for_best_model: str = "val_loss",
    automatic_hp_tuning: bool = False,
    use_eval_strategy: bool = False,
    eval_strategy: Optional[str] = None,
    eval_steps: Optional[int] = None,
    load_best_model_at_end: bool = False,
    huggingface_token: str = "",
    custom_model: str = "",
    quantization_enabled: bool = False
):
    """
    Treina o modelo utilizando PyTorch Lightning e Optuna para ajuste automático de hiperparâmetros.

    Args:
        data_path (str): Caminho para o arquivo de dados de treinamento.
        model_name (str, optional): Nome do modelo pré-treinado ou caminho para modelo personalizado. Defaults to 'gpt2'.
        output_dir (str, optional): Diretório para salvar o modelo treinado. Defaults to './fine-tune-gpt2'.
        num_epochs (int, optional): Número de épocas. Defaults to 3.
        batch_size (int, optional): Tamanho do batch. Defaults to 32.
        max_length (int, optional): Comprimento máximo dos tokens. Defaults to 128.
        learning_rate (float, optional): Taxa de aprendizado. Defaults to 5e-5.
        validation_split (float, optional): Proporção de validação. Defaults to 0.1.
        fp16 (bool, optional): Uso de precisão reduzida. Defaults to False.
        metric_for_best_model (str, optional): Métrica para o melhor modelo. Defaults to "val_loss".
        automatic_hp_tuning (bool, optional): Se True, realiza ajuste automático de hiperparâmetros. Defaults to False.
        use_eval_strategy (bool, optional): Se True, usa estratégia de avaliação. Defaults to False.
        eval_strategy (Optional[str], optional): Estratégia de avaliação ('steps' ou 'epoch'). Defaults to None.
        eval_steps (Optional[int], optional): Número de steps entre avaliações. Defaults to None.
        load_best_model_at_end (bool, optional): Se True, carrega o melhor modelo ao final do treinamento. Defaults to False.
        huggingface_token (str, optional): Token de autenticação do HuggingFace. Defaults to "".
        custom_model (str, optional): Caminho ou nome do modelo personalizado do HuggingFace. Defaults to "".
        quantization_enabled (bool, optional): Se True, aplica quantização após o treinamento. Defaults to False.
    """
    st.info("Iniciando o treinamento com PyTorch Lightning e Optuna...")

    start_time = time.time()

    # Validação de parâmetros
    if not validate_parameters(
        data_path, model_name, output_dir, num_epochs, batch_size, max_length,
        learning_rate, validation_split, fp16, custom_model
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

        # Definir a função objective dentro da função de treinamento para capturar variáveis locais
        def objective(trial):
            # Sugerir hiperparâmetros adicionais
            lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            batch_size_trial = trial.suggest_categorical('batch_size', [16, 32, 64])
            optimizer_choice = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])

            # Inicializar o LightningModule com os hiperparâmetros sugeridos
            model = GPT2LightningModule(
                model_name=model_name,
                learning_rate=lr,
                max_length=max_length,
                batch_size=batch_size_trial
            )

            # Preparar os dados
            model.prepare_data(data_path, validation_split)

            # Logger
            logger = TensorBoardLogger("tb_logs", name="gpt2_optuna")

            # Callback de pruning
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric_for_best_model)

            # Configurar o Trainer com o callback de pruning
            trainer_trial = PLTrainer(
                max_epochs=num_epochs,
                devices=1 if torch.cuda.is_available() else "auto",
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision=16 if fp16 else 32,
                logger=logger,
                callbacks=[pruning_callback],
                gradient_clip_val=1.0,
                enable_progress_bar=False
            )

            # Selecionar otimizador
            if optimizer_choice == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            elif optimizer_choice == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            trainer_trial.optimizers = optimizer

            # Treinar o modelo
            trainer_trial.fit(model)

            # Avaliar o modelo
            eval_result = trainer_trial.callback_metrics.get(metric_for_best_model)
            if eval_result is None:
                raise ValueError(f"Métrica '{metric_for_best_model}' não encontrada.")
            return eval_result.item()

        if automatic_hp_tuning:
            # Criar o estudo e otimizar com persistência
            study = optuna.create_study(direction="minimize", study_name="GPT2_Optuna_Study", storage="sqlite:///optuna_study.db", load_if_exists=True)
            study.optimize(objective, n_trials=20, timeout=600)  # Ajuste de n_trials e timeout conforme necessário

            st.success(f"Ajuste de hiperparâmetros concluído. Melhor {metric_for_best_model}: {study.best_trial.value}")
            st.write(f"Melhores parâmetros: {study.best_trial.params}")

            # Treinar o modelo final com os melhores hiperparâmetros
            best_lr = study.best_trial.params['learning_rate']
            best_batch_size = study.best_trial.params['batch_size']
            best_optimizer = study.best_trial.params['optimizer']

            # Inicializar o LightningModule com os melhores hiperparâmetros
            model = GPT2LightningModule(
                model_name=model_name,
                learning_rate=best_lr,
                max_length=max_length,
                batch_size=best_batch_size
            )

            # Preparar os dados
            model.prepare_data(data_path, validation_split)

            # Logger
            logger = TensorBoardLogger("tb_logs", name="gpt2_final")

            # Configurar o Trainer sem o callback de pruning
            trainer = PLTrainer(
                max_epochs=num_epochs,
                devices=1 if torch.cuda.is_available() else "auto",
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision=16 if fp16 else 32,
                logger=logger,
                gradient_clip_val=1.0,
                enable_progress_bar=False
            )

            # Selecionar otimizador
            if best_optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)
            elif best_optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)

            trainer.optimizers = optimizer

            # Treinar o modelo final
            trainer.fit(model)

            # Testar o modelo final
            trainer.test(model)

            # Salvar o modelo final
            model.model.save_pretrained(output_dir)
            model.tokenizer.save_pretrained(output_dir)

            # Verificar se todos os arquivos foram salvos
            required_files = ['pytorch_model.bin', 'config.json', 'vocab.json', 'merges.txt']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
            if missing_files:
                st.error(f"O modelo no diretório '{output_dir}' está incompleto. Arquivos faltantes: {missing_files}")
            else:
                end_time = time.time()
                total_time = end_time - start_time
                st.success(f"Treinamento concluído e modelo salvo em '{output_dir}'. Tempo total: {str(timedelta(seconds=int(total_time)))}.")

            # Aplicar quantização se habilitada
            if quantization_enabled:
                quantized_output_dir = os.path.join(output_dir, "quantized")
                quantize_model(
                    model_path=output_dir,
                    quantized_output_dir=quantized_output_dir,
                    dtype=torch.qint8
                )

        else:
            # Treinar sem ajuste automático de hiperparâmetros
            train_with_transformers(
                data_path=data_path,
                model_name=model_name,
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
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                use_eval_strategy=use_eval_strategy,
                eval_strategy=eval_strategy,
                eval_steps_hf=eval_steps if eval_strategy == "steps" else None,
                progress_bar=None,  # Já gerenciado no train_with_transformers
                status_text=None,
                huggingface_token=huggingface_token,
                custom_model=custom_model
            )
    
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Verifique o caminho e tente novamente.")
    except ValueError as ve:
        st.error(f"Valor inválido: {ve}")
    except Exception as e:
        st.error(f"Ocorreu um erro durante o treinamento: {e}")

    if __name__ == "__main__":
        st.warning("Este módulo não deve ser executado diretamente.")
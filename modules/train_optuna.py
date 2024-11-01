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

# Load environment variables from .env file, if it exists
load_dotenv()

# Load configurations from config.yaml
config = load_config('config/config.yaml')

# Callback to update the Streamlit progress bar
class StreamlitProgressCallback(pl.Callback):
    """
    Custom callback to update the Streamlit progress bar during training with PyTorch Lightning.
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
            self.status_text.text("Starting training...")

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
                    self.status_text.text(f"Estimated remaining time: {est_time_str}")

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
    Validates input parameters before starting training.

    Args:
        data_path (str): Path to the training data file.
        model_name (str): Name of the pre-trained model or path to a custom model.
        output_dir (str): Directory to save the trained model.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size during training.
        max_length (int): Maximum token length.
        learning_rate (float): Learning rate.
        validation_split (float): Validation proportion.
        fp16 (bool): Use of reduced precision.
        custom_model (str): Path or name of the custom HuggingFace model.

    Returns:
        bool: True if all parameters are valid, False otherwise.
    """
    errors = []
    if not os.path.exists(data_path):
        errors.append(f"The data file '{data_path}' was not found.")
    if num_epochs < 1:
        errors.append("The number of epochs must be at least 1.")
    if batch_size < 1:
        errors.append("The batch size must be at least 1.")
    if not (16 <= max_length <= 1024):
        errors.append("The maximum token length must be between 16 and 1024.")
    if not (1e-6 <= learning_rate <= 1e-3):
        errors.append("The learning rate must be between 1e-6 and 1e-3.")
    if not (0.0 <= validation_split <= 0.5):
        errors.append("The validation proportion must be between 0 and 0.5.")
    if custom_model and not isinstance(custom_model, str):
        errors.append("The custom model path or name must be a string.")
    if huggingface_token := os.getenv(config['huggingface']['token_env_var'], ""):
        if not isinstance(huggingface_token, str):
            errors.append("The HuggingFace token must be a string.")

    if errors:
        for error in errors:
            st.error(error)
        return False
    return True

# Definition of the custom LightningModule
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
        Prepares the data for training and validation.

        Args:
            data_path (str): Path to the training data file.
            validation_split (float): Validation proportion.
        """
        if data_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files={'train': data_path}, split='train')
        elif data_path.endswith('.txt'):
            dataset = load_dataset('text', data_files={'train': data_path}, split='train')
        else:
            raise ValueError("Unsupported file format. Use '.jsonl' or '.txt'.")

        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split)
            self.train_dataset = split_dataset['train']
            self.val_dataset = split_dataset['test']
        else:
            self.train_dataset = dataset
            self.val_dataset = None

        # Tokenization function
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        # Apply tokenization
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

        # Set dataset format for PyTorch
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
    Trains the model using PyTorch Lightning and Optuna for automatic hyperparameter tuning.

    Args:
        data_path (str): Path to the training data file.
        model_name (str, optional): Name of the pre-trained model or path to a custom model. Defaults to 'gpt2'.
        output_dir (str, optional): Directory to save the trained model. Defaults to './fine-tune-gpt2'.
        num_epochs (int, optional): Number of epochs. Defaults to 3.
        batch_size (int, optional): Batch size. Defaults to 32.
        max_length (int, optional): Maximum token length. Defaults to 128.
        learning_rate (float, optional): Learning rate. Defaults to 5e-5.
        validation_split (float, optional): Validation proportion. Defaults to 0.1.
        fp16 (bool, optional): Use of reduced precision. Defaults to False.
        metric_for_best_model (str, optional): Metric for the best model. Defaults to "val_loss".
        automatic_hp_tuning (bool, optional): If True, perform automatic hyperparameter tuning. Defaults to False.
        use_eval_strategy (bool, optional): If True, use evaluation strategy. Defaults to False.
        eval_strategy (Optional[str], optional): Evaluation strategy ('steps' or 'epoch'). Defaults to None.
        eval_steps (Optional[int], optional): Number of steps between evaluations. Defaults to None.
        load_best_model_at_end (bool, optional): If True, load the best model at the end of training. Defaults to False.
        huggingface_token (str, optional): HuggingFace authentication token. Defaults to "".
        custom_model (str, optional): Path or name of the custom HuggingFace model. Defaults to "".
        quantization_enabled (bool, optional): If True, apply quantization after training. Defaults to False.
    """
    st.info("Starting training with PyTorch Lightning and Optuna...")

    start_time = time.time()

    # Parameter validation
    if not validate_parameters(
        data_path, model_name, output_dir, num_epochs, batch_size, max_length,
        learning_rate, validation_split, fp16, custom_model
    ):
        st.error("Invalid training parameters. Please correct the errors above and try again.")
        return

    try:
        # Configure HuggingFace authentication if token is provided
        if huggingface_token:
            os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
            from huggingface_hub import login
            login(token=huggingface_token)
            st.success("Successfully authenticated with HuggingFace.")

        # Define the objective function within the training function to capture local variables
        def objective(trial):
            # Suggest additional hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            batch_size_trial = trial.suggest_categorical('batch_size', [16, 32, 64])
            optimizer_choice = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])

            # Initialize the LightningModule with suggested hyperparameters
            model = GPT2LightningModule(
                model_name=model_name,
                learning_rate=lr,
                max_length=max_length,
                batch_size=batch_size_trial
            )

            # Prepare the data
            model.prepare_data(data_path, validation_split)

            # Logger
            logger = TensorBoardLogger("tb_logs", name="gpt2_optuna")

            # Pruning callback
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric_for_best_model)

            # Configure the Trainer with the pruning callback
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

            # Select optimizer
            if optimizer_choice == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            elif optimizer_choice == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            trainer_trial.optimizers = optimizer

            # Train the model
            trainer_trial.fit(model)

            # Evaluate the model
            eval_result = trainer_trial.callback_metrics.get(metric_for_best_model)
            if eval_result is None:
                raise ValueError(f"Metric '{metric_for_best_model}' not found.")
            return eval_result.item()

        if automatic_hp_tuning:
            # Create the study and optimize with persistence
            study = optuna.create_study(direction="minimize", study_name="GPT2_Optuna_Study", storage="sqlite:///optuna_study.db", load_if_exists=True)
            study.optimize(objective, n_trials=20, timeout=600)  # Adjust n_trials and timeout as needed

            st.success(f"Hyperparameter tuning completed. Best {metric_for_best_model}: {study.best_trial.value}")
            st.write(f"Best parameters: {study.best_trial.params}")

            # Train the final model with the best hyperparameters
            best_lr = study.best_trial.params['learning_rate']
            best_batch_size = study.best_trial.params['batch_size']
            best_optimizer = study.best_trial.params['optimizer']

            # Initialize the LightningModule with the best hyperparameters
            model = GPT2LightningModule(
                model_name=model_name,
                learning_rate=best_lr,
                max_length=max_length,
                batch_size=best_batch_size
            )

            # Prepare the data
            model.prepare_data(data_path, validation_split)

            # Logger
            logger = TensorBoardLogger("tb_logs", name="gpt2_final")

            # Configure the Trainer without the pruning callback
            trainer = PLTrainer(
                max_epochs=num_epochs,
                devices=1 if torch.cuda.is_available() else "auto",
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision=16 if fp16 else 32,
                logger=logger,
                gradient_clip_val=1.0,
                enable_progress_bar=False
            )

            # Select optimizer
            if best_optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)
            elif best_optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)

            trainer.optimizers = optimizer

            # Train the final model
            trainer.fit(model)

            # Test the final model
            trainer.test(model)

            # Save the final model
            model.model.save_pretrained(output_dir)
            model.tokenizer.save_pretrained(output_dir)

            # Verify that all files have been saved
            required_files = ['pytorch_model.bin', 'config.json', 'vocab.json', 'merges.txt']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
            if missing_files:
                st.error(f"The model in directory '{output_dir}' is incomplete. Missing files: {missing_files}")
            else:
                end_time = time.time()
                total_time = end_time - start_time
                st.success(f"Training completed and model saved in '{output_dir}'. Total time: {str(timedelta(seconds=int(total_time)))}.")

            # Apply quantization if enabled
            if quantization_enabled:
                quantized_output_dir = os.path.join(output_dir, "quantized")
                quantize_model(
                    model_path=output_dir,
                    quantized_output_dir=quantized_output_dir,
                    dtype=torch.qint8
                )

        else:
            # Train without automatic hyperparameter tuning
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
                progress_bar=None,  # Already managed in train_with_transformers
                status_text=None,
                huggingface_token=huggingface_token,
                custom_model=custom_model
            )
    
    except FileNotFoundError:
        st.error("Data file not found. Please check the path and try again.")
    except ValueError as ve:
        st.error(f"Invalid value: {ve}")
    except Exception as e:
        st.error(f"An error occurred during training: {e}")

    if __name__ == "__main__":
        st.warning("This module should not be run directly.")

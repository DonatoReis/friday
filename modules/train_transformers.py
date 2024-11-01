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

# Load environment variables from .env file, if it exists
load_dotenv()

# Custom callback to update the Streamlit progress bar during training
class StreamlitProgressCallback(TrainerCallback):
    """
    Custom callback to update the Streamlit progress bar during training with HuggingFace Trainer.
    """

    def __init__(self, progress_bar: Optional[st.delta_generator.DeltaGenerator], status_text: Optional[st.delta_generator.DeltaGenerator]):
        """
        Initializes the callback with the progress bar and status text.

        Args:
            progress_bar (Optional[st.delta_generator.DeltaGenerator]): Streamlit progress bar.
            status_text (Optional[st.delta_generator.DeltaGenerator]): Streamlit status text.
        """
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if self.progress_bar:
            self.progress_bar.progress(0)
        if self.status_text:
            self.status_text.text("Starting training...")

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
                    self.status_text.text(f"Estimated remaining time: {est_time_str}")
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
    Validates input parameters before starting training.

    Args:
        All parameters required for training.

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
    if save_steps < 100:
        errors.append("The number of steps to save must be at least 100.")
    if logging_steps < 10:
        errors.append("The number of steps to log must be at least 10.")
    if save_total_limit < 1:
        errors.append("The total save limit must be at least 1.")
    if gradient_accumulation_steps < 1:
        errors.append("The number of gradient accumulation steps must be at least 1.")
    if use_eval_strategy:
        if eval_strategy not in ["steps", "epoch"]:
            errors.append("The evaluation strategy must be 'steps' or 'epoch'.")
        if eval_strategy == "steps" and (eval_steps_hf is None or eval_steps_hf < 100):
            errors.append("The number of steps for evaluation must be at least 100.")
    if huggingface_token and not isinstance(huggingface_token, str):
        errors.append("The HuggingFace token must be a string.")
    if custom_model and not isinstance(custom_model, str):
        errors.append("The custom model path or name must be a string.")

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
    Trains the model using the HuggingFace Trainer.

    Args:
        data_path (str): Path to the training data file.
        model_name (str, optional): Name of the pre-trained model or path to a custom model. Defaults to 'gpt2'.
        output_dir (str, optional): Directory to save the trained model. Defaults to './fine-tune-gpt2'.
        num_epochs (int, optional): Number of epochs. Defaults to 3.
        batch_size (int, optional): Batch size. Defaults to 32.
        max_length (int, optional): Maximum token length. Defaults to 128.
        learning_rate (float, optional): Learning rate. Defaults to 5e-5.
        validation_split (float, optional): Validation proportion. Defaults to 0.1.
        save_steps (int, optional): Number of steps between saves. Defaults to 500.
        logging_steps (int, optional): Number of steps between logs. Defaults to 100.
        save_total_limit (int, optional): Total save limit. Defaults to 2.
        fp16 (bool, optional): Use of reduced precision. Defaults to False.
        load_best_model_at_end (bool, optional): Load the best model at the end. Defaults to False.
        metric_for_best_model (str, optional): Metric for the best model. Defaults to "eval_loss".
        gradient_accumulation_steps (int, optional): Number of gradient accumulation steps before updating weights. Defaults to 1.
        use_eval_strategy (bool, optional): If True, use evaluation strategy. Defaults to False.
        eval_strategy (Optional[str], optional): Evaluation strategy ('steps' or 'epoch'). Defaults to None.
        eval_steps_hf (Optional[int], optional): Number of steps between evaluations. Defaults to None.
        progress_bar (Optional[st.delta_generator.DeltaGenerator], optional): Progress bar. Defaults to None.
        status_text (Optional[st.delta_generator.DeltaGenerator], optional): Status text. Defaults to None.
        huggingface_token (str, optional): HuggingFace authentication token. Defaults to "".
        custom_model (str, optional): Path or name of the custom HuggingFace model. Defaults to "".
    """
    st.info("Starting training with HuggingFace Trainer...")
    
    start_time = time.time()

    # Parameter validation
    if not validate_parameters(
        data_path, model_name, output_dir, num_epochs, batch_size, max_length,
        learning_rate, validation_split, save_steps, logging_steps,
        save_total_limit, fp16, gradient_accumulation_steps,
        use_eval_strategy, eval_strategy, eval_steps_hf,
        huggingface_token, custom_model
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

        # Determine the type of dataset
        if data_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files={'train': data_path}, split='train')
        elif data_path.endswith('.txt'):
            dataset = load_dataset('text', data_files={'train': data_path}, split='train')
        else:
            st.error("Unsupported file format. Use '.jsonl' or '.txt'.")
            return

        # Split the dataset into training and validation
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None

        # Load tokenizer and model
        tokenizer = GPT2TokenizerFast.from_pretrained(custom_model if custom_model else model_name)
        model = GPT2LMHeadModel.from_pretrained(custom_model if custom_model else model_name)

        # Add padding token if necessary
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        # Tokenization function
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        # Apply tokenization
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

        # Set dataset format for PyTorch
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if tokenized_eval:
            tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Training settings
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
            "report_to": "none",  # Disable reporting to avoid conflicts with Streamlit
            #"use_cpu": not torch.cuda.is_available() if fp16 is None else not fp16,
            "no_cuda": not torch.cuda.is_available(),  # Correct substitution
        }

        training_args = TrainingArguments(**training_args_dict)

        # Initialize Trainer with callbacks to update the progress bar
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

        # Start training
        trainer.train()

        # Evaluate the model if there is a validation set
        if eval_dataset:
            eval_result = trainer.evaluate()
            st.write(f"**Final Evaluation:** {eval_result}")

        # Save model and tokenizer
        st.info("Saving the model and tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        st.success("Model and tokenizer saved successfully.")

        # Verify that all files have been saved
        required_files = ['config.json', 'vocab.json', 'merges.txt', 'tokenizer.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]

        if missing_files:
            st.error(f"The model in directory '{output_dir}' is incomplete. Missing files: {missing_files}")
        else:
            end_time = time.time()
            total_time = end_time - start_time
            st.success(f"Training completed and model saved in '{output_dir}'. Total time: {str(timedelta(seconds=int(total_time)))}.")
      
    except FileNotFoundError:
        st.error("Data file not found. Please check the path and try again.")
    except ValueError as ve:
        st.error(f"Invalid value: {ve}")
    except Exception as e:
        st.error(f"An error occurred during training: {e}")

    if __name__ == "__main__":
        st.warning("This module should not be run directly.")

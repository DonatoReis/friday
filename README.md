# ⚠️ In Testing Phase

This project is currently in a testing phase. There are still known errors that need to be fixed. If you find any issues, please report them.

# 🤖 Friday Train Application

Welcome to the **Friday Train Application**! This project is an interactive web interface developed with **Streamlit** that allows users to interact with a custom GPT-2 model for various functionalities, including chat, next word prediction, sentiment analysis, and model quantization.

> **Main Goal**: The primary goal of this project is to train AI using **GPT-2** models and custom models.

## 📖 Table of Contents
- [📝 Description](#-description)
- [✨ Features](#-features)
- [💻 Technologies Used](#-technologies-used)
- [🚀 Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [🛠️ Usage](#%EF%B8%8F-usage)
- [🤝 Contribution](#-contribution)
- [📜 License](#-license)
- [📫 Contact](#-contact)
- [🙏 Acknowledgements](#-acknowledgements)

## 📝 Description

This project provides a user-friendly interface for interacting with **GPT-2** language models, allowing users to:

- **Chatbot**: Chat interactively with the model.
- **Next Word Prediction**: Predict the next word in a given text.
- **Sentiment Analysis**: Analyze the sentiment of provided texts.
- **Model Quantization**: Reduce the size of the model for efficient deployment.

Additionally, the project supports continuous training, hyperparameter tuning with **Optuna**, and feedback collection for ongoing model improvement.

## ✨ Features

- **Interactive Web Interface**: Built with Streamlit for easy interaction.
- **Custom Training**: Fine-tune the GPT-2 model with custom datasets.
- **Hyperparameter Optimization**: Use Optuna to find the best settings.
- **Sentiment Analysis**: Integration with sentiment analysis pipelines.
- **Model Quantization**: Reduce model size to improve performance and lower costs.
- **User Feedback**: Collect and analyze feedback for continuous learning.

## 💻 Technologies Used

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [Optuna](https://optuna.org/)
- [PyYAML](https://pyyaml.org/)
- [Dotenv](https://github.com/theskumar/python-dotenv)

## 🚀 Installation

Follow the steps below to set up the environment and install the necessary dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/DonatoReis/friday.git
cd friday
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to isolate project dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔧 Configuration

### 1. Environment Variables Setup

Create a `.env` file in the project's root to store sensitive variables like the **Hugging Face** authentication tokens.

```bash
touch .env
```

Open the `.env` file and add the following:

```env
HUGGINGFACE_TOKEN=your_token_here
```

> **Note**: Replace `your_token_here` with your **Hugging Face** access token.

### 2. Configuration File

Edit the `config/config.yaml` file to adjust the settings as needed. This file controls parameters such as model paths, training hyperparameters, quantization options, and more.

```yaml
general:
  app_title: "ChatGPT Streamlit Application"

training:
  output_dir: "./fine-tune-gpt2"
  save_steps: 500
  logging_steps: 100
  gradient_accumulation_steps: 1

optimization:
  fp16_option: "auto"  # Options: "auto", "fp16", "fp32"

chatbot:
  temperature: 1.0
  top_k: 50
  top_p: 0.95
  penalty: 1.0
  debug_mode: False

prediction:
  temperature: 1.0
  top_k: 50
  top_p: 0.95
  penalty: 1.0
  debug_mode: False

sentiment_analysis:
  model: "pysentimiento/bertweet-pt-sentiment"

huggingface:
  token_env_var: "HUGGINGFACE_TOKEN"

quantization:
  enabled: True
  dtype: "torch.qint8"  # Options: "torch.qint8", "torch.float16"
  strategy: "dynamic"   # Options: "dynamic", "static", "hybrid"
  feedback_file: "feedback_data.jsonl"

other:
  # Add other settings as needed
```

## 🛠️ Usage

After installation and configuration, you can start the **Streamlit** application by running the following command:

```bash
streamlit run main.py
```

This will open the application in your default web browser. From there, you can access different functionalities through the available tabs:

- **Chatbot**: Interact with the GPT-2 model.
- **Next Word Prediction**: Input text to predict the next word.
- **Sentiment Analysis**: Analyze the sentiment of provided texts.
- **Model Quantization**: Reduce model size for efficient deployment.
- **User Feedback**: Provide feedback on model responses.

## 🤝 Contribution

Contributions are welcome! Feel free to open **issues** and submit **pull requests**.

## 📜 License

This project is licensed under the **MIT** License. See the LICENSE file for more details.

## 📫 Contact

If you have any questions or suggestions, feel free to reach out:

- **Email**: [your-email@example.com](mailto:your-email@example.com)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com)
- **GitHub**: [your-username](https://github.com/your-username)

## 🙏 Acknowledgements

- **OpenAI** for providing the language models.
- **Hugging Face** for their NLP tools and models.
- **Streamlit** for making interactive web interfaces easy to build.
- To all contributors and users who help improve this project!

# 🤖 ChatGPT Streamlit Application

Bem-vindo ao ChatGPT Streamlit Application! Este projeto é uma interface web interativa desenvolvida com Streamlit que permite aos usuários interagirem com um modelo GPT-2 customizado para diversas funcionalidades, incluindo chat, previsão de próxima palavra, análise de sentimentos e quantização de modelos.

## 📖 Índice

- [Descrição](#descrição)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Contribuição](#contribuição)
- [Licença](#licença)
- [Contato](#contato)
- [Agradecimentos](#agradecimentos)

## 📝 Descrição

Este projeto fornece uma interface amigável para interagir com modelos de linguagem GPT-2, permitindo que os usuários:

- **Chatbot:** Conversar com o modelo de forma interativa.
- **Previsão de Próxima Palavra:** Prever a próxima palavra em um texto fornecido.
- **Análise de Sentimentos:** Analisar o sentimento de textos inseridos.
- **Quantização de Modelo:** Reduzir o tamanho do modelo para implantação eficiente.

Além disso, o projeto suporta treinamento contínuo, ajuste de hiperparâmetros com Optuna e coleta de feedback para aprimoramento contínuo do modelo.

## ✨ Funcionalidades

- **Interface Web Interativa:** Desenvolvida com Streamlit para facilitar a interação.
- **Treinamento Personalizado:** Fine-tuning do modelo GPT-2 com datasets personalizados.
- **Otimização de Hiperparâmetros:** Utilização do Optuna para encontrar as melhores configurações.
- **Análise de Sentimentos:** Integração com pipelines de análise de sentimentos.
- **Quantização de Modelo:** Redução do tamanho do modelo para melhorar a performance e reduzir custos.
- **Feedback do Usuário:** Coleta e análise de feedback para aprendizado contínuo.

## 💻 Tecnologias Utilizadas

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [Optuna](https://optuna.org/)
- [PyYAML](https://pyyaml.org/)
- [Dotenv](https://github.com/theskumar/python-dotenv)

## 🚀 Instalação

Siga os passos abaixo para configurar o ambiente e instalar as dependências necessárias.

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/chatgpt-streamlit-app.git
cd chatgpt-streamlit-app

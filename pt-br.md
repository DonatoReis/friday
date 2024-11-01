# ⚠️ Em Fase de Testes

Este projeto está em fase de testes. Ainda existem erros conhecidos que precisam ser corrigidos. Caso encontre algum problema, por favor, reporte-o.

# 🤖 Friday Train Application

Bem-vindo ao **Friday Train Application**! Este projeto é uma interface web interativa desenvolvida com **Streamlit** que permite aos usuários interagirem com um modelo GPT-2 customizado para diversas funcionalidades, incluindo chat, previsão de próxima palavra, análise de sentimentos e quantização de modelos.

> **Objetivo Principal**: Este projeto tem como objetivo principal treinar IA utilizando modelos **GPT-2** e modelos personalizados.

## 📖 Índice
- [📝 Descrição](#-descrição)
- [✨ Funcionalidades](#-funcionalidades)
- [💻 Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [🚀 Instalação](#-instalação)
- [🔧 Configuração](#-configuração)
- [🛠️ Uso](#%EF%B8%8F-uso)
- [🤝 Contribuição](#-contribuição)
- [📜 Licença](#-licença)
- [📫 Contato](#-contato)
- [🙏 Agradecimentos](#-agradecimentos)

## 📝 Descrição

Este projeto fornece uma interface amigável para interagir com modelos de linguagem **GPT-2**, permitindo que os usuários:

- **Chatbot**: Conversar com o modelo de forma interativa.
- **Previsão de Próxima Palavra**: Prever a próxima palavra em um texto fornecido.
- **Análise de Sentimentos**: Analisar o sentimento de textos inseridos.
- **Quantização de Modelo**: Reduzir o tamanho do modelo para implantação eficiente.

Além disso, o projeto suporta treinamento contínuo, ajuste de hiperparâmetros com **Optuna** e coleta de feedback para aprimoramento contínuo do modelo.

## ✨ Funcionalidades

- **Interface Web Interativa**: Desenvolvida com Streamlit para facilitar a interação.
- **Treinamento Personalizado**: Fine-tuning do modelo GPT-2 com datasets personalizados.
- **Otimização de Hiperparâmetros**: Utilização do Optuna para encontrar as melhores configurações.
- **Análise de Sentimentos**: Integração com pipelines de análise de sentimentos.
- **Quantização de Modelo**: Redução do tamanho do modelo para melhorar a performance e reduzir custos.
- **Feedback do Usuário**: Coleta e análise de feedback para aprendizado contínuo.

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
git clone https://github.com/DonatoReis/friday.git
cd friday
```

### 2. Crie um Ambiente Virtual

É recomendado utilizar um ambiente virtual para isolar as dependências do projeto.

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔧 Configuração

### 1. Configuração de Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto para armazenar variáveis sensíveis, como tokens de autenticação do **Hugging Face**.

```bash
touch .env
```

Abra o arquivo `.env` e adicione o seguinte:

```env
HUGGINGFACE_TOKEN=seu_token_aqui
```

> **Nota**: Substitua `seu_token_aqui` pelo seu token de acesso do **Hugging Face**.

### 2. Arquivo de Configuração

Edite o arquivo `config/config.yaml` para ajustar as configurações conforme suas necessidades. Este arquivo controla parâmetros como caminhos de modelos, hiperparâmetros de treinamento, opções de quantização, entre outros.

```yaml
general:
  app_title: "ChatGPT Streamlit Application"

training:
  output_dir: "./fine-tune-gpt2"
  save_steps: 500
  logging_steps: 100
  gradient_accumulation_steps: 1

optimization:
  fp16_option: "auto"  # Opções: "auto", "fp16", "fp32"

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
  dtype: "torch.qint8"  # Opções: "torch.qint8", "torch.float16"
  strategy: "dynamic"   # Opções: "dynamic", "static", "hybrid"
  feedback_file: "feedback_data.jsonl"

other:
  # Adicione outras configurações conforme necessário
```

## 🛠️ Uso

Após a instalação e configuração, você pode iniciar a aplicação **Streamlit** executando o seguinte comando:

```bash
streamlit run main.py
```

Isso abrirá a aplicação no seu navegador padrão. A partir daí, você poderá acessar diferentes funcionalidades através das abas disponíveis:

- **Chatbot**: Interaja com o modelo GPT-2.
- **Previsão de Próxima Palavra**: Insira um texto para prever a próxima palavra.
- **Análise de Sentimentos**: Analise o sentimento de textos fornecidos.
- **Quantização do Modelo**: Reduza o tamanho do modelo para implantação eficiente.
- **Feedback do Usuário**: Forneça feedback sobre as respostas do modelo.

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir **issues** e enviar **pull requests**.

## 📜 Licença

Este projeto está licenciado sob a Licença **MIT**. Veja o arquivo LICENSE para mais detalhes.

## 📫 Contato

Se você tiver qualquer dúvida ou sugestão, sinta-se à vontade para entrar em contato:

- **Email**: [seu-email@example.com](mailto:seu-email@example.com)
- **LinkedIn**: [Seu Perfil no LinkedIn](https://www.linkedin.com)
- **GitHub**: [seu-usuario](https://github.com/seu-usuario)

## 🙏 Agradecimentos

- **OpenAI** por fornecer os modelos de linguagem.
- **Hugging Face** por suas ferramentas e modelos de NLP.
- **Streamlit** por facilitar a criação de interfaces web interativas.
- A todos os contribuidores e usuários que ajudam a melhorar este projeto!

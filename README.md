# 🌾 Smart Agriculture ML

Este projeto utiliza técnicas de Machine Learning para oferecer recomendações inteligentes de culturas agrícolas. O sistema analisa variáveis cruciais do solo e clima — como níveis de Nitrogênio (N), Fósforo (P), Potássio (K), Temperatura, Umidade, pH e Pluviosidade — para determinar a cultura mais eficiente para o cultivo.

## 🧠 Abordagem Técnica
O projeto foi estruturado para suportar um fluxo de tomada de decisão baseado em dados:
- **Modelo de Recomendação (KNN):** Utilizamos o algoritmo *K-Nearest Neighbors* para classificar a melhor cultura baseada na similaridade de características com dados históricos, permitindo uma recomendação robusta e rápida.
- **Análise de Padrões (Apriori):** Implementamos o algoritmo *Apriori* para identificar regras de associação entre as variáveis ambientais, ajudando a compreender quais condições climáticas e de solo costumam ocorrer juntas, enriquecendo a análise dos dados.

## 🚀 Tecnologias Utilizadas
- **Python 3.13**
- **Scikit-learn**: Para a implementação do modelo KNN.
- **MLxtend**: Para a análise de associação com o algoritmo Apriori.
- **Pandas**: Para manipulação e tratamento dos dados.

## 📂 Estrutura do Projeto
```text
├── data/               # Dataset (Crop_recommendation.csv)
├── notebooks/          # Análises exploratórias e protótipos
├── src/                # Código fonte modularizado
│   ├── data_processing.py
│   └── models.py
├── main.py             # Script principal de execução
└── requirements.txt    # Dependências do projeto
```

## 🛠 Como rodar o projeto
Siga estes 4 passos para configurar e executar o projeto em sua máquina local:

### 1. Clone o repositório
Abra o seu terminal e execute:
```bash
git clone https://github.com/rodrigo-nery99/smart-agriculture-ml.git
cd smart-agriculture-ml
```

### 2. Configure o ambiente virtual
Crie e ative o ambiente para isolar as dependências do projeto:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as dependências
Com o ambiente virtual ativado, instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 4. Execute o projeto
Finalmente, execute o script principal:
```bash
python main.py
```

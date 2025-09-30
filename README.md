# 🔍 Predição de Status de Crimes

Aplicação Streamlit para prever a probabilidade de um crime ser **Concluído** ou **Arquivado** usando modelo de regressão logística.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://regressao-geraldo.streamlit.app)

## 📊 Dataset

O dataset contém 5.000 ocorrências de crimes com as seguintes características:
- **Status**: Concluído, Arquivado, Em Investigação (excluído do modelo)
- **Features**: Tipo de crime, modus operandi, arma utilizada, quantidade de vítimas/suspeitos

## 🚀 Deploy no Streamlit Cloud

### Opção 1: Deploy Automático
1. Faça fork deste repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte sua conta GitHub
4. Selecione este repositório
5. Clique em "Deploy!"

### Opção 2: Executar Localmente
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run app.py
```

## 🎯 Funcionalidades

### 📈 Análise Exploratória
- Distribuição do status das ocorrências
- Análise por tipo de crime
- Taxa de conclusão por bairro
- Estatísticas descritivas

### 🤖 Modelo de Predição
- **Regressão Logística** ou **Random Forest**
- Codificação automática de variáveis categóricas
- Normalização de dados numéricos
- Métricas de performance (acurácia, precisão, recall)

### 🔮 Interface de Predição
- Seleção interativa de características do crime
- Cálculo de probabilidades em tempo real
- Visualizações das probabilidades
- Interpretação dos resultados

### 📊 Visualizações
- Gráficos de pizza e barras
- Matriz de confusão
- Importância das features
- Análise geográfica por bairro

## 🎛️ Como Usar

1. **Explore os dados** na seção de análise exploratória
2. **Escolha o modelo** (Regressão Logística ou Random Forest)
3. **Configure as características** do crime na interface de predição
4. **Clique em "Prever Status"** para obter as probabilidades
5. **Analise os resultados** e a interpretação fornecida

## 📋 Features do Modelo

- **Variáveis Categóricas**: Bairro, tipo de crime, modus operandi, arma, sexo do suspeito, órgão responsável
- **Variáveis Numéricas**: Quantidade de vítimas/suspeitos, idade, coordenadas geográficas, data
- **Target**: Status binário (1 = Concluído, 0 = Arquivado)

## 🔍 Interpretação dos Resultados

- **> 60% Conclusão**: Alta probabilidade de o caso ser concluído
- **> 60% Arquivamento**: Alta probabilidade de o caso ser arquivado  
- **Probabilidades equilibradas**: Caso pode ter qualquer desfecho

## 📁 Estrutura do Projeto

```
regressaoGeraldo/
├── app.py                              # Aplicação Streamlit principal
├── requirements.txt                    # Dependências Python
├── dataset_ocorrencias_delegacia_5.csv # Dataset de crimes
└── README.md                          # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Interface web interativa
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Modelos de machine learning
- **Plotly**: Visualizações interativas
- **Matplotlib/Seaborn**: Gráficos adicionais

## 📈 Métricas de Performance

O modelo fornece:
- Acurácia geral
- Matriz de confusão
- Relatório de classificação detalhado
- Importância das features (Random Forest)

---

**Desenvolvido para análise preditiva de status de crimes** 🚔

# 🔍 Sistema Integrado de Predição de Status de Crimes

Sistema completo de machine learning que combina **análise supervisionada** (Regressão Logística + Random Forest) e **não supervisionada** (Clustering K-Means + Detecção de Anomalias) para prever a probabilidade de um crime ser **Concluído** ou **Arquivado**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://regressao-geraldo.streamlit.app)

## 📊 Dataset

O sistema utiliza um dataset padrão com **5.000 ocorrências** de crimes com as seguintes características:
- **Status**: Concluído (33.1%), Arquivado (32.8%), Em Investigação (34.1% - excluído do modelo)
- **Features Selecionadas**: 
  - Tipo de crime (categórica)
  - Modus operandi (categórica) 
  - Arma utilizada (categórica)
  - Quantidade de vítimas (numérica)
  - Quantidade de suspeitos (numérica)
- **Localização**: Bairros de Recife com coordenadas geográficas reais

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

### 🧠 Análise Supervisionada
- **Regressão Logística** com normalização StandardScaler
- **Random Forest** com 100 estimadores
- Codificação automática de variáveis categóricas
- Métricas de performance (acurácia, precisão, matriz de confusão)
- Importância das features (Random Forest)

### 🧩 Análise Não Supervisionada
- **K-Means Clustering** com 6 clusters
- **Detecção de Anomalias** (Isolation Forest + LOF)
- Identificação de casos "fora do padrão"
- Filtros avançados para análise de anomalias
- Integração com modelo supervisionado

### 🔮 Interface de Predição Integrada
- Seleção interativa de características do crime
- Predição simultânea de status e cluster
- Cálculo de probabilidades em tempo real
- Análise do cluster predito com características dominantes
- Interpretação contextualizada dos resultados

### 🗺️ Mapa de Hotspots
- Mapa interativo com Folium
- Marcadores por bairro com estatísticas reais
- Cores baseadas na taxa de conclusão
- Popups informativos com métricas detalhadas

### 📈 Análise Exploratória
- Distribuição do status das ocorrências
- Análise por tipo de crime
- Estatísticas descritivas
- Visualizações interativas com Plotly

### 📊 Visualizações Avançadas
- Gráficos de pizza e barras interativos
- Matriz de confusão
- Box plots de probabilidades por cluster
- Análise de anomalias por tipo de crime
- Métricas por cluster com acurácia

## 🎛️ Como Usar

### **Fluxo Principal:**
1. **Explore os dados** na seção de análise exploratória
2. **Escolha o modelo** (Regressão Logística ou Random Forest)
3. **Configure as características** do crime na interface de predição
4. **Clique em "Prever Status e Cluster"** para obter resultados integrados
5. **Analise os resultados** com interpretação contextualizada

### **Funcionalidades Avançadas:**
6. **🗺️ Mapa de Hotspots**: Explore a distribuição geográfica dos crimes
7. **🚨 Detecção de Anomalias**: Identifique casos "fora do padrão"
8. **🧩 Análise de Clusters**: Compreenda padrões de crimes similares
9. **📊 Métricas por Cluster**: Analise performance por grupo de casos

## 📋 Arquitetura do Sistema

### **Features Selecionadas (5 características):**
- **Tipo de Crime** (categórica): Homicídio, Roubo, Furto, Tráfico, etc.
- **Modus Operandi** (categórica): Assalto a Mão Armada, Fraude Online, etc.
- **Arma Utilizada** (categórica): Arma de Fogo, Faca, Objeto Contundente, etc.
- **Quantidade de Vítimas** (numérica): 0-4 vítimas
- **Quantidade de Suspeitos** (numérica): 0-4 suspeitos

### **Target:**
- **Status Binário**: 1 = Concluído, 0 = Arquivado
- **Filtro**: Casos "Em Investigação" são excluídos do modelo

## 🔍 Interpretação dos Resultados

### **Probabilidades de Status:**
- **> 60% Conclusão**: Alta probabilidade de o caso ser concluído
- **> 60% Arquivamento**: Alta probabilidade de o caso ser arquivado  
- **Probabilidades equilibradas**: Caso pode ter qualquer desfecho

### **Análise de Cluster:**
- **Cluster Predito**: Grupo de casos com características similares
- **Taxa de Conclusão do Cluster**: Percentual histórico de casos resolvidos no grupo
- **Características Dominantes**: Padrões mais comuns no cluster (tipo de crime, modus operandi, etc.)

### **Insights Integrados:**
- **Consistência**: Compare probabilidade do modelo com taxa histórica do cluster
- **Contexto**: Entenda o padrão de casos similares para melhor interpretação
- **Anomalias**: Identifique casos que fogem dos padrões esperados

## 📁 Estrutura do Projeto

```
regressaoGeraldo/
├── app.py                              # Aplicação Streamlit principal
├── requirements.txt                    # Dependências Python otimizadas
├── dataset_ocorrencias_delegacia_5.csv # Dataset padrão (5.000 casos)
├── RELATORIO_SISTEMA.md               # Documentação técnica completa
└── README.md                          # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Interface web interativa
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Scikit-learn**: Modelos de machine learning
- **Plotly**: Visualizações interativas
- **Folium**: Mapas interativos

## 📈 Métricas de Performance

### **Modelos Supervisionados:**
- **Acurácia**: ~75-80% (Regressão Logística e Random Forest)
- **Precisão**: Balanceada entre classes
- **Matriz de Confusão**: Análise de erros
- **Relatório de Classificação**: Métricas detalhadas por classe

### **Modelos Não Supervisionados:**
- **Clustering**: 6 clusters com distribuição balanceada
- **Detecção de Anomalias**: ~10% dos casos identificados como "fora do padrão"
- **Integração**: Análise conjunta supervisionado + não supervisionado

## 🎯 Aplicações Práticas

### **Para Investigadores:**
- Priorização de casos com maior probabilidade de conclusão
- Identificação de padrões em crimes similares
- Detecção de casos anômalos que requerem atenção especial
- Análise geográfica para otimização de recursos

### **Para Gestores:**
- Planejamento estratégico baseado em dados
- Métricas de performance por tipo de crime
- Identificação de hotspots para alocação de recursos
- Relatórios automatizados para stakeholders

---

**Sistema Integrado de Machine Learning para Análise Preditiva de Crimes** 🚔🔍

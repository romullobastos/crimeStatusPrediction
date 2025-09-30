import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Predição de Status de Crimes",
    page_icon="🔍",
    layout="wide"
)

# Título principal
st.title("🔍 Predição de Status de Crimes")
st.markdown("**Modelo de Regressão Logística para prever probabilidade de conclusão/arquivamento de crimes**")
st.markdown("*Features: Tipo de Crime, Modus Operandi, Arma, Quantidade de Vítimas/Suspeitos*")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_ocorrencias_delegacia_5.csv')
    
    # Converter data_ocorrencia para datetime
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
    
    return df

df = load_data()

# Preparar dados para o modelo
def prepare_data(df):
    # Selecionar features categóricas e numéricas (removendo idade_suspeito)
    categorical_features = ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']
    numerical_features = ['quantidade_vitimas', 'quantidade_suspeitos']
    
    # Codificar variáveis categóricas
    le_dict = {}
    df_encoded = df.copy()
    
    for feature in categorical_features:
        le = LabelEncoder()
        df_encoded[feature + '_encoded'] = le.fit_transform(df_encoded[feature].astype(str))
        le_dict[feature] = le
    
    # Selecionar features para o modelo
    feature_columns = [f + '_encoded' for f in categorical_features] + numerical_features
    X = df_encoded[feature_columns]
    y = df_encoded['status_binario']
    
    return X, y, le_dict, feature_columns

# Filtrar dados (excluir "Em Investigação")
df_filtered = df[df['status_investigacao'] != 'Em Investigação'].copy()
df_filtered['status_binario'] = (df_filtered['status_investigacao'] == 'Concluído').astype(int)

# Preparar dados
X, y, le_dict, feature_columns = prepare_data(df_filtered)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar modelo
st.header("🤖 Modelo de Predição")

model_choice = st.selectbox("Escolha o modelo:", ["Regressão Logística", "Random Forest"])

if model_choice == "Regressão Logística":
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

# Interface de predição
st.header("🎯 Predição de Status")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Selecione as características do crime:")
    
    # Inputs para predição (apenas features relevantes)
    tipo_crime = st.selectbox("Tipo de Crime", df['tipo_crime'].unique())
    modus_operandi = st.selectbox("Modus Operandi", df['descricao_modus_operandi'].unique())
    arma = st.selectbox("Arma Utilizada", df['arma_utilizada'].unique())

with col2:
    st.subheader("Informações numéricas:")
    
    qtd_vitimas = st.slider("Quantidade de Vítimas", 0, 4, 1)
    qtd_suspeitos = st.slider("Quantidade de Suspeitos", 0, 4, 1)

# Botão de predição
if st.button("🔮 Prever Status", type="primary"):
    # Preparar dados de entrada (apenas features relevantes)
    input_data = {
        'tipo_crime': tipo_crime,
        'descricao_modus_operandi': modus_operandi,
        'arma_utilizada': arma,
        'quantidade_vitimas': qtd_vitimas,
        'quantidade_suspeitos': qtd_suspeitos
    }
    
    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Codificar variáveis categóricas
    for feature in ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']:
        input_df[feature + '_encoded'] = le_dict[feature].transform(input_df[feature].astype(str))
    
    # Selecionar features
    X_input = input_df[feature_columns]
    
    # Fazer predição
    if model_choice == "Regressão Logística":
        X_input_scaled = scaler.transform(X_input)
        proba = model.predict_proba(X_input_scaled)[0]
    else:
        proba = model.predict_proba(X_input)[0]
    
    # Exibir resultados
    st.subheader("🎯 Resultado da Predição")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probabilidade de Arquivamento", f"{proba[0]:.1%}")
        st.metric("Probabilidade de Conclusão", f"{proba[1]:.1%}")
    
    with col2:
        # Gráfico de barras das probabilidades
        fig_proba = px.bar(x=['Arquivado', 'Concluído'], y=proba, 
                          title="Probabilidades de Status",
                          labels={'x': 'Status', 'y': 'Probabilidade'})
        fig_proba.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_proba, use_container_width=True)
    
    # Interpretação
    if proba[1] > 0.6:
        st.success("✅ **Alta probabilidade de CONCLUSÃO** - O caso tem características que favorecem a conclusão da investigação.")
    elif proba[0] > 0.6:
        st.warning("⚠️ **Alta probabilidade de ARQUIVAMENTO** - O caso tem características que podem levar ao arquivamento.")
    else:
        st.info("🤔 **Probabilidades equilibradas** - O caso pode ter qualquer um dos dois desfechos.")

# Sidebar para filtros
st.sidebar.header("📊 Filtros e Configurações")

st.sidebar.metric("Total de Ocorrências", len(df_filtered))
st.sidebar.metric("Concluídos", len(df_filtered[df_filtered['status_binario'] == 1]))
st.sidebar.metric("Arquivados", len(df_filtered[df_filtered['status_binario'] == 0]))

# Análise exploratória
st.header("📈 Análise Exploratória dos Dados")

col1, col2 = st.columns(2)

with col1:
    # Distribuição do status
    status_counts = df_filtered['status_investigacao'].value_counts()
    fig_pie = px.pie(values=status_counts.values, names=status_counts.index, 
                     title="Distribuição do Status das Ocorrências")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Status por tipo de crime
    status_crime = pd.crosstab(df_filtered['tipo_crime'], df_filtered['status_investigacao'])
    fig_bar = px.bar(status_crime, title="Status por Tipo de Crime", 
                     labels={'value': 'Quantidade', 'index': 'Tipo de Crime'})
    st.plotly_chart(fig_bar, use_container_width=True)

# Métricas do modelo
st.header("📊 Performance do Modelo")

accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Acurácia", f"{accuracy:.3f}")
with col2:
    st.metric("Precisão", f"{accuracy_score(y_test, y_pred):.3f}")
with col3:
    st.metric("Amostras de Teste", len(y_test))

# Matriz de confusão
st.subheader("📊 Matriz de Confusão")
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                   labels=dict(x="Predito", y="Real", color="Quantidade"),
                   x=['Arquivado', 'Concluído'], y=['Arquivado', 'Concluído'])
st.plotly_chart(fig_cm, use_container_width=True)

# Relatório de classificação
st.subheader("📋 Relatório de Classificação")
report = classification_report(y_test, y_pred, target_names=['Arquivado', 'Concluído'], output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Feature importance (apenas para Random Forest)
if model_choice == "Random Forest":
    st.subheader("🔍 Importância das Features")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = px.bar(feature_importance.head(10), x='importance', y='feature',
                           title="Top 10 Features Mais Importantes",
                           orientation='h')
    st.plotly_chart(fig_importance, use_container_width=True)

# Análise por tipo de crime
st.header("🔍 Análise por Tipo de Crime")

crime_analysis = df_filtered.groupby('tipo_crime')['status_binario'].agg(['count', 'sum', 'mean']).reset_index()
crime_analysis.columns = ['Tipo_Crime', 'Total_Casos', 'Concluidos', 'Taxa_Conclusao']
crime_analysis['Taxa_Conclusao'] = crime_analysis['Taxa_Conclusao'].round(3)

# Gráfico de taxa de conclusão por tipo de crime
fig_crime = px.bar(crime_analysis, x='Tipo_Crime', y='Taxa_Conclusao',
                   title="Taxa de Conclusão por Tipo de Crime",
                   labels={'Taxa_Conclusao': 'Taxa de Conclusão'})
fig_crime.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig_crime, use_container_width=True)

# Tabela com estatísticas por tipo de crime
st.subheader("📊 Estatísticas por Tipo de Crime")
st.dataframe(crime_analysis.sort_values('Taxa_Conclusao', ascending=False))

# Footer
st.markdown("---")
st.markdown("**Desenvolvido com Streamlit** | Modelo de Regressão Logística para Predição de Status de Crimes")
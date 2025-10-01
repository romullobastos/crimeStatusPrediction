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
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Predição de Status de Crimes",
    page_icon="🔍",
    layout="wide"
)

# Título principal
st.title("🔍 Predição de Status de Crimes com Análise de Clusters")
st.markdown("**Modelo Integrado: Regressão Logística + Clustering para prever probabilidade de conclusão/arquivamento**")
st.markdown("*Features Alinhadas: Tipo de Crime, Modus Operandi, Arma, Quantidade de Vítimas/Suspeitos (Ambos os modelos)*")

# Organização visual
st.markdown("---")
st.header("🧠 Análise Supervisionada (Modelo)")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_ocorrencias_delegacia_5.csv')
    
    # Converter data_ocorrencia para datetime
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
    
    return df

df = load_data()

# Preparar dados para o modelo (REGRESSÃO LOGÍSTICA)
def prepare_data(df):
    # Selecionar features categóricas e numéricas (alinhadas com clustering)
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

# Criar modelo de clustering (MESMAS FEATURES DA REGRESSÃO)
def create_clustering_model(df):
    """Cria modelo de clustering com as MESMAS features da regressão"""
    # Selecionar features para clustering (EXATAMENTE as mesmas da regressão)
    categorical_features_cluster = ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']
    numerical_features_cluster = ['quantidade_vitimas', 'quantidade_suspeitos']
    
    # Codificar variáveis categóricas para clustering
    df_cluster = df.copy()
    le_cluster = {}
    
    for feature in categorical_features_cluster:
        le = LabelEncoder()
        df_cluster[feature + '_encoded'] = le.fit_transform(df_cluster[feature].astype(str))
        le_cluster[feature] = le
    
    # Preparar dados para clustering (MESMAS features da regressão)
    cluster_columns = [f + '_encoded' for f in categorical_features_cluster] + numerical_features_cluster
    X_cluster = df_cluster[cluster_columns]
    
    # Normalizar dados para clustering
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    # Aplicar K-Means (usando 6 clusters como no modelo original)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    
    # Adicionar clusters ao dataframe
    df_cluster['cluster'] = clusters
    
    return df_cluster, kmeans, scaler_cluster, le_cluster, cluster_columns

# Criar modelo de clustering
df_with_clusters, kmeans_model, scaler_cluster, le_cluster, cluster_columns = create_clustering_model(df_filtered)

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
if st.button("🔮 Prever Status e Cluster", type="primary"):
    # Preparar dados de entrada (MESMAS features para ambos os modelos)
    input_data = {
        'tipo_crime': tipo_crime,
        'descricao_modus_operandi': modus_operandi,
        'arma_utilizada': arma,
        'quantidade_vitimas': qtd_vitimas,
        'quantidade_suspeitos': qtd_suspeitos
    }
    
    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Codificar variáveis categóricas para regressão
    for feature in ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']:
        input_df[feature + '_encoded'] = le_dict[feature].transform(input_df[feature].astype(str))
    
    # Selecionar features para regressão
    X_input = input_df[feature_columns]
    
    # Fazer predição de status
    if model_choice == "Regressão Logística":
        X_input_scaled = scaler.transform(X_input)
        proba = model.predict_proba(X_input_scaled)[0]
    else:
        proba = model.predict_proba(X_input)[0]
    
    # Usar os mesmos dados para clustering (features já alinhadas)
    input_df_cluster = input_df.copy()
    
    # Codificar variáveis categóricas para clustering
    for feature in ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']:
        input_df_cluster[feature + '_encoded'] = le_cluster[feature].transform(input_df_cluster[feature].astype(str))
    
    # Selecionar features para clustering
    X_input_cluster = input_df_cluster[cluster_columns]
    X_input_cluster_scaled = scaler_cluster.transform(X_input_cluster)
    
    # Fazer predição de cluster
    predicted_cluster = kmeans_model.predict(X_input_cluster_scaled)[0]
    
    # Exibir resultados
    st.subheader("🎯 Resultado da Predição")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probabilidade de Arquivamento", f"{proba[0]:.1%}")
        st.metric("Probabilidade de Conclusão", f"{proba[1]:.1%}")
    
    with col2:
        st.metric("Cluster Predito", f"Cluster {predicted_cluster}")
        
        # Análise do cluster predito
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == predicted_cluster]
        cluster_completion_rate = cluster_data['status_binario'].mean() * 100
        st.metric("Taxa de Conclusão do Cluster", f"{cluster_completion_rate:.1f}%")
    
    with col3:
        # Gráfico de barras das probabilidades
        fig_proba = px.bar(x=['Arquivado', 'Concluído'], y=proba, 
                          title="Probabilidades de Status",
                          labels={'x': 'Status', 'y': 'Probabilidade'})
        fig_proba.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_proba, use_container_width=True)
    
    # Análise do cluster predito
    st.subheader(f"📊 Análise do Cluster {predicted_cluster}")
    
    cluster_analysis = cluster_data.groupby('status_investigacao').size()
    cluster_analysis_pct = cluster_data['status_investigacao'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribuição no Cluster:**")
        st.dataframe(cluster_analysis_pct.round(1))
    
    with col2:
        # Características dominantes do cluster
        st.write("**Características Dominantes:**")
        tipo_dominante = cluster_data['tipo_crime'].mode()[0]
        modus_dominante = cluster_data['descricao_modus_operandi'].mode()[0]
        arma_dominante = cluster_data['arma_utilizada'].mode()[0]
        
        st.write(f"• **Tipo de Crime:** {tipo_dominante}")
        st.write(f"• **Modus Operandi:** {modus_dominante}")
        st.write(f"• **Arma:** {arma_dominante}")
        st.write(f"• **Vítimas médias:** {cluster_data['quantidade_vitimas'].mean():.1f}")
        st.write(f"• **Suspeitos médios:** {cluster_data['quantidade_suspeitos'].mean():.1f}")
    
    # Interpretação
    if proba[1] > 0.6:
        st.success("✅ **Alta probabilidade de CONCLUSÃO** - O caso tem características que favorecem a conclusão da investigação.")
    elif proba[0] > 0.6:
        st.warning("⚠️ **Alta probabilidade de ARQUIVAMENTO** - O caso tem características que podem levar ao arquivamento.")
    else:
        st.info("🤔 **Probabilidades equilibradas** - O caso pode ter qualquer um dos dois desfechos.")
    
    # Interpretação do cluster
    if cluster_completion_rate > 60:
        st.info(f"🔍 **Cluster {predicted_cluster}** tem alta taxa de conclusão ({cluster_completion_rate:.1f}%), indicando casos similares tendem a ser resolvidos.")
    elif cluster_completion_rate < 40:
        st.warning(f"🔍 **Cluster {predicted_cluster}** tem baixa taxa de conclusão ({cluster_completion_rate:.1f}%), indicando casos similares tendem a ser arquivados.")
    else:
        st.info(f"🔍 **Cluster {predicted_cluster}** tem taxa equilibrada de conclusão ({cluster_completion_rate:.1f}%).")

# Sidebar para filtros
st.sidebar.header("📊 Filtros e Configurações")

st.sidebar.metric("Total de Ocorrências", len(df_filtered))
st.sidebar.metric("Concluídos", len(df_filtered[df_filtered['status_binario'] == 1]))
st.sidebar.metric("Arquivados", len(df_filtered[df_filtered['status_binario'] == 0]))

# Guia rápido (glossário) para usuários não técnicos
with st.sidebar.expander("Guia rápido (o que é cada coisa?)", expanded=False):
    st.markdown("- **Cluster**: grupo de casos parecidos.\n- **Probabilidade**: quão provável um caso ser concluído.\n- **Acurácia**: o quanto o modelo acerta.\n- **Taxa de conclusão**: % de casos concluídos.")

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
    # Status por tipo de crime (visão geral do conjunto rotulado)
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

# Análise de clusters
st.markdown("")

# =====================
# Relação Não Supervisionado × Supervisionado
# =====================
st.markdown("---")
st.header("🧩 Análise Não Supervisionada (Clusters)")
st.subheader("🔗 Relação entre Clusters (K-Means) e Predições do Modelo")

# Recriar features codificadas para TODO o conjunto filtrado usando os mesmos encoders
categorical_features_all = ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']
feature_columns_all = [f + '_encoded' for f in categorical_features_all] + ['quantidade_vitimas', 'quantidade_suspeitos']

df_encoded_all = df_filtered.copy()
for feature in categorical_features_all:
    df_encoded_all[feature + '_encoded'] = le_dict[feature].transform(df_encoded_all[feature].astype(str))

X_all = df_encoded_all[feature_columns_all]

# Probabilidades preditas para TODO o conjunto (coerente com o modelo escolhido)
if model_choice == "Regressão Logística":
    X_all_scaled = scaler.transform(X_all)
    proba_all = model.predict_proba(X_all_scaled)[:, 1]
else:
    proba_all = model.predict_proba(X_all)[:, 1]

pred_label_all = (proba_all >= 0.5).astype(int)

# Anexar probabilidades e rótulos ao dataframe com clusters
df_rel = df_with_clusters.copy()
df_rel = df_rel.reset_index(drop=True)

# Garantir alinhamento por índice com df_filtered
df_rel['proba_concluido'] = pd.Series(proba_all).values
df_rel['predito_binario'] = pd.Series(pred_label_all).values
df_rel['status_predito'] = np.where(df_rel['predito_binario'] == 1, 'Concluído', 'Arquivado')

st.subheader("Distribuição de Probabilidades por Cluster")
fig_box = px.box(
    df_rel,
    x='cluster', y='proba_concluido', color='cluster',
    labels={'cluster': 'Cluster', 'proba_concluido': 'Probabilidade de Conclusão (Predita)'},
    title='Probabilidades de Conclusão por Cluster (Supervisionado vs Clusters)'
)
fig_box.update_layout(showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# Métricas por cluster: tamanho, taxa real, taxa predita média e acurácia por cluster
st.subheader("Métricas por Cluster")
cluster_metrics = df_rel.groupby('cluster').apply(
    lambda g: pd.Series({
        'Total_Casos': int(len(g)),
        'Taxa_Conclusao_Real': float(g['status_binario'].mean()),
        'Prob_Predita_Media': float(g['proba_concluido'].mean()),
        'Acuracia_Predito': float((g['predito_binario'] == g['status_binario']).mean())
    })
).reset_index()

cluster_metrics['Taxa_Conclusao_Real'] = cluster_metrics['Taxa_Conclusao_Real'].round(3)
cluster_metrics['Prob_Predita_Media'] = cluster_metrics['Prob_Predita_Media'].round(3)
cluster_metrics['Acuracia_Predito'] = cluster_metrics['Acuracia_Predito'].round(3)

st.dataframe(cluster_metrics.sort_values('Acuracia_Predito', ascending=False))

col_a, col_b = st.columns(2)
with col_a:
    fig_acc = px.bar(
        cluster_metrics,
        x='cluster', y='Acuracia_Predito',
        title='Acurácia por Cluster',
        labels={'Acuracia_Predito': 'Acurácia'}
    )
    fig_acc.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig_acc, use_container_width=True)

with col_b:
    fig_cal = px.bar(
        cluster_metrics,
        x='cluster', y='Prob_Predita_Media',
        title='Probabilidade Predita Média por Cluster',
        labels={'Prob_Predita_Media': 'Probabilidade Média'}
    )
    fig_cal.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig_cal, use_container_width=True)

# =====================
# Modo Simplificado: Insights em linguagem natural
# (posicionado aqui para garantir que cluster_metrics já existe)
# =====================
st.header("💡 Insights em linguagem simples")

# 1) Em que grupos os casos tendem a ser concluídos?
top_clusters = cluster_metrics.sort_values('Taxa_Conclusao_Real', ascending=False).head(3)
def to_ratio(p):
    # Converte percentual (0-1) em expressão tipo "7 em cada 10"
    if pd.isna(p):
        return "—"
    denom = 10
    num = int(round(p * denom))
    num = max(0, min(num, denom))
    return f"{num} em cada {denom}"

txt_top = ", ".join([
    f"Cluster {int(r['cluster'])} ({to_ratio(r['Taxa_Conclusao_Real'])} casos concluídos)"
    for _, r in top_clusters.iterrows()
]) if len(top_clusters) else "—"
st.markdown(f"- **Onde mais conclui:** {txt_top}")

# 2) Onde o modelo mais acerta?
top_acc = cluster_metrics.sort_values('Acuracia_Predito', ascending=False).head(3)
txt_acc = ", ".join([
    f"Cluster {int(r['cluster'])} ({to_ratio(r['Acuracia_Predito'])} acertos)"
    for _, r in top_acc.iterrows()
]) if len(top_acc) else "—"
st.markdown(f"- **Onde o modelo mais acerta:** {txt_acc}")

# 3) Como interpretar uma probabilidade?
st.markdown("- **Como ler a probabilidade:** acima de 70% ≈ 7 em 10 chances; abaixo de 30% ≈ 3 em 10; no meio, incerteza.")

# 4) Explicação curta de uso
st.markdown("- **Como usar:** selecione as características do caso e veja a probabilidade e o grupo parecido. Compare com as métricas por cluster acima para entender o contexto.")
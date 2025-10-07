import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, make_scorer, roc_auc_score, f1_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import folium
import warnings
warnings.filterwarnings('ignore')

# Configuração do Plotly para suprimir avisos de depreciação
import plotly.io as pio
pio.templates.default = "plotly"

# Configuração para suprimir avisos específicos do Plotly
import logging
logging.getLogger('plotly').setLevel(logging.CRITICAL)
logging.getLogger('plotly.graph_objects').setLevel(logging.CRITICAL)
logging.getLogger('plotly.express').setLevel(logging.CRITICAL)

# Configuração adicional para suprimir avisos de depreciação
import os
os.environ['PLOTLY_DISABLE_DEPRECATION_WARNINGS'] = '1'

# Suprimir warnings do Python relacionados ao Plotly
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='plotly')
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')
warnings.filterwarnings('ignore', message='.*deprecated.*')
warnings.filterwarnings('ignore', message='.*will be removed.*')
warnings.filterwarnings('ignore', message='.*Use config instead.*')
warnings.filterwarnings('ignore', message='.*Please replace.*')

# Configuração adicional para suprimir avisos do Plotly
import logging
logging.getLogger('plotly').disabled = True
logging.getLogger('plotly.graph_objects').disabled = True
logging.getLogger('plotly.express').disabled = True

# Importar bibliotecas necessárias
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Predição de Status de Crimes",
    page_icon="🔍",
    layout="wide"
)

# Título principal
st.title("🔍 Sistema de Predição de Crimes")
st.markdown("**Sistema Inteligente para Prever se um Crime será Resolvido ou Arquivado**")
st.markdown("*Baseado em características como: tipo de crime, como foi cometido, arma usada e número de pessoas envolvidas*")

# Organização visual
st.markdown("---")

# Seção principal com layout melhorado
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🤖 Como Funciona o Sistema")
    st.markdown("""
    **O sistema usa inteligência artificial para analisar crimes e prever se eles serão resolvidos ou arquivados.**
    """)


# Cards informativos
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h4 style="margin-top: 0; color: #1f77b4;">🔍 O que o sistema analisa</h4>
    <ul style="margin-bottom: 0;">
    <li>Tipo de crime cometido</li>
    <li>Como o crime foi executado</li>
    <li>Arma utilizada</li>
    <li>Quantidade de vítimas</li>
    <li>Quantidade de suspeitos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #f0f8e8; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h4 style="margin-top: 0; color: #2e7d32;">📊 O que o sistema faz</h4>
    <ul style="margin-bottom: 0;">
    <li>Agrupa crimes similares</li>
    <li>Identifica padrões de resolução</li>
    <li>Detecta casos atípicos</li>
    <li>Calcula probabilidades</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Carregar dados
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('dataset_ocorrencias_delegacia_5.csv')
    
    # Converter data_ocorrencia para datetime
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
    
    return df

# Carregar dataset padrão
df = load_data()
st.sidebar.info("📁 Usando dataset padrão")

# Preparar dados para o modelo (REGRESSÃO LOGÍSTICA)
@st.cache_data(show_spinner=False)
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
df_filtered = df[~df['status_investigacao'].str.contains('Em Investiga', na=False)].copy()
df_filtered['status_binario'] = (df_filtered['status_investigacao'].str.contains('Conclu', na=False)).astype(int)

# Mostrar informações sobre o dataset
st.sidebar.info(f"📊 Dataset: {len(df)} registros")
st.sidebar.info(f"📋 Colunas: {len(df.columns)}")


# Preparar dados
X, y, le_dict, feature_columns = prepare_data(df_filtered)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar modelo de clustering
@st.cache_resource(show_spinner=False)
def create_clustering_model(df):
    """Cria modelo de clustering com as mesmas features da regressão"""
    categorical_features = ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']
    numerical_features = ['quantidade_vitimas', 'quantidade_suspeitos']
    
    df_cluster = df.copy()
    le_cluster = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        df_cluster[feature + '_encoded'] = le.fit_transform(df_cluster[feature].astype(str))
        le_cluster[feature] = le
    
    cluster_columns = [f + '_encoded' for f in categorical_features] + numerical_features
    X_cluster = df_cluster[cluster_columns]
    
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    
    df_cluster['cluster'] = clusters
    
    return df_cluster, kmeans, scaler_cluster, le_cluster, cluster_columns

# Função para detecção de anomalias
@st.cache_resource(show_spinner=False)
def detect_anomalies(df):
    """Detecta anomalias usando Isolation Forest e LOF"""
    categorical_features = ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada']
    numerical_features = ['quantidade_vitimas', 'quantidade_suspeitos']
    
    df_anomaly = df.copy()
    le_anomaly = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        df_anomaly[feature + '_encoded'] = le.fit_transform(df_anomaly[feature].astype(str))
        le_anomaly[feature] = le
    
    anomaly_columns = [f + '_encoded' for f in categorical_features] + numerical_features
    X_anomaly = df_anomaly[anomaly_columns]
    
    scaler_anomaly = StandardScaler()
    X_anomaly_scaled = scaler_anomaly.fit_transform(X_anomaly)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_anomalies = iso_forest.fit_predict(X_anomaly_scaled)
    
    # Ajustar n_neighbors baseado no tamanho dos dados
    n_neighbors = min(20, max(5, len(X_anomaly_scaled) // 10))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1, novelty=True)
    lof.fit(X_anomaly_scaled)
    lof_anomalies = lof.predict(X_anomaly_scaled)
    
    df_anomaly['iso_anomaly'] = iso_anomalies
    df_anomaly['lof_anomaly'] = lof_anomalies
    df_anomaly['is_anomaly'] = ((iso_anomalies == -1) | (lof_anomalies == -1)).astype(int)
    
    return df_anomaly, iso_forest, lof, scaler_anomaly, le_anomaly, anomaly_columns

# Criar modelo de clustering
df_with_clusters, kmeans_model, scaler_cluster, le_cluster, cluster_columns = create_clustering_model(df_filtered)

# Detectar anomalias
df_with_anomalies, iso_model, lof_model, scaler_anomaly, le_anomaly, anomaly_columns = detect_anomalies(df_filtered)

# Treinar modelo
st.header("🎯 Faça sua Predição")

st.markdown("""
**Selecione as características do crime abaixo e o sistema irá:**
- Calcular a probabilidade de ser resolvido ou arquivado
- Mostrar em qual grupo de crimes similares ele se encaixa
- Indicar se é um caso atípico que merece atenção especial
""")

model_choice = st.selectbox("Escolha o modelo:", ["Random Forest"], disabled=True)
st.info("💡 **Random Forest** é um algoritmo de inteligência artificial que combina múltiplas 'árvores de decisão' para fazer predições mais precisas e confiáveis.")

# Configurações do sistema
st.subheader("⚙️ Configurações do Sistema")

# Explicação simples sobre tunagem
st.markdown("""
**O que é ajuste automático?**
- O sistema pode testar diferentes configurações para encontrar a melhor precisão
- Isso pode melhorar a qualidade das predições, mas demora mais tempo
- Você pode escolher se quer usar ou não
""")

# Opção simples de ativar/desativar
tuning_enabled = st.radio(
    "Escolha uma opção:",
    ["🚀 Usar configurações rápidas (recomendado)", "🔍 Ajustar automaticamente para melhor precisão"],
    help="A primeira opção é mais rápida, a segunda pode ser mais precisa"
)

# Se escolher ajuste automático, mostrar opções simples
if "Ajustar automaticamente" in tuning_enabled:
    st.markdown("**Configurações de ajuste:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Opção simples de velocidade vs precisão
        speed_choice = st.selectbox(
            "Velocidade do ajuste:",
            ["Rápido (5 tentativas)", "Médio (15 tentativas)", "Lento (30 tentativas)"],
            help="Mais tentativas = melhor precisão, mas demora mais"
        )
        
        # Mapear para número de iterações
        if "Rápido" in speed_choice:
            n_iter = 5
        elif "Médio" in speed_choice:
            n_iter = 15
        else:
            n_iter = 30
    
    with col2:
        # Critério de qualidade simplificado
        quality_choice = st.selectbox(
            "Critério de qualidade:",
            ["Precisão geral", "Balanceamento de classes"],
            help="Precisão geral: foca na acurácia total. Balanceamento: trata classes desiguais melhor"
        )
        
        # Mapear para scoring
        if "Precisão geral" in quality_choice:
            scoring = 'f1'
        else:
            scoring = 'roc_auc'
    
    # Configurações fixas para simplificar
    search_type = "Busca Aleatória"
    cv_folds = 5
    
    st.info(f"🔧 O sistema vai testar {n_iter} configurações diferentes para encontrar a melhor precisão.")
else:
    # Configurações padrão quando não usar tunagem
    search_type = "Busca Exaustiva"
    scoring = 'roc_auc'
    cv_folds = 5
    n_iter = None

# Preparar objetos de tunagem
best_params = None
best_cv_score = None

# Função de treinamento com cache inteligente
@st.cache_resource(show_spinner=False)
def train_model_with_cache(X_train, y_train, tuning_enabled, search_type, scoring, cv_folds, n_iter):
    """Treina modelo com cache baseado nas configurações de tunagem"""
    
    if "Ajustar automaticamente" in tuning_enabled:
        # Espaço de busca otimizado para Random Forest
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced']
        }
        
        if search_type == "Busca Aleatória":
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid_rf,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                random_state=42,
                n_jobs=-1
            )
        else:
            search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid_rf,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
        
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, search.best_score_
    else:
        # Usar configurações otimizadas padrão
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            max_features='sqrt',
            min_samples_split=10,
            min_samples_leaf=1,
            bootstrap=False,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model, None, None

# Treinar modelo
with st.spinner("🔍 Treinando modelo..."):
    model, best_params, best_cv_score = train_model_with_cache(
        X_train, y_train, tuning_enabled, search_type, scoring, cv_folds, n_iter
    )

# Predições
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Interface de predição
st.header("📝 Preencha as Informações do Crime")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Características do Crime:")
    
    # Inputs para predição (apenas features relevantes)
    tipo_crime = st.selectbox("Tipo de Crime", df['tipo_crime'].unique())
    modus_operandi = st.selectbox("Como foi cometido", df['descricao_modus_operandi'].unique())
    arma = st.selectbox("Arma Utilizada", df['arma_utilizada'].unique())

with col2:
    st.subheader("Pessoas Envolvidas:")
    
    qtd_vitimas = st.slider("Quantidade de Vítimas", 0, 4, 1)
    qtd_suspeitos = st.slider("Quantidade de Suspeitos", 0, 4, 1)

# Botão de predição
if st.button("🔮 Analisar Crime", type="primary"):
    # Preparar dados de entrada
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
    
    # Verificar se é anomalia
    X_input_anomaly = input_df_cluster[anomaly_columns]
    X_input_anomaly_scaled = scaler_anomaly.transform(X_input_anomaly)

    iso_pred = iso_model.predict(X_input_anomaly_scaled)[0]
    
    # Usar o modelo LOF já treinado para predição
    lof_pred = lof_model.predict(X_input_anomaly_scaled)[0]
    is_anomaly = (iso_pred == -1) or (lof_pred == -1)
    
    # Exibir resultados
    st.subheader("🎯 O Que o Sistema Descobriu")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probabilidade de Arquivamento", f"{proba[0]:.1%}")
        st.metric("Probabilidade de Conclusão", f"{proba[1]:.1%}")
    
    with col2:
        st.metric("Grupo de Crimes Similares", f"Grupo {predicted_cluster}")
        
        # Análise do cluster predito
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == predicted_cluster]
        cluster_completion_rate = cluster_data['status_binario'].mean() * 100
        st.metric("Taxa de Conclusão do Grupo", f"{cluster_completion_rate:.1f}%")
    
    with col3:
        # Status da anomalia
        if is_anomaly:
            st.error("🚨 **CASO ATÍPICO**")
            st.markdown("Este crime tem características muito diferentes dos casos normais e merece atenção especial.")
        else:
            st.success("✅ **CASO PADRÃO**")
            st.markdown("Este crime segue padrões similares aos casos já conhecidos.")

        # Exibir probabilidades em formato de texto
        st.info(f"📊 **Probabilidade de Arquivamento:** {proba[0]:.1%} | **Probabilidade de Conclusão:** {proba[1]:.1%}")
    
    # Análise do cluster predito
    st.subheader(f"📊 O Que Sabemos Sobre o Grupo {predicted_cluster}")
    
    cluster_analysis = cluster_data.groupby('status_investigacao').size()
    cluster_analysis_pct = cluster_data['status_investigacao'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Como os crimes deste grupo costumam terminar:**")
        st.dataframe(cluster_analysis_pct.round(1))
    
    with col2:
        # Características dominantes do cluster
        st.write("**🔍 O que é comum neste grupo de crimes:**")
        tipo_dominante = cluster_data['tipo_crime'].mode()[0]
        modus_dominante = cluster_data['descricao_modus_operandi'].mode()[0]
        arma_dominante = cluster_data['arma_utilizada'].mode()[0]
        
        st.write(f"• **Tipo de Crime mais comum:** {tipo_dominante}")
        st.write(f"• **Como costuma ser cometido:** {modus_dominante}")
        st.write(f"• **Arma mais usada:** {arma_dominante}")
        st.write(f"• **Número médio de vítimas:** {cluster_data['quantidade_vitimas'].mean():.1f}")
        st.write(f"• **Suspeitos médios:** {cluster_data['quantidade_suspeitos'].mean():.1f}")
    
    # Interpretação
    if proba[1] > 0.6:
        st.success("✅ **Alta probabilidade de CONCLUSÃO** - O caso tem características que favorecem a resolução da investigação.")
    elif proba[0] > 0.6:
        st.warning("⚠️ **Alta probabilidade de ARQUIVAMENTO** - O caso tem características que podem levar ao arquivamento.")
    else:
        st.info("🤔 **Probabilidades equilibradas** - O caso pode ter qualquer um dos dois desfechos.")
    
    # Interpretação do cluster
    if cluster_completion_rate > 60:
        st.info(f"🔍 **Grupo {predicted_cluster}** tem alta taxa de conclusão ({cluster_completion_rate:.1f}%), indicando que crimes similares tendem a ser resolvidos.")
    elif cluster_completion_rate < 40:
        st.warning(f"🔍 **Grupo {predicted_cluster}** tem baixa taxa de conclusão ({cluster_completion_rate:.1f}%), indicando que crimes similares tendem a ser arquivados.")
    else:
        st.info(f"🔍 **Grupo {predicted_cluster}** tem taxa equilibrada de conclusão ({cluster_completion_rate:.1f}%).")

# Tela de Anomalias
st.markdown("---")
st.header("🚨 Detecção de Casos Atípicos")

st.markdown("""
**O sistema identifica crimes que são muito diferentes dos casos normais.**

**🔍 Por que isso é importante?**
- **Casos atípicos** podem indicar novos tipos de crime
- **Merecem atenção especial** da polícia
- **Podem revelar padrões** que não foram identificados antes
- **Ajudam a melhorar** as estratégias de investigação

**📊 Como funciona:**
O sistema compara cada crime com todos os outros e identifica aqueles que têm características muito diferentes do padrão normal.
""")

# Estatísticas de anomalias
anomaly_stats = df_with_anomalies['is_anomaly'].value_counts()
total_anomalies = anomaly_stats.get(1, 0)
total_normal = anomaly_stats.get(0, 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Casos Atípicos Encontrados", total_anomalies)
with col2:
    st.metric("Casos Normais", total_normal)
with col3:
    anomaly_rate = (total_anomalies / len(df_with_anomalies)) * 100
    st.metric("Taxa de Casos Atípicos", f"{anomaly_rate:.1f}%")

# Filtros para anomalias
st.subheader("🔍 Como Quer Ver os Casos Atípicos?")

st.markdown("""
**O que são esses filtros?**
- O sistema usa duas formas diferentes de encontrar casos atípicos
- Você pode escolher ver casos encontrados por cada método ou por ambos
- Isso ajuda a entender melhor quais casos são realmente diferentes
""")

col1, col2 = st.columns(2)
with col1:
    show_iso_anomalies = st.checkbox("📊 Mostrar casos encontrados pelo método principal", value=True, help="Método que identifica casos muito diferentes do padrão normal")
with col2:
    show_lof_anomalies = st.checkbox("🔍 Mostrar casos encontrados pelo método de comparação", value=True, help="Método que compara com casos similares para encontrar diferenças")

# Filtrar anomalias
anomaly_filter = df_with_anomalies['is_anomaly'] == 1
if show_iso_anomalies and not show_lof_anomalies:
    anomaly_filter = df_with_anomalies['iso_anomaly'] == -1
elif show_lof_anomalies and not show_iso_anomalies:
    anomaly_filter = df_with_anomalies['lof_anomaly'] == -1

anomalies_df = df_with_anomalies[anomaly_filter]

if len(anomalies_df) > 0:
    st.subheader(f"📋 Casos Que Precisam de Atenção Especial ({len(anomalies_df)} casos encontrados)")
    
    st.info("""
    💡 **Por que estes casos são especiais?**
    - São muito diferentes dos crimes normais que vemos
    - Podem ter características únicas que merecem investigação especial
    - Podem indicar novos tipos de crimes ou padrões criminais
    """)
    
    # Selecionar colunas para exibir
    display_columns = ['tipo_crime', 'descricao_modus_operandi', 'arma_utilizada', 
                      'quantidade_vitimas', 'quantidade_suspeitos', 'status_investigacao']
    
    # Filtrar apenas colunas que existem
    available_columns = [col for col in display_columns if col in anomalies_df.columns]
    
    if available_columns:
        # Renomear colunas para melhor compreensão
        display_df = anomalies_df[available_columns].copy()
        display_df.columns = ['Tipo de Crime', 'Como foi Cometido', 'Arma Utilizada', 
                             'Quantidade de Vítimas', 'Quantidade de Suspeitos', 'Desfecho']
        
        st.dataframe(display_df, width='stretch')
        
        # Análise das anomalias
        st.subheader("📊 O Que Podemos Aprender Destes Casos?")
        
        # Cards de estatísticas gerais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔍 Total de Casos Atípicos", len(anomalies_df))
        
        with col2:
            arquivados = len(anomalies_df[anomalies_df['status_investigacao'].str.contains('Arquivado', na=False)])
            taxa_arquivamento = (arquivados / len(anomalies_df)) * 100
            st.metric("📋 Taxa de Arquivamento", f"{taxa_arquivamento:.1f}%")
        
        with col3:
            concluidos = len(anomalies_df[anomalies_df['status_investigacao'].str.contains('Concluído', na=False)])
            taxa_conclusao = (concluidos / len(anomalies_df)) * 100
            st.metric("✅ Taxa de Conclusão", f"{taxa_conclusao:.1f}%")
        
        st.markdown("---")
        
        # Top crimes atípicos
        st.subheader("📋 Top 10 Crimes Mais Atípicos")
        crime_dist = anomalies_df['tipo_crime'].value_counts()
        top_crimes = crime_dist.head(10)
        
        # Criar cards simples para cada crime
        cols = st.columns(2)
        for i, (crime, count) in enumerate(top_crimes.items()):
            percentage = (count / len(anomalies_df)) * 100
            col_idx = i % 2
            
            with cols[col_idx]:
                # Emoji baseado na posição
                if i == 0:
                    emoji = "🥇"
                elif i == 1:
                    emoji = "🥈"
                elif i == 2:
                    emoji = "🥉"
                else:
                    emoji = "🔸"
                
                st.info(f"{emoji} **{crime}**: {count} casos ({percentage:.1f}%)")
        
        st.markdown("---")
        
        # Distribuição por status
        st.subheader("📈 Como Terminaram Estes Casos Especiais?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            arquivados = len(anomalies_df[anomalies_df['status_investigacao'].str.contains('Arquivado', na=False)])
            perc_arquivados = (arquivados / len(anomalies_df)) * 100
            st.metric("📋 Casos Arquivados", arquivados, f"{perc_arquivados:.1f}% dos atípicos")
        
        with col2:
            concluidos = len(anomalies_df[anomalies_df['status_investigacao'].str.contains('Concluído', na=False)])
            perc_concluidos = (concluidos / len(anomalies_df)) * 100
            st.metric("✅ Casos Concluídos", concluidos, f"{perc_concluidos:.1f}% dos atípicos")
    else:
        st.warning("⚠️ Não foi possível mostrar os detalhes dos casos atípicos.")
else:
    st.info("""
    ℹ️ **Nenhum caso atípico encontrado com os filtros selecionados.**
    
    Isso pode significar que:
    - Todos os casos seguem padrões normais
    - Os filtros estão muito restritivos
    - Tente ajustar os filtros acima para ver mais casos
    """)

# Mapa de Hotspots (por último)
st.markdown("---")
st.header("🗺️ Mapa de Hotspots")

st.markdown("""
**O mapa mostra onde os crimes acontecem com mais frequência na cidade.**

**🔍 O que é um hotspot?**
- **Hotspot** = área com alta concentração de crimes
- **Círculos vermelhos** = áreas com muitos crimes
- **Círculos menores** = áreas com poucos crimes
- **Cores mais escuras** = maior concentração de crimes

**📊 Como usar o mapa:**
- **Clique nos círculos** para ver detalhes de cada área
- **Use o zoom** para explorar bairros específicos
- **Compare as áreas** para identificar padrões geográficos
- **Identifique zonas críticas** que precisam de mais atenção policial

**🎯 Por que isso é importante?**
- Ajuda a **planejar patrulhamento** policial
- Identifica **áreas de risco** para a população
- Permite **alocação de recursos** de forma mais eficiente
- Facilita a **análise de padrões** geográficos dos crimes
""")

# Função para criar mapa de hotspots
def create_hotspot_map(df):
    """Cria mapa de hotspots usando Folium"""
    # Coordenadas reais de Recife
    center_lat, center_lon = -8.0476, -34.8770  # Recife como referência
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Usar coordenadas reais dos bairros de Recife
    bairro_coords = {}
    
    # Coordenadas reais dos bairros de Recife (baseado nos bairros do dataset)
    recife_bairros = {
        'Imbiribeira': [-8.1114, -34.9435],  # Zona Sul
        'Boa Viagem': [-8.1000, -34.8833],   # Zona Sul, próximo à praia
        'Santo Amaro': [-8.0969, -34.8984], # Zona Sul
        'Afogados': [-8.1351, -34.9135],    # Zona Sul
        'Tamarineira': [-8.0333, -34.8833], # Zona Norte
        'Torre': [-8.0500, -34.9000],       # Centro
        'Casa Forte': [-8.0331, -34.9181],  # Zona Norte
        'Graças': [-8.0500, -34.9000],      # Centro
        'Espinheiro': [-8.0500, -34.8833],  # Centro
        'Pina': [-8.1000, -34.8833]        # Zona Sul
    }
    
    # Mapear cada bairro do dataset para suas coordenadas reais
    for bairro in df['bairro'].unique():
        if bairro in recife_bairros:
            bairro_coords[bairro] = recife_bairros[bairro]
        else:
            # Para bairros não mapeados, usar coordenadas próximas ao centro de Recife
            lat_offset = np.random.uniform(-0.03, 0.03)
            lon_offset = np.random.uniform(-0.03, 0.03)
            bairro_coords[bairro] = [center_lat + lat_offset, center_lon + lon_offset]
    
    # Contar crimes por bairro
    crime_counts = df['bairro'].value_counts()
    
    # Normalizar contagens para tamanho dos círculos (0.1 a 0.8)
    max_count = crime_counts.max()
    min_count = crime_counts.min()
    
    if max_count > min_count:
        normalized_sizes = 0.1 + 0.7 * (crime_counts - min_count) / (max_count - min_count)
    else:
        normalized_sizes = [0.4] * len(crime_counts)
    
    # Adicionar círculos para cada bairro
    for bairro, count in crime_counts.items():
        if bairro in bairro_coords:
            lat, lon = bairro_coords[bairro]
            size = normalized_sizes[bairro]
            
            # Cor baseada na quantidade de crimes
            if count >= max_count * 0.8:
                color = 'red'
            elif count >= max_count * 0.6:
                color = 'orange'
            elif count >= max_count * 0.4:
                color = 'yellow'
            else:
                color = 'green'
            
            # Adicionar círculo
            folium.CircleMarker(
                location=[lat, lon],
                radius=size * 50,  # Escalar para tamanho visível
                popup=f"<b>{bairro}</b><br>Crimes: {count}",
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
    
    return m

# Criar e exibir mapa
if 'bairro' in df_filtered.columns:
    with st.spinner("🗺️ Criando mapa de hotspots..."):
        hotspot_map = create_hotspot_map(df_filtered)
        st.components.v1.html(hotspot_map._repr_html_(), height=600)
        
        # Estatísticas do mapa
        col1, col2, col3 = st.columns(3)
        with col1:
            total_bairros = df_filtered['bairro'].nunique()
            st.metric("Bairros no Mapa", total_bairros)
        with col2:
            bairro_mais_crimes = df_filtered['bairro'].value_counts().index[0]
            crimes_mais_bairro = df_filtered['bairro'].value_counts().iloc[0]
            st.metric("Bairro com Mais Crimes", f"{bairro_mais_crimes} ({crimes_mais_bairro} crimes)")
        with col3:
            total_crimes = len(df_filtered)
            st.metric("Total de Crimes no Mapa", total_crimes)
else:
    st.warning("⚠️ Coluna 'bairro' não encontrada no dataset. Mapa de hotspots não pode ser gerado.")

# Sidebar para informações
st.sidebar.header("📊 Informações do Sistema")

st.sidebar.metric("Total de Crimes Analisados", len(df_filtered))
st.sidebar.metric("Crimes Resolvidos", len(df_filtered[df_filtered['status_binario'] == 1]))
st.sidebar.metric("Crimes Arquivados", len(df_filtered[df_filtered['status_binario'] == 0]))

# Guia explicativo
with st.sidebar.expander("📚 Como Entender os Resultados", expanded=True):
    st.markdown("""
    **🎯 Probabilidades:**
    - **Alta (>70%)**: Muito provável que aconteça
    - **Média (30-70%)**: Incerto, pode acontecer ou não
    - **Baixa (<30%)**: Pouco provável que aconteça
    
    **👥 Grupos de Crimes:**
    - Crimes similares são agrupados juntos
    - Cada grupo tem características parecidas
    - Ajuda a entender padrões
    
    **🚨 Casos Atípicos:**
    - Crimes muito diferentes do normal
    - Merecem atenção especial
    - Podem indicar novos tipos de crime
    
    **🔧 Ajuste Automático:**
    - Melhora a precisão do sistema
    - Testa diferentes configurações
    - Pode demorar alguns minutos
    """)

# Análise exploratória
st.header("📈 Visão Geral dos Dados")

st.markdown("**Aqui você pode ver como os crimes estão distribuídos no sistema:**")

col1, col2 = st.columns(2)

with col1:
    # Distribuição do status
    status_counts = df_filtered['status_investigacao'].value_counts()
    st.write("**📊 Como terminam os crimes no sistema:**")
    for status, count in status_counts.items():
        percentage = (count / len(df_filtered)) * 100
        st.write(f"• {status}: {count} casos ({percentage:.1f}% do total)")

with col2:
    # Status por tipo de crime
    status_crime = pd.crosstab(df_filtered['tipo_crime'], df_filtered['status_investigacao'])
    st.write("**📈 Como cada tipo de crime costuma terminar:**")
    st.dataframe(status_crime, width='stretch')

# Métricas do modelo
st.header("📊 Qualidade do Sistema")

accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Precisão do Sistema", f"{accuracy:.1%}")
    st.caption("Quantos casos o sistema acerta")
with col2:
    # Calcular precisão média
    precision = precision_score(y_test, y_pred, average='weighted')
    st.metric("Confiabilidade", f"{precision:.1%}")
    st.caption("Quão confiável é o sistema")
with col3:
    st.metric("Casos Testados", len(y_test))
    st.caption("Quantos casos foram usados para testar")

# Configurações do sistema (simplificado)
st.subheader("⚙️ Configurações do Sistema")
st.info("O sistema usa Random Forest com configurações otimizadas para análise de crimes.")





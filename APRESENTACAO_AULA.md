# 🎓 Apresentação para Aula - Sistema de Predição de Crimes

---

## 📋 **Slide 1: Título e Objetivo**

# 🔍 Sistema Integrado de Predição de Status de Crimes
## Regressão Logística + Clustering K-Means

**Objetivo:** Desenvolver um sistema de machine learning para prever a probabilidade de conclusão ou arquivamento de investigações criminais

**Integração:** Análise supervisionada (regressão) + não supervisionada (clustering)

---

## 📊 **Slide 2: Dataset e Características**

### **Dados Analisados:**
- **5.000 ocorrências** policiais
- **Período:** 2022-2025
- **Status:** Concluído (33.1%) vs Arquivado (32.8%) vs Em Investigação (34.1%)

### **Features Selecionadas (5 características):**
1. **Tipo de Crime** (10 tipos)
2. **Modus Operandi** (10 modalidades)
3. **Arma Utilizada** (5 tipos)
4. **Quantidade de Vítimas** (0-4)
5. **Quantidade de Suspeitos** (0-4)

### **Features Excluídas:**
- Bairro (evitar viés geográfico)
- Idade do Suspeito (alinhamento entre modelos)

---

## 🤖 **Slide 3: Arquitetura do Sistema**

### **Modelo 1: Regressão Logística (Supervisionado)**
- **Função:** Prediz probabilidade de conclusão/arquivamento
- **Features:** 5 características alinhadas
- **Target:** Status binário (1=Concluído, 0=Arquivado)
- **Performance:** Acurácia ~51%

### **Modelo 2: K-Means Clustering (Não Supervisionado)**
- **Função:** Agrupa crimes similares em clusters
- **Clusters:** 6 grupos identificados
- **Features:** Mesmas 5 características da regressão
- **Objetivo:** Fornecer contexto adicional

---

## 📈 **Slide 4: Resultados do Clustering**

### **6 Clusters Identificados:**

| Cluster | Casos | Tipo Dominante | Taxa Conclusão |
|---------|-------|----------------|----------------|
| 0 | 548 | Ameaça | 53.8% |
| 1 | 553 | Tráfico de Drogas | 50.5% |
| 2 | 598 | Homicídio | 52.2% |
| 3 | 535 | Tráfico de Drogas | 46.5% |
| 4 | 533 | Ameaça | 52.2% |
| 5 | 529 | Violência Doméstica | 45.7% |

### **Distribuição Balanceada:**
- Cada cluster representa ~16-18% dos casos
- Características distintas por cluster
- Taxas de conclusão variadas (45.7% - 53.8%)

---

## 🎯 **Slide 5: Análise por Tipo de Crime**

### **Crimes com Maior Taxa de Conclusão:**
1. **Ameaça:** 52.5% (178/339)
2. **Furto:** 52.0% (169/325)
3. **Homicídio:** 51.5% (168/326)

### **Crimes com Menor Taxa de Conclusão:**
1. **Estupro:** 48.0% (157/327)
2. **Estelionato:** 48.5% (159/328)
3. **Tráfico de Drogas:** 49.0% (169/345)

### **Insights:**
- Crimes mais graves (Homicídio) têm boa taxa de conclusão
- Crimes complexos (Estupro, Estelionato) são mais difíceis de resolver
- Ameaças são resolvidas com maior frequência

---

## 🔄 **Slide 6: Integração dos Modelos**

### **Fluxo de Predição:**

1. **Entrada:** Usuário seleciona características do crime
2. **Processamento Paralelo:**
   - Regressão → Probabilidade de status
   - Clustering → Classificação em cluster
3. **Análise Integrada:**
   - Probabilidades de conclusão/arquivamento
   - Cluster predito com características dominantes
   - Taxa de conclusão histórica do cluster
   - Interpretação contextualizada

### **Vantagens:**
- **Complementaridade:** Probabilidade + Contexto
- **Validação:** Consistência entre modelos
- **Insights:** Padrões de crimes similares

---

## 📊 **Slide 7: Performance do Sistema**

### **Métricas de Regressão Logística:**
- **Acurácia:** 50.8%
- **Precisão (Arquivado):** 50.6%
- **Precisão (Concluído):** 50.9%
- **Recall (Arquivado):** 47.4%
- **Recall (Concluído):** 54.1%

### **Métricas de Clustering:**
- **Inércia:** 9,400.76
- **Iterações:** 48
- **Distribuição:** Balanceada entre clusters

### **Observação:**
- Modelo balanceado (não há viés para uma classe)
- Performance similar para ambas as classes
- Clustering bem distribuído

---

## 🛠️ **Slide 8: Implementação Técnica**

### **Tecnologias:**
- **Frontend:** Streamlit (Interface web)
- **Backend:** Python 3.9+
- **ML:** Scikit-learn
- **Visualização:** Plotly, Matplotlib
- **Deploy:** Streamlit Cloud

### **Funcionalidades:**
- Interface interativa de predição
- Análise exploratória dos dados
- Métricas de performance em tempo real
- Visualizações dinâmicas
- Deploy na nuvem

### **Arquitetura:**
- Código modular e bem documentado
- Configuração para deploy
- Documentação completa

---

## 📈 **Slide 9: Insights e Descobertas**

### **Padrões Identificados:**

1. **Correlação Vítimas vs Suspeitos:** -0.007 (praticamente nula)
2. **Distribuição de Armas:**
   - Explosivos: 21.0%
   - Nenhum: 20.5%
   - Objeto Contundente: 20.1%

3. **Modus Operandi Mais Comum:**
   - Fraude Online: 11.1%
   - Estupro Coletivo: 10.3%
   - Invasão Residencial: 10.3%

### **Fatores Influenciadores:**
- Tipo de crime é o principal determinante
- Modus operandi influencia complexidade
- Quantidade de vítimas/suspeitos tem baixa correlação

---

## 🎯 **Slide 10: Aplicações Práticas**

### **Para Investigadores:**
- **Priorização:** Identificar casos com maior probabilidade de conclusão
- **Alocação de Recursos:** Direcionar esforços para casos promissores
- **Análise de Padrões:** Compreender características de crimes similares

### **Para Gestores:**
- **Planejamento:** Prever carga de trabalho
- **Métricas:** Acompanhar taxa de conclusão por tipo
- **Otimização:** Identificar gargalos na investigação

### **Para Acadêmicos:**
- **Pesquisa:** Padrões de resolução de crimes
- **Políticas:** Efetividade de diferentes abordagens
- **Metodologias:** Integração de ML em investigações

---

## 🔮 **Slide 11: Limitações e Melhorias**

### **Limitações Atuais:**
- Dados baseados em padrões passados
- Apenas 5 características consideradas
- Específico para região dos dados
- Não considera evolução temporal

### **Melhorias Futuras:**
- **Novas Features:** Complexidade, recursos, experiência
- **Modelos Avançados:** Deep Learning, Ensemble Methods
- **Integração:** Dados socioeconômicos, geográficos

### **Próximos Passos:**
- Validação com dados reais
- Implementação em ambiente de produção
- Treinamento de usuários

---

## 📋 **Slide 12: Conclusões**

### **Contribuições:**
1. **Integração Inovadora:** Combina análise supervisionada e não supervisionada
2. **Interface Acessível:** Facilita uso por profissionais não técnicos
3. **Insights Valiosos:** Revela padrões ocultos nos dados
4. **Aplicação Prática:** Diretamente aplicável em investigações reais

### **Impacto Esperado:**
- **Eficiência:** Redução no tempo de investigação
- **Precisão:** Melhor direcionamento de recursos
- **Transparência:** Decisões baseadas em dados
- **Aprendizado:** Compreensão de padrões criminais

### **Valor Acadêmico:**
- Framework replicável
- Inovação em criminologia
- Base para pesquisas futuras

---

## ❓ **Slide 13: Perguntas e Discussão**

### **Tópicos para Discussão:**
1. **Ética:** Uso de IA em investigações criminais
2. **Viés:** Como evitar discriminação algorítmica
3. **Transparência:** Explicabilidade das decisões
4. **Implementação:** Desafios práticos de adoção

### **Perguntas Frequentes:**
- Como garantir a confiabilidade do sistema?
- Quais são os riscos de dependência da tecnologia?
- Como validar resultados com especialistas?
- Qual o papel do investigador humano?

---

## 📚 **Slide 14: Referências e Contatos**

### **Tecnologias Utilizadas:**
- **Streamlit:** Interface web
- **Scikit-learn:** Machine learning
- **Pandas/NumPy:** Processamento de dados
- **Plotly:** Visualizações

### **Métricas de Avaliação:**
- Accuracy, Precision, Recall
- Confusion Matrix
- Silhouette Score

### **Contato:**
- **Repositório:** [GitHub Link]
- **Aplicação:** [Streamlit Cloud Link]
- **Relatório:** RELATORIO_SISTEMA.md

---

## 🎯 **Slide 15: Demonstração Prática**

### **Live Demo:**
1. Acessar aplicação: http://localhost:8501
2. Selecionar características do crime
3. Executar predição
4. Analisar resultados:
   - Probabilidades de status
   - Cluster predito
   - Características dominantes
   - Taxa de conclusão histórica

### **Cenários de Teste:**
- Crime de Ameaça com Arma de Fogo
- Homicídio com Objeto Contundente
- Tráfico de Drogas com Explosivos

---

**Obrigado pela atenção!** 🎓

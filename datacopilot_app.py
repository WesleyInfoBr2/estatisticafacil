import streamlit as st
import pandas as pd
from openai import OpenAI
import io
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv
import numpy as np
import traceback
import logging
import os
import datetime
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings

# Configuração de logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'datacopilot_errors_{datetime.datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Função robusta para extração de palavras em análises textuais
def extrair_palavras_robusta(texto):
    """
    Função robusta para extrair palavras de textos, lidando com diferentes tipos de entrada.
    
    Parâmetros:
    texto (any): Texto a ser processado, pode ser string, número, None, etc.
    
    Retorna:
    list: Lista de palavras extraídas do texto, ou lista vazia se o texto não for processável.
    """
    # Verifica se o texto é uma string e não é NaN, caso contrário retorna lista vazia
    if not isinstance(texto, str) or pd.isna(texto):
        return []
    # Remover caracteres especiais e números, mantendo apenas letras e espaços (incluindo acentuados)
    texto_limpo = re.sub(r'[^a-zA-ZÀ-ú\s]', '', str(texto))
    # Converter para minúsculas e dividir em palavras
    palavras = texto_limpo.lower().split()
    return palavras

# Função para identificar temas/tags em textos
def identificar_temas(texto, num_temas=5):
    """
    Identifica os principais temas/tags em um texto.
    
    Parâmetros:
    texto (any): Texto a ser analisado
    num_temas (int): Número de temas principais a retornar
    
    Retorna:
    list: Lista dos principais temas identificados
    """
    try:
        # Usa a função robusta para extrair palavras
        palavras = extrair_palavras_robusta(texto)
        
        # Remove stopwords em português (palavras comuns sem significado analítico)
        stopwords_pt = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocês', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem'])
        palavras_filtradas = [p for p in palavras if p not in stopwords_pt and len(p) > 2]
        
        # Conta frequência das palavras
        contador = Counter(palavras_filtradas)
        
        # Retorna os temas mais comuns
        temas = [tema for tema, _ in contador.most_common(num_temas)]
        return temas if temas else ["sem_tema_identificado"]
    except Exception as e:
        logging.error(f"Erro ao identificar temas: {str(e)}")
        return ["erro_ao_processar_temas"]

# Função para análise de sentimento
def analisar_sentimento(texto):
    """
    Analisa o sentimento de um texto.
    
    Parâmetros:
    texto (any): Texto a ser analisado
    
    Retorna:
    str: Classificação do sentimento (Muito positivo, Positivo, Neutro, Negativo, Muito negativo)
    """
    try:
        # Verifica se o texto é processável
        if not isinstance(texto, str) or pd.isna(texto) or texto.strip() == "":
            return "Neutro"
        
        # Tenta baixar o VADER se necessário
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nltk.download('vader_lexicon', quiet=True)
        
        # Analisa o sentimento
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(texto)
        compound = score['compound']
        
        # Classifica o sentimento
        if compound >= 0.5:
            return "Muito positivo"
        elif 0.1 <= compound < 0.5:
            return "Positivo"
        elif -0.1 < compound < 0.1:
            return "Neutro"
        elif -0.5 < compound <= -0.1:
            return "Negativo"
        else:
            return "Muito negativo"
    except Exception as e:
        logging.error(f"Erro ao analisar sentimento: {str(e)}")
        return "Erro na análise"

# Função para gerar código Python com tratamento de erros de API
def gerar_codigo_python(df, question, is_text_analysis=False):
    """
    Gera código Python para responder à pergunta do usuário, com tratamento de erros de API.
    
    Parâmetros:
    df: DataFrame pandas
    question: Pergunta do usuário
    is_text_analysis: Se a pergunta é sobre análise de texto
    
    Retorna:
    str: Código Python gerado ou código de fallback em caso de erro
    """
    try:
        # Escolher os exemplos apropriados com base no tipo de pergunta
        quantitative_examples = """
        Pergunta: "Qual a média de vendas?"
        Resposta:
        ```python
        # Verificar se a coluna existe
        if 'Vendas' in df.columns:
            media_vendas = df['Vendas'].mean()
            st.markdown(f"A média de vendas é {media_vendas:.2f}.")
        else:
            st.error("Coluna 'Vendas' não encontrada no DataFrame.")
            colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
            if colunas_numericas:
                st.write(f"Colunas numéricas disponíveis: {', '.join(colunas_numericas)}")
        ```
        
        Pergunta: "Mostre a distribuição de vendas por região"
        Resposta:
        ```python
        # Verificar se as colunas existem
        if 'Região' in df.columns and 'Valor_Venda' in df.columns:
            # Agrupar vendas por região
            vendas_por_regiao = df.groupby('Região')['Valor_Venda'].sum().reset_index()
            
            # Criar gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Região', y='Valor_Venda', data=vendas_por_regiao, ax=ax)
            ax.set_title('Distribuição de Vendas por Região')
            ax.set_ylabel('Valor Total de Vendas')
            st.pyplot(fig)
            plt.close(fig)
        else:
            colunas_faltantes = []
            if 'Região' not in df.columns: colunas_faltantes.append('Região')
            if 'Valor_Venda' not in df.columns: colunas_faltantes.append('Valor_Venda')
            st.error(f"Colunas necessárias não encontradas: {', '.join(colunas_faltantes)}")
            st.write(f"Colunas disponíveis: {', '.join(df.columns.tolist())}")
        ```
        
        Pergunta: "Quais são os 5 produtos mais vendidos?"
        Resposta:
        ```python
        # Verificar se a coluna existe
        if 'Produto' in df.columns:
            # Contar ocorrências de cada produto
            produtos_mais_vendidos = df['Produto'].value_counts().reset_index().head(5)
            produtos_mais_vendidos.columns = ['Produto', 'Quantidade']
            
            # Exibir tabela
            st.write("Os 5 produtos mais vendidos são:")
            st.dataframe(produtos_mais_vendidos)
            
            # Criar gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Quantidade', y='Produto', data=produtos_mais_vendidos, ax=ax)
            ax.set_title('5 Produtos Mais Vendidos')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("Coluna 'Produto' não encontrada no DataFrame.")
            colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if colunas_categoricas:
                st.write(f"Colunas categóricas disponíveis: {', '.join(colunas_categoricas)}")
        ```
        """
        
        text_analysis_examples = """
        Pergunta: "Analise os comentários e identifique os principais temas"
        Resposta:
        ```python
        # Verificar se existe uma coluna de comentários
        colunas_texto = df.select_dtypes(include=['object']).columns
        if len(colunas_texto) == 0:
            st.error("Não foram encontradas colunas de texto no DataFrame.")
        else:
            # Usar a primeira coluna de texto como coluna de comentários
            coluna_comentarios = colunas_texto[0]
            st.write(f"Analisando a coluna: {coluna_comentarios}")
            
            # Criar coluna de temas usando a função identificar_temas
            df_temp = df.copy()
            df_temp['Temas'] = df_temp[coluna_comentarios].apply(lambda x: identificar_temas(x, num_temas=3))
            
            # Contar frequência de cada tema
            todos_temas = [tema for lista_temas in df_temp['Temas'] for tema in lista_temas]
            contador_temas = Counter(todos_temas)
            temas_comuns = contador_temas.most_common(10)
            
            # Criar DataFrame para visualização
            df_temas = pd.DataFrame(temas_comuns, columns=['Tema', 'Frequência'])
            
            # Mostrar resultados
            st.write("### Principais temas identificados nos comentários:")
            st.dataframe(df_temas)
            
            # Criar gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Frequência', y='Tema', data=df_temas, ax=ax)
            ax.set_title('Temas mais frequentes nos comentários')
            st.pyplot(fig)
            plt.close(fig)
        ```
        
        Pergunta: "Qual o sentimento dos comentários?"
        Resposta:
        ```python
        # Verificar se existe uma coluna de comentários
        colunas_texto = df.select_dtypes(include=['object']).columns
        if len(colunas_texto) == 0:
            st.error("Não foram encontradas colunas de texto no DataFrame.")
        else:
            # Usar a primeira coluna de texto como coluna de comentários
            coluna_comentarios = colunas_texto[0]
            st.write(f"Analisando o sentimento da coluna: {coluna_comentarios}")
            
            # Criar coluna de sentimento usando a função analisar_sentimento
            df_temp = df.copy()
            df_temp['Sentimento'] = df_temp[coluna_comentarios].apply(analisar_sentimento)
            
            # Contar frequência de cada categoria de sentimento
            contagem_sentimentos = df_temp['Sentimento'].value_counts().reset_index()
            contagem_sentimentos.columns = ['Sentimento', 'Contagem']
            
            # Mostrar resultados
            st.write("### Análise de sentimento dos comentários:")
            st.dataframe(contagem_sentimentos)
            
            # Criar gráfico de pizza
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(contagem_sentimentos['Contagem'], labels=contagem_sentimentos['Sentimento'], 
                   autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Distribuição de sentimentos nos comentários')
            st.pyplot(fig)
            plt.close(fig)
        ```
        
        Pergunta: "Identifique os temas e sentimentos nos comentários"
        Resposta:
        ```python
        # Verificar se existe uma coluna de comentários
        colunas_texto = df.select_dtypes(include=['object']).columns
        if len(colunas_texto) == 0:
            st.error("Não foram encontradas colunas de texto no DataFrame.")
        else:
            # Usar a primeira coluna de texto como coluna de comentários
            coluna_comentarios = colunas_texto[0]
            st.write(f"Analisando a coluna: {coluna_comentarios}")
            
            # Criar colunas de temas e sentimento
            df_temp = df.copy()
            df_temp['Temas'] = df_temp[coluna_comentarios].apply(lambda x: identificar_temas(x, num_temas=3))
            df_temp['Sentimento'] = df_temp[coluna_comentarios].apply(analisar_sentimento)
            
            # Contar frequência de cada tema
            todos_temas = [tema for lista_temas in df_temp['Temas'] for tema in lista_temas]
            contador_temas = Counter(todos_temas)
            temas_comuns = contador_temas.most_common(10)
            
            # Criar DataFrame para visualização de temas
            df_temas = pd.DataFrame(temas_comuns, columns=['Tema', 'Frequência'])
            
            # Contar frequência de cada categoria de sentimento
            contagem_sentimentos = df_temp['Sentimento'].value_counts().reset_index()
            contagem_sentimentos.columns = ['Sentimento', 'Contagem']
            
            # Mostrar resultados de temas
            st.write("### Principais temas identificados nos comentários:")
            st.dataframe(df_temas)
            
            # Criar gráfico de barras para temas
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Frequência', y='Tema', data=df_temas, ax=ax1)
            ax1.set_title('Temas mais frequentes nos comentários')
            st.pyplot(fig1)
            plt.close(fig1)
            
            # Mostrar resultados de sentimento
            st.write("### Análise de sentimento dos comentários:")
            st.dataframe(contagem_sentimentos)
            
            # Criar gráfico de pizza para sentimentos
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.pie(contagem_sentimentos['Contagem'], labels=contagem_sentimentos['Sentimento'], 
                   autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            ax2.set_title('Distribuição de sentimentos nos comentários')
            st.pyplot(fig2)
            plt.close(fig2)
            
            # Análise cruzada de temas por sentimento
            st.write("### Análise cruzada: Temas mais comuns por sentimento")
            
            # Criar um dicionário para armazenar temas por sentimento
            temas_por_sentimento = {}
            for sentimento in df_temp['Sentimento'].unique():
                # Filtrar comentários por sentimento
                df_sentimento = df_temp[df_temp['Sentimento'] == sentimento]
                # Extrair temas deste sentimento
                temas_sentimento = [tema for lista_temas in df_sentimento['Temas'] for tema in lista_temas]
                # Contar frequência
                contador = Counter(temas_sentimento)
                # Armazenar os 5 mais comuns
                temas_por_sentimento[sentimento] = contador.most_common(5)
            
            # Exibir temas por sentimento
            for sentimento, temas in temas_por_sentimento.items():
                if temas:  # Verificar se há temas para este sentimento
                    st.write(f"**Temas mais comuns em comentários {sentimento}:**")
                    df_temas_sentimento = pd.DataFrame(temas, columns=['Tema', 'Frequência'])
                    st.dataframe(df_temas_sentimento)
        ```
        """
        
        examples_to_use = text_analysis_examples if is_text_analysis else quantitative_examples
        
        prompt = f"""
        Você é um assistente especializado em análise de dados com Python. Gere código Python para responder à pergunta do usuário sobre um DataFrame pandas.
        
        Informações sobre o DataFrame:
        - Nome da variável: df
        - Colunas: {', '.join(df.columns.tolist())}
        - Tipos de dados: {dict(df.dtypes.astype(str))}
        - Primeiras linhas: {df.head(3).to_dict()}
        
        Pergunta do usuário: {question}
        
        Gere apenas o código Python necessário para responder à pergunta. Use pandas, matplotlib, seaborn e numpy conforme necessário.
        Inclua visualizações quando apropriado. Use st.write() ou st.markdown() para exibir resultados textuais e st.pyplot() para gráficos.
        Não inclua explicações, apenas o código Python.
        
        Exemplos de perguntas e respostas esperadas:
        {examples_to_use}
        """
        
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Você é um assistente especializado em análise de dados com Python."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        
        code = response.choices[0].message.content.strip()
        if "```python" in code and "```" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    except Exception as e:
        # Registrar o erro no arquivo de log
        error_details = traceback.format_exc()
        logging.error(f"Erro ao gerar código com a API OpenAI: {str(e)}\nDetalhes:\n{error_details}")
        
        # Gerar código de fallback baseado no tipo de análise
        if is_text_analysis:
            return gerar_codigo_fallback_texto(df)
        else:
            return gerar_codigo_fallback_numerico(df)

# Função para gerar código de fallback para análise de texto
def gerar_codigo_fallback_texto(df):
    """
    Gera código Python de fallback para análise de texto quando a API falha.
    
    Parâmetros:
    df: DataFrame pandas
    
    Retorna:
    str: Código Python de fallback para análise de texto
    """
    return """# Verificar se existe uma coluna de comentários
colunas_texto = df.select_dtypes(include=['object']).columns
if len(colunas_texto) == 0:
    st.error("Não foram encontradas colunas de texto no DataFrame.")
else:
    # Usar a primeira coluna de texto como coluna de comentários
    coluna_comentarios = colunas_texto[0]
    st.write(f"Analisando a coluna: {coluna_comentarios}")
    
    # Criar coluna de temas usando a função identificar_temas
    df_temp = df.copy()
    df_temp['Temas'] = df_temp[coluna_comentarios].apply(lambda x: identificar_temas(x, num_temas=3))
    df_temp['Sentimento'] = df_temp[coluna_comentarios].apply(analisar_sentimento)
    
    # Contar frequência de cada tema
    todos_temas = [tema for lista_temas in df_temp['Temas'] for tema in lista_temas]
    contador_temas = Counter(todos_temas)
    temas_comuns = contador_temas.most_common(10)
    
    # Criar DataFrame para visualização
    df_temas = pd.DataFrame(temas_comuns, columns=['Tema', 'Frequência'])
    
    # Mostrar resultados
    st.write("### Principais temas identificados nos comentários:")
    st.dataframe(df_temas)
    
    # Criar gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Frequência', y='Tema', data=df_temas, ax=ax)
    ax.set_title('Temas mais frequentes nos comentários')
    st.pyplot(fig)
    plt.close(fig)
    
    # Contar frequência de cada categoria de sentimento
    contagem_sentimentos = df_temp['Sentimento'].value_counts().reset_index()
    contagem_sentimentos.columns = ['Sentimento', 'Contagem']
    
    # Mostrar resultados de sentimento
    st.write("### Análise de sentimento dos comentários:")
    st.dataframe(contagem_sentimentos)
    
    # Criar gráfico de pizza para sentimentos
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.pie(contagem_sentimentos['Contagem'], labels=contagem_sentimentos['Sentimento'], 
           autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Distribuição de sentimentos nos comentários')
    st.pyplot(fig2)
    plt.close(fig2)"""

# Função para gerar código de fallback para análise numérica
def gerar_codigo_fallback_numerico(df):
    """
    Gera código Python de fallback para análise numérica quando a API falha.
    
    Parâmetros:
    df: DataFrame pandas
    
    Retorna:
    str: Código Python de fallback para análise numérica
    """
    return """# Análise básica do DataFrame
st.write("### Resumo do DataFrame:")
st.write(f"Número de linhas: {df.shape[0]}")
st.write(f"Número de colunas: {df.shape[1]}")

# Identificar colunas numéricas e categóricas
colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Estatísticas descritivas para colunas numéricas
if colunas_numericas:
    st.write("### Estatísticas descritivas para colunas numéricas:")
    st.dataframe(df[colunas_numericas].describe())
    
    # Criar histogramas para as primeiras 5 colunas numéricas (ou menos)
    for i, col in enumerate(colunas_numericas[:5]):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Distribuição de {col}')
        st.pyplot(fig)
        plt.close(fig)

# Análise de colunas categóricas
if colunas_categoricas:
    st.write("### Análise de colunas categóricas:")
    for i, col in enumerate(colunas_categoricas[:3]):  # Limitar a 3 colunas
        # Contar valores únicos
        valor_counts = df[col].value_counts().reset_index().head(10)
        valor_counts.columns = [col, 'Contagem']
        
        st.write(f"**Top 10 valores em '{col}':**")
        st.dataframe(valor_counts)
        
        # Criar gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Contagem', y=col, data=valor_counts, ax=ax)
        ax.set_title(f'Top 10 valores em {col}')
        st.pyplot(fig)
        plt.close(fig)"""

# Configure sua API Key via secrets.toml ou diretamente aqui
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("EstatísticaFácil - Seu Analista de Dados com IA")
st.markdown("""Faça upload de um arquivo CSV, XLSX ou TXT e pergunte algo em linguagem natural. 
A IA vai gerar o código Python, exibir, e você pode decidir se quer rodar.""" )

# Inicializar st.session_state se não existir
if "df" not in st.session_state: st.session_state.df = None
if "dfs_dict" not in st.session_state: st.session_state.dfs_dict = None
if "selected_sheet_name" not in st.session_state: st.session_state.selected_sheet_name = None
if "show_initial_analysis" not in st.session_state: st.session_state.show_initial_analysis = False
if "last_uploaded_file_id" not in st.session_state: st.session_state.last_uploaded_file_id = None
if "widget_key_prefix" not in st.session_state: st.session_state.widget_key_prefix = "initial_prefix"

def on_file_uploader_change():
    if st.session_state.file_uploader_main is None and st.session_state.last_uploaded_file_id is not None:
        st.session_state.df = None
        st.session_state.dfs_dict = None
        st.session_state.selected_sheet_name = None
        st.session_state.show_initial_analysis = False
        st.session_state.last_uploaded_file_id = None
        st.session_state.widget_key_prefix = f"cleared_{np.random.randint(10000)}"
        keys_to_delete = [k for k in st.session_state.keys() if not k.startswith(st.session_state.widget_key_prefix) and 
                          any(k.startswith(p) for p in ["sheet_selector_", "show_analysis_checkbox_", 
                                                       "col_num_select_", "col_cat_select_", 
                                                       "question_input_", "code_editor_", "run_code_checkbox_"])]
        for key_del in keys_to_delete:
            if key_del in st.session_state: del st.session_state[key_del]

uploaded_file = st.file_uploader("Faça upload do arquivo (CSV, XLSX, TXT)", 
                                 type=["csv", "xlsx", "txt"], 
                                 key="file_uploader_main", 
                                 on_change=on_file_uploader_change)

if uploaded_file is not None:
    process_new_file = False
    if st.session_state.last_uploaded_file_id != uploaded_file.file_id:
        process_new_file = True
        st.session_state.last_uploaded_file_id = uploaded_file.file_id
        st.session_state.widget_key_prefix = f"file_{uploaded_file.file_id}"
        st.session_state.df = None
        st.session_state.dfs_dict = None
        st.session_state.selected_sheet_name = None
        keys_to_delete = []
        for k in st.session_state.keys():
            is_dynamic_widget_key = any(k.startswith(p) for p in ["sheet_selector_", "show_analysis_checkbox_", 
                                                                  "col_num_select_", "col_cat_select_", 
                                                                  "question_input_", "code_editor_", "run_code_checkbox_"])
            if is_dynamic_widget_key and not k.startswith(st.session_state.widget_key_prefix):
                keys_to_delete.append(k)
        for key_del in keys_to_delete:
            if key_del in st.session_state: del st.session_state[key_del]

    if process_new_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_extension == "csv":
                sniffer = csv.Sniffer()
                try:
                    sample_bytes = uploaded_file.read(2048); uploaded_file.seek(0)
                    dialect = sniffer.sniff(sample_bytes.decode("utf-8-sig"))
                    df_temp = pd.read_csv(uploaded_file, sep=dialect.delimiter, encoding="utf-8-sig", on_bad_lines="warn")
                except (csv.Error, UnicodeDecodeError):
                    uploaded_file.seek(0); df_temp = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", on_bad_lines="warn", skipinitialspace=True)
                    if df_temp.shape[1] == 1:
                        uploaded_file.seek(0); df_temp = pd.read_csv(uploaded_file, sep=",", encoding="utf-8-sig", on_bad_lines="warn", skipinitialspace=True)
                st.session_state.dfs_dict = {"CSV_Data": df_temp}
                st.session_state.selected_sheet_name = "CSV_Data"
            elif file_extension == "xlsx":
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                if not sheet_names:
                    st.error("O arquivo XLSX não contém planilhas."); st.stop()
                st.session_state.dfs_dict = {name: excel_file.parse(name) for name in sheet_names}
                st.session_state.selected_sheet_name = sheet_names[0]
            elif file_extension == "txt":
                try:
                    sample_bytes = uploaded_file.read(2048); uploaded_file.seek(0)
                    dialect = csv.Sniffer().sniff(sample_bytes.decode("utf-8-sig"))
                    df_temp = pd.read_csv(uploaded_file, sep=dialect.delimiter, encoding="utf-8-sig", on_bad_lines="warn")
                except (csv.Error, UnicodeDecodeError):
                    uploaded_file.seek(0); df_temp = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8-sig", on_bad_lines="warn")
                st.session_state.dfs_dict = {"TXT_Data": df_temp}
                st.session_state.selected_sheet_name = "TXT_Data"
            else:
                st.error("Formato de arquivo não suportado."); st.stop()
            if st.session_state.selected_sheet_name and st.session_state.dfs_dict:
                st.session_state.df = st.session_state.dfs_dict[st.session_state.selected_sheet_name]
            else:
                 st.error("Não foi possível carregar dados da planilha selecionada."); st.stop()
        except Exception as e:
            st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
            st.session_state.last_uploaded_file_id = None
            st.stop()

    if uploaded_file.name.endswith(".xlsx") and st.session_state.dfs_dict and len(st.session_state.dfs_dict) > 1:
        sheet_names_list = list(st.session_state.dfs_dict.keys())
        selectbox_key_sheet = f"sheet_selector_{st.session_state.widget_key_prefix}"
        current_selected_sheet_from_session = st.session_state.get("selected_sheet_name", sheet_names_list[0])
        if current_selected_sheet_from_session not in sheet_names_list:
            current_selected_sheet_from_session = sheet_names_list[0]
        default_sheet_index = sheet_names_list.index(current_selected_sheet_from_session)
        
        def update_df_on_sheet_selection_change():
            selected_sheet = st.session_state[selectbox_key_sheet]
            st.session_state.selected_sheet_name = selected_sheet
            st.session_state.df = st.session_state.dfs_dict[selected_sheet]
        
        st.selectbox("Selecione a planilha:", sheet_names_list, 
                     key=selectbox_key_sheet, 
                     index=default_sheet_index,
                     on_change=update_df_on_sheet_selection_change)

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Converter colunas numéricas com formatos especiais (moeda, porcentagem, etc.)
        def coluna_e_numerica(coluna):
            """Verifica se uma coluna pode ser convertida para numérica após limpeza."""
            if pd.api.types.is_numeric_dtype(coluna):
                return True
            
            # Se não for numérica, tenta converter uma amostra
            amostra = coluna.dropna().head(100)
            if len(amostra) == 0:
                return False
            
            # Tenta converter strings que podem representar números
            try:
                # Converte strings com formatos especiais
                convertidos = []
                for val in amostra:
                    if not isinstance(val, str):
                        convertidos.append(val)
                        continue
                    
                    # Remove caracteres não numéricos, preservando ponto e vírgula
                    val_limpo = val.replace('R$', '').replace('$', '').replace('%', '')
                    val_limpo = val_limpo.strip()
                    
                    # Substitui vírgula por ponto se houver apenas uma vírgula
                    if val_limpo.count(',') == 1 and val_limpo.count('.') == 0:
                        val_limpo = val_limpo.replace(',', '.')
                    # Se houver vírgula como separador de milhar e ponto como decimal
                    elif val_limpo.count(',') >= 1 and val_limpo.count('.') == 1:
                        val_limpo = val_limpo.replace(',', '')
                    
                    # Tenta converter para float
                    try:
                        convertidos.append(float(val_limpo))
                    except ValueError:
                        return False
                
                # Se todos os valores foram convertidos com sucesso
                return True
            except:
                return False
        
        # Função para converter colunas para o formato numérico correto
        def converter_para_numerico(df):
            df_convertido = df.copy()
            for col in df.columns:
                if coluna_e_numerica(df[col]):
                    # Se já for numérica, mantém como está
                    if pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    
                    # Converte strings com formatos especiais
                    try:
                        # Cria uma série temporária para conversão
                        serie_temp = []
                        for val in df[col]:
                            if pd.isna(val):
                                serie_temp.append(np.nan)
                                continue
                            if not isinstance(val, str):
                                serie_temp.append(val)
                                continue
                            
                            # Remove caracteres não numéricos, preservando ponto e vírgula
                            val_limpo = val.replace('R$', '').replace('$', '').replace('%', '')
                            val_limpo = val_limpo.strip()
                            
                            # Substitui vírgula por ponto se houver apenas uma vírgula
                            if val_limpo.count(',') == 1 and val_limpo.count('.') == 0:
                                val_limpo = val_limpo.replace(',', '.')
                            # Se houver vírgula como separador de milhar e ponto como decimal
                            elif val_limpo.count(',') >= 1 and val_limpo.count('.') == 1:
                                val_limpo = val_limpo.replace(',', '')
                            
                            # Tenta converter para float
                            try:
                                serie_temp.append(float(val_limpo))
                            except ValueError:
                                serie_temp.append(np.nan)
                        
                        # Atualiza a coluna com os valores convertidos
                        df_convertido[col] = serie_temp
                    except:
                        # Se falhar, mantém a coluna original
                        pass
            
            return df_convertido
        
        # Aplicar conversão robusta
        df = converter_para_numerico(df)
        st.session_state.df = df
        
        # Exibir os dados
        st.subheader("Dados Carregados:")
        st.dataframe(df.head(10))
        
        # Opção para mostrar análise inicial
        checkbox_key = f"show_analysis_checkbox_{st.session_state.widget_key_prefix}"
        show_analysis = st.checkbox("Mostrar análise inicial dos dados", key=checkbox_key)
        
        if show_analysis:
            st.subheader("Estatísticas Descritivas Gerais:")
            
            # Preparar estatísticas descritivas
            desc = df.describe(include='all').T
            
            # Adicionar tipo de variável como primeira coluna
            desc.insert(0, 'Tipo da variável', df.dtypes.astype(str))
            
            # Renomear colunas para português
            rename_dict = {
                'count': 'Contagem',
                'unique': 'Contagem únicos',
                'top': 'Primeiro',
                'freq': 'Último',
                'mean': 'Média',
                'std': 'Desvio padrão',
                'min': 'Mínimo',
                '25%': '25%',
                '50%': 'Mediana',
                '75%': '75%',
                'max': 'Máximo'
            }
            desc = desc.rename(columns=rename_dict)
            
            # Adicionar coeficiente de variação para colunas numéricas
            for col in df.select_dtypes(include=np.number).columns:
                if 'Média' in desc.columns and 'Desvio padrão' in desc.columns:
                    if desc.loc[col, 'Média'] != 0:
                        desc.loc[col, 'Coeficiente de variação'] = desc.loc[col, 'Desvio padrão'] / desc.loc[col, 'Média']
                    else:
                        desc.loc[col, 'Coeficiente de variação'] = np.nan
            
            # Exibir estatísticas
            st.dataframe(desc)
            
            # Gráficos para colunas numéricas
            st.subheader("Gráficos para Colunas Numéricas:")
            colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
            
            if colunas_numericas:
                col_num_select_key = f"col_num_select_{st.session_state.widget_key_prefix}"
                col_num = st.selectbox("Selecione uma coluna numérica:", colunas_numericas, key=col_num_select_key)
                
                # Histograma
                st.write(f"Histograma de {col_num}:")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col_num].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                
                # Boxplot com rótulos para outliers
                st.write(f"Boxplot de {col_num}:")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=df[col_num].dropna(), ax=ax)
                
                # Identificar outliers e adicionar rótulos
                box_data = df[col_num].dropna()
                Q1 = box_data.quantile(0.25)
                Q3 = box_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_low = Q1 - 1.5 * IQR
                outlier_high = Q3 + 1.5 * IQR
                
                outliers = box_data[(box_data < outlier_low) | (box_data > outlier_high)]
                
                if not outliers.empty:
                    # Obter posições x dos outliers no boxplot
                    outlier_positions = []
                    for outlier in outliers:
                        outlier_positions.append(outlier)
                    
                    # Adicionar rótulos com posicionamento alternado para evitar sobreposição
                    vertical_offset = 0.02 * (box_data.max() - box_data.min())
                    for i, (pos, val) in enumerate(zip(outlier_positions, outliers)):
                        # Alternar posição vertical para evitar sobreposição
                        offset_multiplier = (i % 3) - 1  # Alterna entre -1, 0, 1
                        
                        # Aumentar o deslocamento se os outliers estiverem muito próximos
                        if i > 0 and abs(outlier_positions[i] - outlier_positions[i-1]) < (box_data.max() - box_data.min()) * 0.05:
                            offset_multiplier *= 2
                        
                        y_pos = val + offset_multiplier * vertical_offset
                        ax.annotate(f'{val:.2f}', xy=(0, val), xytext=(0, y_pos),
                                   ha='center', va='center', fontsize=8,
                                   arrowprops=dict(arrowstyle='-', lw=0.5))
                
                st.pyplot(fig)
                plt.close(fig)
            
            # Gráficos para colunas categóricas
            st.subheader("Gráficos para Colunas Categóricas:")
            colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if colunas_categoricas:
                col_cat_select_key = f"col_cat_select_{st.session_state.widget_key_prefix}"
                col_cat = st.selectbox("Selecione uma coluna categórica:", colunas_categoricas, key=col_cat_select_key)
                
                # Gráfico de barras para as categorias mais frequentes
                value_counts = df[col_cat].value_counts().head(10)
                
                st.write(f"Contagem das 10 categorias mais frequentes em {col_cat}:")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)
        
        # Interface para perguntas em linguagem natural
        st.subheader("Faça uma pergunta sobre os dados:")
        question_key = f"question_input_{st.session_state.widget_key_prefix}"
        question = st.text_input("Exemplo: Qual a média de idade? Mostre a distribuição por gênero.", key=question_key)
        
        current_widget_prefix = st.session_state.widget_key_prefix
        
        if question:
            # Detectar se a pergunta é sobre análise de texto
            is_text_analysis = any(keyword in question.lower() for keyword in 
                                  ['comentário', 'comentarios', 'texto', 'sentimento', 'tema', 'temas', 
                                   'assunto', 'assuntos', 'tópico', 'topicos', 'opinião', 'opinioes'])
            
            try:
                # Gerar código Python para responder à pergunta
                code = gerar_codigo_python(df, question, is_text_analysis)
                
                # Exibir o código gerado
                # st.subheader("Código gerado pela IA:")
                st.expander("Código gerado pela IA:"):
                    st.code(code, language="python")
            
                code_editor_key = f"code_editor_{current_widget_prefix}_{question[:15].replace(' ','_')}"
                code_editado = st.text_area("Edite o código se desejar (AVANÇADO):", value=code, height=300, key=code_editor_key)
                
                st.warning("⚠️ ATENÇÃO: Executar código gerado por IA pode ser arriscado. Revise o código antes de executar. Não execute código que você não entenda ou não confie.")
                
                run_code_key_check = f"run_code_checkbox_{current_widget_prefix}_{question[:15].replace(' ','_')}"
                run_code = st.checkbox("Sim, entendo os riscos e desejo executar o código.", key=run_code_key_check)
                
                if run_code and code_editado.strip():
                    try:
                        # Definir funções úteis para análise de texto
                        from collections import Counter
                        import re
                        import pandas as pd
                        
                        # Verificar se o código menciona colunas que não existem no DataFrame
                        # e adicionar verificações de segurança
                        code_lines = code_editado.split('\n')
                        safe_code_lines = []
                        
                        # Adicionar verificações de segurança no início do código
                        safe_code_lines.append("# Verificações de segurança adicionadas automaticamente")
                        safe_code_lines.append("import traceback")
                        safe_code_lines.append("try:")
                        
                        # Adicionar o código original com indentação
                        for line in code_lines:
                            safe_code_lines.append("    " + line)
                        
                        # Adicionar tratamento de exceções no final
                        safe_code_lines.append("except KeyError as e:")
                        safe_code_lines.append("    coluna = str(e).strip(\"'\")")
                        safe_code_lines.append("    st.error(f\"Erro: Coluna '{coluna}' não encontrada no DataFrame.\")")
                        safe_code_lines.append("    st.write(f\"Colunas disponíveis: {', '.join(df.columns.tolist())}\")")
                        safe_code_lines.append("    logging.error(f\"KeyError: {str(e)} - Coluna não encontrada no DataFrame\")")
                        safe_code_lines.append("except NameError as e:")
                        safe_code_lines.append("    var_name = str(e).split(\"'\")[1] if \"'\" in str(e) else str(e)")
                        safe_code_lines.append("    st.error(f\"Erro: Variável '{var_name}' não definida.\")")
                        safe_code_lines.append("    st.write(\"Verifique se todas as variáveis estão sendo definidas antes de serem usadas.\")")
                        safe_code_lines.append("    logging.error(f\"NameError: {str(e)} - Variável não definida\")")
                        safe_code_lines.append("except Exception as e:")
                        safe_code_lines.append("    st.error(\"Não foi possível processar sua solicitação.\")")
                        safe_code_lines.append("    with st.expander(\"Detalhes técnicos do erro\"):")
                        safe_code_lines.append("        st.code(traceback.format_exc())")
                        safe_code_lines.append("    logging.error(f\"Erro na execução do código: {str(e)}\\n{traceback.format_exc()}\")")
                        
                        # Juntar as linhas em um único código
                        safe_code = "\n".join(safe_code_lines)
                        
                        # Incluir a função extrair_palavras_robusta no ambiente de execução
                        def extrair_palavras(texto):
                            return extrair_palavras_robusta(texto)
                        
                        # Tratar __builtins__ de forma compatível com diferentes ambientes Python
                        # Em alguns ambientes, __builtins__ é um dict, em outros é um módulo com __dict__
                        if isinstance(__builtins__, dict):
                            allowed_builtins = {k: v for k, v in __builtins__.items() if k in ["print", "len", "range", "list", "dict", "str", "int", "float", "bool", "True", "False", "None", "max", "min", "sum", "abs", "round", "all", "any", "isinstance", "getattr", "hasattr", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "sorted", "zip", "map", "filter", "enumerate", "reversed", "set", "tuple"]}
                        else:
                            allowed_builtins = {k: getattr(__builtins__, k) for k in dir(__builtins__) if k in ["print", "len", "range", "list", "dict", "str", "int", "float", "bool", "True", "False", "None", "max", "min", "sum", "abs", "round", "all", "any", "isinstance", "getattr", "hasattr", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "sorted", "zip", "map", "filter", "enumerate", "reversed", "set", "tuple"]}
                        
                        safe_globals = {
                            "__builtins__": allowed_builtins, 
                            "pd": pd, 
                            "plt": plt, 
                            "sns": sns, 
                            "np": np, 
                            "st": st, 
                            "df": df.copy(), 
                            "io": io,
                            "Counter": Counter,
                            "re": re,
                            "extrair_palavras": extrair_palavras,
                            "extrair_palavras_robusta": extrair_palavras_robusta,
                            "identificar_temas": identificar_temas,
                            "analisar_sentimento": analisar_sentimento,
                            "traceback": traceback,
                            "logging": logging
                        }
                        
                        # Compilar o código para verificar erros de sintaxe
                        compile(safe_code, "<string>", "exec")
                        
                        # Executar o código em ambiente seguro
                        exec(safe_code, safe_globals)
                        
                    except Exception as e:
                        # Registrar o erro no arquivo de log
                        error_details = traceback.format_exc()
                        logging.error(f"Erro ao executar código gerado pela IA: {str(e)}\nCódigo:\n{code_editado}\nDetalhes:\n{error_details}")
                        
                        # Mostrar mensagem amigável ao usuário
                        st.error("Não foi possível processar sua solicitação. Nossa equipe técnica foi notificada sobre o problema.")
                        
                        # Mostrar detalhes do erro em um expander (opcional para depuração)
                        with st.expander("Detalhes técnicos do erro (para desenvolvedores)"):
                            st.warning("O erro foi registrado para análise posterior.")
                            st.code(error_details)
                            
                elif run_code and not code_editado.strip():
                     st.error("O campo de código está vazio.")
                     
            except Exception as e:
                # Registrar o erro no arquivo de log
                error_details = traceback.format_exc()
                logging.error(f"Erro ao gerar código com a API OpenAI: {str(e)}\nDetalhes:\n{error_details}")
                
                # Mostrar mensagem amigável ao usuário
                st.error("Não foi possível gerar o código usando a API OpenAI. Estamos usando uma análise alternativa.")
                
                # Mostrar detalhes do erro em um expander
                with st.expander("Detalhes técnicos do erro"):
                    st.warning("O erro foi registrado para análise posterior.")
                    st.code(error_details)
                    
                    # Orientações para resolver problemas de API
                    st.subheader("Possíveis soluções:")
                    st.markdown("""
                    1. **Verifique a chave da API OpenAI**:
                       - Confirme se a chave está configurada corretamente em `st.secrets["OPENAI_API_KEY"]`
                       - Verifique se a chave não expirou ou atingiu limites de uso
                    
                    2. **Problemas de conectividade**:
                       - Verifique se o servidor tem acesso à internet
                       - Confirme se não há bloqueios de firewall para a API OpenAI
                    
                    3. **Limitações da conta**:
                       - Verifique se sua conta OpenAI tem saldo suficiente
                       - Confirme se não há restrições regionais
                    """)
                
                # Gerar código de fallback baseado no tipo de análise
                fallback_code = gerar_codigo_fallback_texto(df) if is_text_analysis else gerar_codigo_fallback_numerico(df)
                
                st.subheader("Código alternativo gerado:")
                code_editor_key = f"code_editor_fallback_{current_widget_prefix}_{question[:15].replace(' ','_')}"
                code_editado = st.text_area("Edite o código se desejar (AVANÇADO):", value=fallback_code, height=300, key=code_editor_key)
                
                st.warning("⚠️ ATENÇÃO: Executar código gerado por IA pode ser arriscado. Revise o código antes de executar. Não execute código que você não entenda ou não confie.")
                
                run_code_key_check = f"run_code_checkbox_fallback_{current_widget_prefix}_{question[:15].replace(' ','_')}"
                run_code = st.checkbox("Sim, entendo os riscos e desejo executar o código.", key=run_code_key_check)
                
                if run_code and code_editado.strip():
                    try:
                        # Definir funções úteis para análise de texto
                        from collections import Counter
                        import re
                        import pandas as pd
                        
                        # Verificar se o código menciona colunas que não existem no DataFrame
                        # e adicionar verificações de segurança
                        code_lines = code_editado.split('\n')
                        safe_code_lines = []
                        
                        # Adicionar verificações de segurança no início do código
                        safe_code_lines.append("# Verificações de segurança adicionadas automaticamente")
                        safe_code_lines.append("import traceback")
                        safe_code_lines.append("try:")
                        
                        # Adicionar o código original com indentação
                        for line in code_lines:
                            safe_code_lines.append("    " + line)
                        
                        # Adicionar tratamento de exceções no final
                        safe_code_lines.append("except KeyError as e:")
                        safe_code_lines.append("    coluna = str(e).strip(\"'\")")
                        safe_code_lines.append("    st.error(f\"Erro: Coluna '{coluna}' não encontrada no DataFrame.\")")
                        safe_code_lines.append("    st.write(f\"Colunas disponíveis: {', '.join(df.columns.tolist())}\")")
                        safe_code_lines.append("    logging.error(f\"KeyError: {str(e)} - Coluna não encontrada no DataFrame\")")
                        safe_code_lines.append("except NameError as e:")
                        safe_code_lines.append("    var_name = str(e).split(\"'\")[1] if \"'\" in str(e) else str(e)")
                        safe_code_lines.append("    st.error(f\"Erro: Variável '{var_name}' não definida.\")")
                        safe_code_lines.append("    st.write(\"Verifique se todas as variáveis estão sendo definidas antes de serem usadas.\")")
                        safe_code_lines.append("    logging.error(f\"NameError: {str(e)} - Variável não definida\")")
                        safe_code_lines.append("except Exception as e:")
                        safe_code_lines.append("    st.error(\"Não foi possível processar sua solicitação.\")")
                        safe_code_lines.append("    with st.expander(\"Detalhes técnicos do erro\"):")
                        safe_code_lines.append("        st.code(traceback.format_exc())")
                        safe_code_lines.append("    logging.error(f\"Erro na execução do código: {str(e)}\\n{traceback.format_exc()}\")")
                        
                        # Juntar as linhas em um único código
                        safe_code = "\n".join(safe_code_lines)
                        
                        # Incluir a função extrair_palavras_robusta no ambiente de execução
                        def extrair_palavras(texto):
                            return extrair_palavras_robusta(texto)
                        
                        # Tratar __builtins__ de forma compatível com diferentes ambientes Python
                        # Em alguns ambientes, __builtins__ é um dict, em outros é um módulo com __dict__
                        if isinstance(__builtins__, dict):
                            allowed_builtins = {k: v for k, v in __builtins__.items() if k in ["print", "len", "range", "list", "dict", "str", "int", "float", "bool", "True", "False", "None", "max", "min", "sum", "abs", "round", "all", "any", "isinstance", "getattr", "hasattr", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "sorted", "zip", "map", "filter", "enumerate", "reversed", "set", "tuple"]}
                        else:
                            allowed_builtins = {k: getattr(__builtins__, k) for k in dir(__builtins__) if k in ["print", "len", "range", "list", "dict", "str", "int", "float", "bool", "True", "False", "None", "max", "min", "sum", "abs", "round", "all", "any", "isinstance", "getattr", "hasattr", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "sorted", "zip", "map", "filter", "enumerate", "reversed", "set", "tuple"]}
                        
                        safe_globals = {
                            "__builtins__": allowed_builtins, 
                            "pd": pd, 
                            "plt": plt, 
                            "sns": sns, 
                            "np": np, 
                            "st": st, 
                            "df": df.copy(), 
                            "io": io,
                            "Counter": Counter,
                            "re": re,
                            "extrair_palavras": extrair_palavras,
                            "extrair_palavras_robusta": extrair_palavras_robusta,
                            "identificar_temas": identificar_temas,
                            "analisar_sentimento": analisar_sentimento,
                            "traceback": traceback,
                            "logging": logging
                        }
                        
                        # Compilar o código para verificar erros de sintaxe
                        compile(safe_code, "<string>", "exec")
                        
                        # Executar o código em ambiente seguro
                        exec(safe_code, safe_globals)
                        
                    except Exception as e:
                        # Registrar o erro no arquivo de log
                        error_details = traceback.format_exc()
                        logging.error(f"Erro ao executar código de fallback: {str(e)}\nCódigo:\n{code_editado}\nDetalhes:\n{error_details}")
                        
                        # Mostrar mensagem amigável ao usuário
                        st.error("Não foi possível processar sua solicitação. Nossa equipe técnica foi notificada sobre o problema.")
                        
                        # Mostrar detalhes do erro em um expander (opcional para depuração)
                        with st.expander("Detalhes técnicos do erro (para desenvolvedores)"):
                            st.warning("O erro foi registrado para análise posterior.")
                            st.code(error_details)

elif uploaded_file is None and st.session_state.last_uploaded_file_id is None:
    st.info("Por favor, faça o upload de um arquivo para começar a análise.")

st.markdown("---")
st.markdown("Desenvolvido como um protótipo. Use com cautela.")

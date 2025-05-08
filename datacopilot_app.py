
import streamlit as st
import pandas as pd
from openai import OpenAI
import io
import matplotlib.pyplot as plt
import re

# Configure sua API Key via secrets.toml ou diretamente aqui
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("EstatísticaFácil - Seu Analista de Dados com IA")

st.markdown("Faça upload de um arquivo CSV e pergunte algo em linguagem natural. A IA vai gerar o código Python, exibir, e você pode decidir se quer rodar.")

uploaded_file = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";", decimal=',', thousands='.', encoding='utf-8-sig')
    st.write("Visualização inicial dos dados:")
    st.dataframe(df.head())

    # Lista de colunas do tipo 'object'
    colunas_obj = df.select_dtypes(include='object').columns
    
    # Função para verificar se uma coluna 'object' é potencialmente numérica
    def coluna_e_numerica(serie):
        amostra = serie.dropna().astype(str).head(10)
        return all(re.match(r"^[\\d\\.,\\s]+$", val) for val in amostra)
    
    # Aplicar a limpeza e conversão apenas em colunas objetivamente numéricas
    for col in colunas_obj:
        if coluna_e_numerica(df[col]):
            # Substituições para casos brasileiros: vírgula decimal e ponto de milhar
            s = df[col].astype(str)
            s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(s, errors='coerce')
    
    st.write("Tipos após tentativa de conversão:")
    st.write(df.dtypes)

    question = st.text_input("O que você quer saber ou fazer com os dados?")
    
    if question:
        prompt = f'''
        Você é um analista de dados em Python. Recebeu o seguinte DataFrame:
        {df.head().to_string()}
        A seguir, o usuário perguntou: "{question}"

        Gere apenas o código Python necessário para responder à pergunta, com comentários claros em cada etapa.
        A aplicação será executada em Streamlit, portanto use st.write(...) para exibir todas as saídas.
        Assuma que o DataFrame principal se chama df.
        Gere apenas o código Python executável, com comentários dentro do código. Não inclua explicações escritas fora do código.
        Gere o código apenas com bibliotecas já importadas.
        Após esse tratamento, aplique os cálculos necessários com base na pergunta do usuário e apresente os resultados com st.dataframe(...), st.write(...) ou visualizações apropriadas.
        '''
        with st.spinner("Gerando código com IA..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            code = response.choices[0].message.content

            raw_code = response.choices[0].message.content
            code = re.sub(r"^```(python)?", "", raw_code.strip(), flags=re.MULTILINE)
            code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE)

        with st.expander("Código gerado pela IA:"):
            st.code(code, language="python")
        with st.expander("Edite o código se desejar:"):
            code_editado = st.text_area("Edite se desejar:", code, height=300)

        run_code = st.checkbox("Executar código?")
        if run_code:
            try:
                # Cria um contexto seguro
                safe_globals = {
                    "__builtins__": __builtins__,
                    "pd": pd,
                    "plt": plt,
                    "st": st,
                    "df": df
                }
                
                # exec(code, safe_globals)
                exec(code_editado, safe_globals)
            except Exception as e:
                st.error(f"Erro ao executar o código: {e}")

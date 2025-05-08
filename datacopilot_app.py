
import streamlit as st
import pandas as pd
from openai import OpenAI
import io
import matplotlib.pyplot as plt
import re

# Configure sua API Key via secrets.toml ou diretamente aqui
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("DataCopilot - Seu Analista de Dados com IA")

st.markdown("Faça upload de um arquivo CSV e pergunte algo em linguagem natural. A IA vai gerar o código Python, exibir, e você pode decidir se quer rodar.")

uploaded_file = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    st.write("Visualização inicial dos dados:")
    st.dataframe(df.head())

    question = st.text_input("O que você quer saber ou fazer com os dados?")
    
    if question:
        prompt = f'''
        Você é um analista de dados em Python. Recebeu o seguinte DataFrame:
        {df.head().to_string()}
        A seguir, o usuário perguntou: "{question}"

        Gere apenas o código Python necessário para responder à pergunta, com comentários claros em cada etapa.
        A aplicação será executada em Streamlit, portanto use st.write(...) para exibir todas as saídas.
        Assuma que o DataFrame principal se chama df.
        Antes de qualquer cálculo ou visualização, identifique automaticamente o tipo de cada variável do DataFrame (numérica, categórica ou textual), e apresente essa lista ao usuário com st.write(...).
        Para variáveis numéricas que ainda estão no formato texto (object), execute uma limpeza usando as seguintes regras:
            Se a variável contém apenas números (sem letras), converta para número.
            Se o valor tem vírgula como separador decimal, troque por ponto.
            Se houver vírgula como separador de milhar (e ponto como decimal), remova a vírgula.
            Use pd.to_numeric(..., errors='coerce') para tratar valores inválidos.
        Não altere colunas categóricas ou com texto descritivo (como nomes, estados, regiões, gênero, escolaridade, faixas etc).
        Gere apenas o código Python executável, com comentários dentro do código. Não inclua explicações escritas fora do código.
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

        st.subheader("Código gerado pela IA:")
        st.code(code, language="python")

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
                
                exec(code, safe_globals)
            except Exception as e:
                st.error(f"Erro ao executar o código: {e}")

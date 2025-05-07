
import streamlit as st
import pandas as pd
import openai
import io
import matplotlib.pyplot as plt

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

        Gere apenas o código Python necessário para responder à pergunta, sem explicações.
        O código deve assumir que o DataFrame se chama 'df' e deve retornar um resultado no final (tabela ou gráfico).
        '''
        with st.spinner("Gerando código com IA..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            code = response.choices[0].message.content

        st.subheader("Código gerado pela IA:")
        st.code(code, language="python")

        run_code = st.checkbox("Executar código?")
        if run_code:
            try:
                # Cria um contexto seguro
                local_vars = {'df': df, 'pd': pd, 'plt': plt, 'st': st}
                exec(code, {}, local_vars)
            except Exception as e:
                st.error(f"Erro ao executar o código: {e}")

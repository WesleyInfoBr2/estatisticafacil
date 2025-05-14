import streamlit as st
import pandas as pd
from openai import OpenAI
import io
import matplotlib.pyplot as plt
import seaborn as sns # Adicionado para gráficos padrão
import re
import csv

# Configure sua API Key via secrets.toml ou diretamente aqui
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# st.set_option("deprecation.showPyplotGlobalUse", False) # Removido/Comentado - Opção obsoleta

st.title("EstatísticaFácil - Seu Analista de Dados com IA")

st.markdown("Faça upload de um arquivo CSV, XLSX ou TXT e pergunte algo em linguagem natural. A IA vai gerar o código Python, exibir, e você pode decidir se quer rodar.")

# Inicializar st.session_state se não existir
if "df" not in st.session_state:
    st.session_state.df = None
if "show_initial_analysis" not in st.session_state:
    st.session_state.show_initial_analysis = False

uploaded_file = st.file_uploader("Faça upload do arquivo (CSV, XLSX, TXT)", type=["csv", "xlsx", "txt"])

if uploaded_file:
    # Limpar estado anterior ao carregar novo arquivo
    st.session_state.df = None 
    st.session_state.show_initial_analysis = False

    file_extension = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_extension == "csv":
            sniffer = csv.Sniffer()
            try:
                sample_bytes = uploaded_file.read(2048) 
                uploaded_file.seek(0) 
                sample_text = sample_bytes.decode("utf-8-sig") 
                dialect = sniffer.sniff(sample_text)
                st.session_state.df = pd.read_csv(uploaded_file, sep=dialect.delimiter, encoding="utf-8-sig", on_bad_lines="warn")
            except (csv.Error, UnicodeDecodeError) as e_sniff:
                st.warning(f"Não foi possível detectar o separador/encoding automaticamente para CSV: {e_sniff}. Tentando com separadores comuns (';' e ',') e encoding utf-8.")
                uploaded_file.seek(0)
                try:
                    st.session_state.df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", on_bad_lines="warn")
                except Exception:
                    uploaded_file.seek(0)
                    st.session_state.df = pd.read_csv(uploaded_file, sep=",", encoding="utf-8-sig", on_bad_lines="warn")
        elif file_extension == "xlsx":
            st.session_state.df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif file_extension == "txt": 
            st.info("Para arquivos TXT, tentaremos inferir o delimitador (tab, ';', ou ','). Pode ser necessário ajustar manualmente se a leitura falhar.")
            sniffer = csv.Sniffer()
            try:
                sample_bytes = uploaded_file.read(2048)
                uploaded_file.seek(0)
                sample_text = sample_bytes.decode("utf-8-sig")
                dialect = sniffer.sniff(sample_text)
                st.session_state.df = pd.read_csv(uploaded_file, sep=dialect.delimiter, encoding="utf-8-sig", on_bad_lines="warn")
            except (csv.Error, UnicodeDecodeError) as e_sniff_txt:
                st.warning(f"Não foi possível detectar o separador/encoding para TXT: {e_sniff_txt}. Tentando com tab, ';' e ','.")
                uploaded_file.seek(0)
                try:
                    st.session_state.df = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8-sig", on_bad_lines="warn")
                except Exception:
                    uploaded_file.seek(0)
                    try:
                        st.session_state.df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", on_bad_lines="warn")
                    except Exception:
                        uploaded_file.seek(0)
                        st.session_state.df = pd.read_csv(uploaded_file, sep=",", encoding="utf-8-sig", on_bad_lines="warn")
        else:
            st.error("Formato de arquivo não suportado. Por favor, faça upload de CSV, XLSX ou TXT.")
            st.session_state.df = None

    except Exception as e:
        st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
        st.session_state.df = None

if st.session_state.df is not None:
    df = st.session_state.df # Usar o df do session_state
    st.write("Visualização inicial dos dados:")
    st.dataframe(df.head())

    colunas_obj = df.select_dtypes(include="object").columns

    def coluna_e_numerica(serie):
        amostra = serie.dropna().astype(str)
        if amostra.empty:
            return False
        numeric_pattern = re.compile(r"^[-+]?(\d{1,3}([,.]\d{3})*|\d+)([,.]\d+)?%?$")
        match_count = amostra.head(min(len(amostra), 50)).apply(lambda x: bool(numeric_pattern.match(x.strip()))).sum()
        return match_count > len(amostra.head(min(len(amostra), 50))) * 0.7

    def robust_convert_to_numeric(val_str_orig, is_percentage_column_hint=False):
        if pd.isna(val_str_orig) or not isinstance(val_str_orig, str) or val_str_orig.strip().lower() == "nan" or val_str_orig.strip() == "":
            return None
        val_str = val_str_orig.strip()
        common_currencies = ["R$", "$", "€", "£", "USD", "BRL", "EUR"] 
        for cur in common_currencies:
            val_str = val_str.replace(cur, "")
        val_str = val_str.strip()
        is_percentage = is_percentage_column_hint
        if val_str.endswith("%"):
            is_percentage = True
            val_str = val_str[:-1].strip()
        if " " in val_str and (val_str.count(",") >= 1 or val_str.count(".") >= 1):
            parts_space = val_str.split(" ")
            if len(parts_space) > 1 and all(len(p) == 3 for p in parts_space[1:-1]) and (parts_space[-1].count(".") == 1 or parts_space[-1].count(",") == 1):
                val_str = val_str.replace(" ", "")
            elif len(parts_space) > 1 and all(p.isdigit() for p in parts_space[:-1]) and (parts_space[-1].count(".") == 1 or parts_space[-1].count(",") == 1):
                 val_str = val_str.replace(" ", "")
        num_dots = val_str.count(".")
        num_commas = val_str.count(",")
        cleaned_val_str = val_str
        if num_dots == 1 and num_commas == 0:
            cleaned_val_str = val_str 
        elif num_commas == 1 and num_dots == 0:
            cleaned_val_str = val_str.replace(",", ".")
        elif num_dots >= 1 and num_commas == 1:
            cleaned_val_str = val_str.replace(".", "").replace(",", ".")
        elif num_commas >= 1 and num_dots == 1:
            cleaned_val_str = val_str.replace(",", "")
        elif num_dots == 0 and num_commas == 0:
            cleaned_val_str = val_str
        try:
            numeric_val = float(cleaned_val_str)
            return numeric_val / 100.0 if is_percentage else numeric_val
        except ValueError:
            try:
                temp_val = re.sub(r"[^\d.,-]+", "", val_str_orig)
                if temp_val.count(",") > 0 and temp_val.count(".") > 0:
                    if temp_val.rfind(",") > temp_val.rfind("."):
                        temp_val = temp_val.replace(".","").replace(",",".")
                    else:
                        temp_val = temp_val.replace(",","")
                elif temp_val.count(",") > 0:
                    temp_val = temp_val.replace(",",".")
                numeric_val = float(temp_val)
                return numeric_val / 100.0 if is_percentage else numeric_val
            except ValueError:
                return None

    for col in colunas_obj:
        if coluna_e_numerica(df[col]):
            sample_for_percentage_check = df[col].dropna().astype(str).head(50)
            percentage_count = sample_for_percentage_check.str.endswith("%").sum()
            is_percentage_col = percentage_count > len(sample_for_percentage_check) * 0.5
            df[col] = df[col].apply(lambda x: robust_convert_to_numeric(x, is_percentage_col))
            try:
                if df[col].dropna().apply(lambda x: x == int(x) if pd.notnull(x) else True).all():
                    df[col] = df[col].astype("Int64")
            except Exception:
                pass 
    
    st.write("Tipos após tentativa de conversão robusta:")
    st.write(df.dtypes)
    st.session_state.df = df # Atualizar o df no session_state após conversão

    # --- Funcionalidade de Análise Inicial --- 
    st.session_state.show_initial_analysis = st.checkbox("Mostrar Análise Inicial (Estatísticas e Gráficos Padrão)?", value=st.session_state.show_initial_analysis)

    if st.session_state.show_initial_analysis:
        st.subheader("Análise Descritiva Inicial")
        st.write("Estatísticas Descritivas Gerais:")
        st.dataframe(df.describe(include="all"))

        colunas_numericas = df.select_dtypes(include=["number"]).columns
        colunas_categoricas = df.select_dtypes(include=["object", "category"]).columns

        if len(colunas_numericas) > 0:
            st.write("**Gráficos para Colunas Numéricas:**")
            col_num_selecionada = st.selectbox("Selecione uma coluna numérica para visualizar:", colunas_numericas)
            if col_num_selecionada:
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(df[col_num_selecionada].dropna(), kde=True, ax=ax_hist)
                ax_hist.set_title(f"Histograma de {col_num_selecionada}")
                st.pyplot(fig_hist)
                plt.close(fig_hist)

                fig_box, ax_box = plt.subplots()
                sns.boxplot(x=df[col_num_selecionada].dropna(), ax=ax_box)
                ax_box.set_title(f"Boxplot de {col_num_selecionada}")
                st.pyplot(fig_box)
                plt.close(fig_box)
        
        if len(colunas_categoricas) > 0:
            st.write("**Gráficos para Colunas Categóricas:**")
            col_cat_selecionada = st.selectbox("Selecione uma coluna categórica para visualizar (top 15 categorias):", colunas_categoricas)
            if col_cat_selecionada:
                fig_bar, ax_bar = plt.subplots()
                counts = df[col_cat_selecionada].value_counts().nlargest(15)
                sns.barplot(x=counts.index, y=counts.values, ax=ax_bar, palette="viridis")
                ax_bar.set_title(f"Contagem de Categorias em {col_cat_selecionada}")
                ax_bar.set_ylabel("Contagem")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig_bar)
                plt.close(fig_bar)
    # --- Fim da Funcionalidade de Análise Inicial ---

    question = st.text_input("O que você quer saber ou fazer com os dados?")
    
    if question:
        # Revisão das f-strings nos exemplos few-shot para garantir uso correto de aspas e interpolação
        prompt = f"""
        Você é um analista de dados em Python. Recebeu o seguinte DataFrame (df):
        Primeiras linhas:
        {df.head().to_string()}
        Tipos de dados das colunas:
        {df.dtypes.to_string()}

        A seguir, o usuário perguntou: "{question}"

        Gere apenas o código Python necessário para responder à pergunta, com comentários claros em cada etapa.
        A aplicação será executada em Streamlit, portanto use st.write(...), st.dataframe(...), st.pyplot(), st.plotly_chart(), etc. para exibir todas as saídas.
        Assuma que o DataFrame principal se chama `df`.
        Gere apenas o código Python executável, com comentários dentro do código. Não inclua explicações escritas fora do código.
        Gere o código apenas com bibliotecas já importadas (pandas as pd, matplotlib.pyplot as plt, streamlit as st, io, seaborn as sns).
        Não use `exec()` ou `eval()` no código gerado.
        Não tente ler ou escrever arquivos no sistema.
        Após qualquer cálculo ou visualização, inclua uma breve interpretação em linguagem humana do resultado, usando st.write() ou st.markdown().

        Exemplo de Pergunta 1: "Qual a média da coluna 'idade'?"
        Código de Resposta Esperado 1:
        ```python
        # Calcula a média da coluna 'idade'
        if 'idade' in df.columns:
            media_idade = df['idade'].mean()
            st.write(f"A média de idade é: {media_idade:.2f}")
            # Interpretação
            # Usando aspas triplas para a string principal do markdown e f-string para interpolação.
            # As quebras de linha dentro das aspas triplas são preservadas.
            st.markdown(f"""A idade média dos indivíduos na base de dados é de {media_idade:.2f} anos. 
            Isso nos dá uma medida central da faixa etária predominante.""" )
        else:
            st.warning("A coluna 'idade' não foi encontrada no DataFrame.")
        ```

        Exemplo de Pergunta 2: "Crie um histograma da coluna 'valor_compra'. Use seaborn."
        Código de Resposta Esperado 2:
        ```python
        # Cria um histograma para a coluna 'valor_compra' usando seaborn
        if 'valor_compra' in df.columns:
            if pd.api.types.is_numeric_dtype(df['valor_compra']):
                fig, ax = plt.subplots()
                sns.histplot(df['valor_compra'].dropna(), kde=True, ax=ax)
                ax.set_title("Histograma de Valor da Compra")
                ax.set_xlabel("Valor da Compra")
                ax.set_ylabel("Frequência")
                st.pyplot(fig)
                plt.close(fig) # Importante para liberar memória
                # Interpretação
                st.markdown(f"""O histograma acima mostra a distribuição dos valores de compra. 
                Podemos observar a frequência de compras em diferentes faixas de valor, 
                ajudando a identificar os tickets mais comuns.""" )
            else:
                st.warning("A coluna 'valor_compra' não é numérica e não pode ser usada para um histograma diretamente.")
        else:
            st.warning("A coluna 'valor_compra' não foi encontrada no DataFrame.")
        ```

        Exemplo de Pergunta 3: "Mostre as 5 primeiras linhas do dataframe"
        Código de Resposta Esperado 3:
        ```python
        # Mostra as 5 primeiras linhas do dataframe
        st.write("As 5 primeiras linhas do DataFrame são:")
        st.dataframe(df.head())
        # Interpretação
        st.markdown("As primeiras cinco linhas do conjunto de dados foram exibidas para uma rápida visualização da estrutura e do conteúdo inicial.")
        ```

        Pergunta do usuário: "{question}"
        Gere apenas o bloco de código Python, com comentários e a interpretação solicitada.
        Sempre use `st.pyplot(fig)` para exibir gráficos matplotlib/seaborn, nunca `plt.show()`.
        Para gráficos Plotly, use `st.plotly_chart(fig)`.
        Certifique-se de que qualquer figura matplotlib/seaborn seja fechada com `plt.close(fig)` após `st.pyplot(fig)`.
        Se a pergunta envolver a criação de um gráfico, gere o código para o gráfico e sua interpretação.
        Se a pergunta for sobre um cálculo, gere o código para o cálculo e sua interpretação.
        Se a pergunta for muito genérica, não relacionada a análise de dados no dataframe, ou pedir para modificar o dataframe `df` de forma destrutiva (ex: `df.dropna(inplace=True)`), informe que não pode realizar a ação ou sugira uma alternativa segura (como criar uma cópia).
        Lembre-se, o dataframe é `df`.
        Sua resposta deve ser apenas o bloco de código Python.
        Não inclua `import` statements no código gerado.
        Se a pergunta for sobre estatísticas descritivas de uma coluna, use `df['nome_coluna'].describe()` e apresente com `st.write()` ou `st.dataframe()`.
        Para visualizações, prefira `matplotlib.pyplot` ou `seaborn` (importado como `sns`).
        Se for usar `plt.subplots()`, sempre use `fig, ax = plt.subplots()`.
        Finalize a resposta.
        Sua resposta:
        f""" 
        with st.spinner("Gerando código com IA..."):
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 
            )
            raw_code = response.choices[0].message.content
            code = re.sub(r"^```(python)?", "", raw_code.strip(), flags=re.MULTILINE)
            code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE)

        with st.expander("Código gerado pela IA:"):
            st.code(code, language="python")
        
        st.warning("⚠️ **Atenção:** O código abaixo foi gerado por uma IA e pode ser editado. A execução de código desconhecido ou modificado pode apresentar riscos de segurança e instabilidade. Execute por sua conta e risco.")

        with st.expander("Edite o código se desejar (AVANÇADO):"):
            code_editado = st.text_area("Edite o código Python aqui:", code, height=300, help="Modifique o código com cuidado. Apenas usuários avançados.")

        run_code = st.checkbox("Sim, entendo os riscos e desejo executar o código.")
        
        if run_code:
            if not code_editado.strip():
                st.error("O campo de código está vazio. Não há nada para executar.")
            else:
                try:
                    allowed_builtins = {
                        "print": print, "len": len, "range": range, "list": list, "dict": dict, "str": str, "int": int, "float": float, "bool": bool,
                        "True": True, "False": False, "None": None, "max": max, "min": min, "sum": sum, "abs": abs, "round": round, "all": all, "any": any,
                        "isinstance": isinstance, "getattr": getattr, "hasattr": hasattr, "Exception": Exception, "ValueError": ValueError,
                        "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError, "sorted": sorted, "zip": zip, "map": map, "filter": filter,
                        "enumerate": enumerate, "reversed": reversed, "set": set, "tuple": tuple, "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
                        "complex": complex, "divmod": divmod, "pow": pow, "repr": repr, "slice": slice, "super": super, "vars": vars, "dir": dir,
                        "globals": lambda: {}, "locals": lambda: {} 
                    }
                    safe_globals = {
                        "__builtins__": allowed_builtins,
                        "pd": pd,
                        "plt": plt,
                        "sns": sns, 
                        "st": st,
                        "df": df.copy(), 
                        "io": io,
                    }
                    
                    try:
                        compile(code_editado, "<string>", "exec")
                    except SyntaxError as se:
                        st.error(f"Erro de Sintaxe no código fornecido: {se}")
                        st.stop()

                    exec(code_editado, safe_globals)

                except Exception as e:
                    st.error(f"Erro ao executar o código: {e}")
                    import traceback
                    with st.expander("Detalhes do Erro (Traceback)"):
                        st.text(traceback.format_exc())
else:
    if uploaded_file is None:
        st.info("Por favor, faça o upload de um arquivo para começar a análise.")

st.markdown("---_---")
st.markdown("Desenvolvido como um protótipo. Use com cautela.")





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
            new_sheet_name = st.session_state[selectbox_key_sheet]
            if new_sheet_name in st.session_state.dfs_dict:
                st.session_state.selected_sheet_name = new_sheet_name
                st.session_state.df = st.session_state.dfs_dict[new_sheet_name]
        selected_sheet_val = st.selectbox("Selecione a planilha para análise:", sheet_names_list, key=selectbox_key_sheet, index=default_sheet_index, on_change=update_df_on_sheet_selection_change)
        if st.session_state.get(selectbox_key_sheet) != st.session_state.selected_sheet_name:
             if st.session_state.get(selectbox_key_sheet) in st.session_state.dfs_dict:
                st.session_state.selected_sheet_name = st.session_state[selectbox_key_sheet]
                st.session_state.df = st.session_state.dfs_dict[st.session_state.selected_sheet_name]

    if st.session_state.df is not None:
        df = st.session_state.df
        current_widget_prefix = st.session_state.widget_key_prefix
        if st.session_state.selected_sheet_name and st.session_state.dfs_dict and len(st.session_state.dfs_dict) > 1 and uploaded_file.name.endswith(".xlsx"):
            st.write(f"Analisando planilha: **{st.session_state.selected_sheet_name}**")
        st.write("Visualização inicial dos dados:")
        st.dataframe(df.head())

        colunas_obj = df.select_dtypes(include="object").columns
        def coluna_e_numerica(serie):
            amostra = serie.dropna().astype(str)
            if amostra.empty: return False
            numeric_pattern = re.compile(r"^[-+]?(\d{1,3}([,.]\d{3})*|\d+)([,.]\d+)?%?$")
            match_count = amostra.head(min(len(amostra), 50)).apply(lambda x: bool(numeric_pattern.match(x.strip()))).sum()
            return match_count > len(amostra.head(min(len(amostra), 50))) * 0.7

        def robust_convert_to_numeric(val_str_orig, is_percentage_column_hint=False):
            if pd.isna(val_str_orig) or not isinstance(val_str_orig, str) or val_str_orig.strip().lower() == "nan" or val_str_orig.strip() == "": return None
            val_str = val_str_orig.strip()
            for cur in ["R$", "$", "€", "£", "USD", "BRL", "EUR"]: val_str = val_str.replace(cur, "")
            val_str = val_str.strip()
            is_percentage = is_percentage_column_hint
            if val_str.endswith("%"): is_percentage = True; val_str = val_str[:-1].strip()
            if " " in val_str and (val_str.count(",") >= 1 or val_str.count(".") >= 1):
                parts_space = val_str.split(" ")
                if len(parts_space) > 1 and all(len(p) == 3 for p in parts_space[1:-1]) and (parts_space[-1].count(".") == 1 or parts_space[-1].count(",") == 1):
                    val_str = val_str.replace(" ", "")
                elif len(parts_space) > 1 and all(p.isdigit() for p in parts_space[:-1]) and (parts_space[-1].count(".") == 1 or parts_space[-1].count(",") == 1):
                    val_str = val_str.replace(" ", "")
            num_dots, num_commas = val_str.count("."), val_str.count(",")
            cleaned_val_str = val_str
            if num_dots == 1 and num_commas == 0: cleaned_val_str = val_str 
            elif num_commas == 1 and num_dots == 0: cleaned_val_str = val_str.replace(",", ".")
            elif num_dots >= 1 and num_commas == 1: cleaned_val_str = val_str.replace(".", "").replace(",", ".")
            elif num_commas >= 1 and num_dots == 1: cleaned_val_str = val_str.replace(",", "")
            try: numeric_val = float(cleaned_val_str); return numeric_val / 100.0 if is_percentage else numeric_val
            except ValueError:
                try:
                    temp_val = re.sub(r"[^\d.,-]+", "", val_str_orig)
                    if temp_val.count(",") > 0 and temp_val.count(".") > 0:
                        temp_val = temp_val.replace(".","").replace(",",".") if temp_val.rfind(",") > temp_val.rfind(".") else temp_val.replace(",","")
                    elif temp_val.count(",") > 0: temp_val = temp_val.replace(",",".")
                    numeric_val = float(temp_val); return numeric_val / 100.0 if is_percentage else numeric_val
                except ValueError: return None

        df_processed = df.copy()
        for col in colunas_obj:
            if coluna_e_numerica(df_processed[col]):
                sample_perc = df_processed[col].dropna().astype(str).head(50)
                is_perc_col = sample_perc.str.endswith("%").sum() > len(sample_perc) * 0.5
                df_processed[col] = df_processed[col].apply(lambda x: robust_convert_to_numeric(x, is_perc_col))
                try:
                    if df_processed[col].dropna().apply(lambda x: pd.notnull(x) and x == int(x)).all():
                        df_processed[col] = df_processed[col].astype("Int64")
                except Exception: pass 
        st.session_state.df = df_processed
        df = st.session_state.df

        checkbox_key_analysis = f"show_analysis_checkbox_{current_widget_prefix}"
        st.session_state.show_initial_analysis = st.checkbox("Mostrar Análise Inicial (Estatísticas e Gráficos Padrão)?", 
                                                           value=st.session_state.get(checkbox_key_analysis, False), 
                                                           key=checkbox_key_analysis)

        if st.session_state.show_initial_analysis:
            st.subheader("Análise Descritiva Inicial")
            st.write("Estatísticas Descritivas Gerais:")
            desc = df.describe(include="all").transpose()
            desc_display = pd.DataFrame(index=desc.index)
            desc_display["Tipo da variável"] = df.dtypes.astype(str).values
            unique_counts_list = [df[col_name].nunique() for col_name in df.columns]
            desc_display["Contagem únicos"] = unique_counts_list
            rename_map = {"count": "Contagem", "top": "Valor Mais Comum", "freq": "Frequência Mais Comum", "mean": "Média", "std": "Desvio padrão", "min": "Mínimo", "25%": "25%", "50%": "Mediana", "75%": "75%", "max": "Máximo"}
            for orig, new in rename_map.items():
                if orig in desc.columns: desc_display[new] = desc[orig]
            desc_display["Primeiro Valor"] = [df[col].iloc[0] if not df[col].empty else None for col in df.columns]
            desc_display["Último Valor"] = [df[col].iloc[-1] if not df[col].empty else None for col in df.columns]
            cv_list = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
                    mean, std = df[col].mean(), df[col].std()
                    cv_list.append(f"{(std / mean) * 100:.2f}%" if mean != 0 and pd.notna(mean) and pd.notna(std) else None)
                else: cv_list.append(None)
            desc_display["Coeficiente de variação"] = cv_list
            order = ["Tipo da variável", "Contagem", "Contagem únicos", "Primeiro Valor", "Último Valor", "Valor Mais Comum", "Frequência Mais Comum", "Média", "Desvio padrão", "Mínimo", "25%", "Mediana", "75%", "Máximo", "Coeficiente de variação"]
            st.dataframe(desc_display[[c for c in order if c in desc_display.columns]])

            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(include=["object", "category"]).columns

            if len(num_cols) > 0:
                st.write("**Gráficos para Colunas Numéricas:**")
                col_num_key_select = f"col_num_select_{current_widget_prefix}"
                sel_num_col = st.selectbox("Selecione uma coluna numérica:", num_cols, key=col_num_key_select)
                if sel_num_col:
                    fig_hist, ax_hist = plt.subplots(); sns.histplot(df[sel_num_col].dropna(), kde=True, ax=ax_hist); ax_hist.set_title(f"Histograma de {sel_num_col}"); st.pyplot(fig_hist); plt.close(fig_hist)
                    fig_box, ax_box = plt.subplots(); box_data = df[sel_num_col].dropna()
                    if not box_data.empty: 
                        sns.boxplot(x=box_data, ax=ax_box); ax_box.set_title(f"Boxplot de {sel_num_col}")
                        Q1 = box_data.quantile(0.25)
                        Q3 = box_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = box_data[(box_data < Q1 - 1.5 * IQR) | (box_data > Q3 + 1.5 * IQR)]
                        if not outliers.empty: 
                            y_offset_increment = 0.03 # Ajuste este valor conforme necessário para o espaçamento vertical
                            y_current_offset = - (len(outliers) // 2) * y_offset_increment # Começar abaixo do centro
                            if len(outliers) % 2 == 0: # Ajuste para número par de outliers para centralizar melhor
                                y_current_offset += y_offset_increment / 2
                            
                            sorted_outliers = sorted(list(outliers))
                            last_x_pos = -float('inf')
                            min_x_separation = (box_data.max() - box_data.min()) * 0.02 # Mínima separação horizontal

                            for i, val in enumerate(sorted_outliers):
                                # Alternar entre posições acima e abaixo
                                if i % 2 == 0:
                                    y_offset = y_current_offset
                                    y_current_offset += y_offset_increment
                                else:
                                    y_offset = -y_current_offset
                                
                                # Aumentar o deslocamento vertical se os pontos estiverem muito próximos horizontalmente
                                if abs(val - last_x_pos) < min_x_separation:
                                    y_offset *= 1.5  # Aumentar o deslocamento vertical
                                
                                ax_box.annotate(f"{val:.2f}", xy=(val, 0), xytext=(0, y_offset), 
                                              textcoords="offset points", ha='center', va='center',
                                              bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                                last_x_pos = val
                        st.pyplot(fig_box); plt.close(fig_box)

            if len(cat_cols) > 0:
                st.write("**Gráficos para Colunas Categóricas:**")
                col_cat_key_select = f"col_cat_select_{current_widget_prefix}"
                sel_cat_col = st.selectbox("Selecione uma coluna categórica:", cat_cols, key=col_cat_key_select)
                if sel_cat_col:
                    value_counts = df[sel_cat_col].value_counts().reset_index()
                    value_counts.columns = [sel_cat_col, 'Contagem']
                    if len(value_counts) > 15: value_counts = value_counts.head(15); st.info("Mostrando apenas as 15 categorias mais frequentes.")
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=sel_cat_col, y='Contagem', data=value_counts, ax=ax_bar)
                    ax_bar.set_title(f"Contagem de {sel_cat_col}")
                    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig_bar); plt.close(fig_bar)

        st.subheader("Pergunte sobre seus dados")
        question_key_input = f"question_input_{current_widget_prefix}"
        question = st.text_input("O que você gostaria de saber sobre seus dados?", key=question_key_input)
        
        if question:
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
            
            Pergunta: "Qual a média de idade?"
            Resposta:
            ```python
            media_idade = df['Idade'].mean()
            st.markdown(f"A idade média dos indivíduos na base de dados é de {media_idade:.2f} anos.")
            ```
            
            Pergunta: "Mostre a distribuição de vendas por região"
            Resposta:
            ```python
            # Agrupar vendas por região
            vendas_por_regiao = df.groupby('Região')['Valor_Venda'].sum().reset_index()
            
            # Criar gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Região', y='Valor_Venda', data=vendas_por_regiao, ax=ax)
            ax.set_title('Distribuição de Vendas por Região')
            ax.set_ylabel('Valor Total de Vendas')
            st.pyplot(fig)
            plt.close(fig)
            ```
            
            Pergunta: "Quais são os 5 produtos mais vendidos?"
            Resposta:
            ```python
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
            ```
            """
            
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
            
            st.subheader("Código gerado pela IA:")
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
                    
                    # Incluir a função extrair_palavras_robusta no ambiente de execução
                    def extrair_palavras(texto):
                        return extrair_palavras_robusta(texto)
                    
                    allowed_builtins = {k: v for k, v in __builtins__.__dict__.items() if k in ["print", "len", "range", "list", "dict", "str", "int", "float", "bool", "True", "False", "None", "max", "min", "sum", "abs", "round", "all", "any", "isinstance", "getattr", "hasattr", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "sorted", "zip", "map", "filter", "enumerate", "reversed", "set", "tuple"]}
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
                        "extrair_palavras_robusta": extrair_palavras_robusta
                    }
                    
                    # Compilar o código para verificar erros de sintaxe
                    compile(code_editado, "<string>", "exec")
                    
                    # Executar o código em ambiente seguro
                    exec(code_editado, safe_globals)
                    
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

elif uploaded_file is None and st.session_state.last_uploaded_file_id is None:
    st.info("Por favor, faça o upload de um arquivo para começar a análise.")

st.markdown("---")
st.markdown("Desenvolvido como um protótipo. Use com cautela.")

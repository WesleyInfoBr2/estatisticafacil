import streamlit as st
import pandas as pd
from openai import OpenAI
import io
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv
import numpy as np

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
# widget_key_prefix é usado para garantir que os widgets sejam recriados quando um novo arquivo é carregado
if "widget_key_prefix" not in st.session_state: st.session_state.widget_key_prefix = "initial_prefix"


# --- Callback para resetar estado quando o arquivo é removido do uploader --- 
def on_file_uploader_change():
    # Esta função é chamada quando o st.file_uploader muda (arquivo carregado ou removido)
    # Se o widget file_uploader_main (que é o nosso uploaded_file) se tornar None, 
    # significa que o usuário limpou o arquivo.
    if st.session_state.file_uploader_main is None and st.session_state.last_uploaded_file_id is not None:
        # Arquivo foi removido pelo usuário
        st.session_state.df = None
        st.session_state.dfs_dict = None
        st.session_state.selected_sheet_name = None
        st.session_state.show_initial_analysis = False
        st.session_state.last_uploaded_file_id = None
        st.session_state.widget_key_prefix = f"cleared_{np.random.randint(10000)}" # Novo prefixo para forçar recriação de widgets
        # Limpar chaves de widgets dinâmicos da sessão anterior explicitamente
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

        # Resetar estado dependente do arquivo para um novo upload
        st.session_state.df = None
        st.session_state.dfs_dict = None
        st.session_state.selected_sheet_name = None
        # Não resetar show_initial_analysis aqui, pois o checkbox controlará seu próprio estado com a nova chave
        # st.session_state.show_initial_analysis = False 

        # Limpar chaves de widgets dinâmicos da sessão anterior (se o prefixo mudou)
        # Esta limpeza é mais para o caso de trocas rápidas de arquivos, o prefixo já ajuda.
        keys_to_delete = []
        for k in st.session_state.keys():
            is_dynamic_widget_key = any(k.startswith(p) for p in ["sheet_selector_", "show_analysis_checkbox_", 
                                                                  "col_num_select_", "col_cat_select_", 
                                                                  "question_input_", "code_editor_", "run_code_checkbox_"])
            if is_dynamic_widget_key and not k.startswith(st.session_state.widget_key_prefix):
                keys_to_delete.append(k)
        for key_del in keys_to_delete:
            if key_del in st.session_state: del st.session_state[key_del]

    # Se é um novo arquivo, processá-lo (ler e popular dfs_dict, df inicial)
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
                    if df_temp.shape[1] == 1: # Provavelmente leu errado, tentar com vírgula
                        uploaded_file.seek(0); df_temp = pd.read_csv(uploaded_file, sep=",", encoding="utf-8-sig", on_bad_lines="warn", skipinitialspace=True)
                st.session_state.dfs_dict = {"CSV_Data": df_temp}
                st.session_state.selected_sheet_name = "CSV_Data"

            elif file_extension == "xlsx":
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                if not sheet_names:
                    st.error("O arquivo XLSX não contém planilhas."); st.stop()
                st.session_state.dfs_dict = {name: excel_file.parse(name) for name in sheet_names}
                st.session_state.selected_sheet_name = sheet_names[0] # Padrão para a primeira planilha
            
            elif file_extension == "txt":
                # Lógica de leitura de TXT (simplificada, pode precisar de mais robustez)
                try:
                    sample_bytes = uploaded_file.read(2048); uploaded_file.seek(0)
                    dialect = csv.Sniffer().sniff(sample_bytes.decode("utf-8-sig"))
                    df_temp = pd.read_csv(uploaded_file, sep=dialect.delimiter, encoding="utf-8-sig", on_bad_lines="warn")
                except (csv.Error, UnicodeDecodeError):
                    uploaded_file.seek(0); df_temp = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8-sig", on_bad_lines="warn") # Tentar tab
                st.session_state.dfs_dict = {"TXT_Data": df_temp}
                st.session_state.selected_sheet_name = "TXT_Data"
            else:
                st.error("Formato de arquivo não suportado."); st.stop()
            
            # Definir o df inicial com base na primeira planilha/dados lidos
            if st.session_state.selected_sheet_name and st.session_state.dfs_dict:
                st.session_state.df = st.session_state.dfs_dict[st.session_state.selected_sheet_name]
            else:
                 st.error("Não foi possível carregar dados da planilha selecionada."); st.stop()

        except Exception as e:
            st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
            st.session_state.last_uploaded_file_id = None # Resetar para permitir novo processamento
            st.stop()

    # --- Lógica para seleção de planilha (para XLSX com múltiplas planilhas) --- 
    if uploaded_file.name.endswith(".xlsx") and st.session_state.dfs_dict and len(st.session_state.dfs_dict) > 1:
        sheet_names_list = list(st.session_state.dfs_dict.keys())
        selectbox_key_sheet = f"sheet_selector_{st.session_state.widget_key_prefix}"

        # Garantir que selected_sheet_name seja válido ou defina um padrão
        current_selected_sheet_from_session = st.session_state.get("selected_sheet_name", sheet_names_list[0])
        if current_selected_sheet_from_session not in sheet_names_list:
            current_selected_sheet_from_session = sheet_names_list[0]
        
        default_sheet_index = sheet_names_list.index(current_selected_sheet_from_session)

        def update_df_on_sheet_selection_change():
            new_sheet_name = st.session_state[selectbox_key_sheet]
            if new_sheet_name in st.session_state.dfs_dict:
                st.session_state.selected_sheet_name = new_sheet_name
                st.session_state.df = st.session_state.dfs_dict[new_sheet_name]
            # Não resetar outros estados aqui, apenas o df principal e selected_sheet_name.

        selected_sheet_val = st.selectbox(
            "Selecione a planilha para análise:", 
            sheet_names_list, 
            key=selectbox_key_sheet,
            index=default_sheet_index,
            on_change=update_df_on_sheet_selection_change
        )
        # Após o selectbox, garantir que selected_sheet_name e df estejam sincronizados com o valor do widget
        # O callback deve cuidar disso, mas uma verificação extra pode ser útil se o callback não rodar imediatamente no re-run
        if st.session_state[selectbox_key_sheet] != st.session_state.selected_sheet_name:
             if st.session_state[selectbox_key_sheet] in st.session_state.dfs_dict:
                st.session_state.selected_sheet_name = st.session_state[selectbox_key_sheet]
                st.session_state.df = st.session_state.dfs_dict[st.session_state.selected_sheet_name]

    # --- O restante do código que usa st.session_state.df --- 
    if st.session_state.df is not None:
        df = st.session_state.df # Usar o df da sessão
        current_widget_prefix = st.session_state.widget_key_prefix

        if st.session_state.selected_sheet_name and st.session_state.dfs_dict and len(st.session_state.dfs_dict) > 1 and uploaded_file.name.endswith(".xlsx"):
            st.write(f"Analisando planilha: **{st.session_state.selected_sheet_name}**")
        
        st.write("Visualização inicial dos dados:")
        st.dataframe(df.head())

        # ... (resto do código: coluna_e_numerica, robust_convert_to_numeric, processamento, checkboxes, gráficos, IA) ...
        # Substituir todas as chaves dinâmicas para usar current_widget_prefix
        # Ex: checkbox_key = f"show_analysis_checkbox_{current_widget_prefix}"

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
        # Usar st.session_state.get para o value do checkbox para evitar erro se a chave não existir no primeiro run
        st.session_state.show_initial_analysis = st.checkbox("Mostrar Análise Inicial (Estatísticas e Gráficos Padrão)?", 
                                                           value=st.session_state.get(checkbox_key_analysis, False), 
                                                           key=checkbox_key_analysis)

        if st.session_state.show_initial_analysis:
            st.subheader("Análise Descritiva Inicial")
            st.write("Estatísticas Descritivas Gerais:")
            desc = df.describe(include="all").transpose()
            desc_display = pd.DataFrame(index=desc.index)
            desc_display["Tipo da variável"] = df.dtypes.astype(str).values
            rename_map = {"count": "Contagem", "unique": "Contagem únicos", "top": "Valor Mais Comum", "freq": "Frequência Mais Comum", "mean": "Média", "std": "Desvio padrão", "min": "Mínimo", "25%": "25%", "50%": "Mediana", "75%": "75%", "max": "Máximo"}
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
                        Q1, Q3, IQR = box_data.quantile(0.25), box_data.quantile(0.75), Q3 - Q1
                        outliers = box_data[(box_data < Q1 - 1.5 * IQR) | (box_data > Q3 + 1.5 * IQR)]
                        if not outliers.empty: 
                            for val in outliers: ax_box.text(val, 0, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="red", bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
                        st.pyplot(fig_box); plt.close(fig_box)
                    else: st.write(f"Coluna {sel_num_col} sem dados para boxplot.")
            
            if len(cat_cols) > 0:
                st.write("**Gráficos para Colunas Categóricas:**")
                col_cat_key_select = f"col_cat_select_{current_widget_prefix}"
                sel_cat_col = st.selectbox("Selecione uma coluna categórica (top 15):", cat_cols, key=col_cat_key_select)
                if sel_cat_col:
                    fig_bar, ax_bar = plt.subplots(); counts = df[sel_cat_col].value_counts().nlargest(15); sns.barplot(x=counts.index, y=counts.values, ax=ax_bar, palette="viridis"); ax_bar.set_title(f"Contagem em {sel_cat_col}"); ax_bar.set_ylabel("Contagem"); plt.xticks(rotation=45, ha="right"); st.pyplot(fig_bar); plt.close(fig_bar)

        question_key_input = f"question_input_{current_widget_prefix}"
        question = st.text_input("O que você quer saber ou fazer com os dados?", key=question_key_input)
        
        if question:
            prompt = f"""... (prompt da IA inalterado) ...""" # Mantido o prompt original por brevidade
            # (O prompt completo da IA está no código original e é longo)
            # Certifique-se de que o prompt completo seja usado aqui.
            # Para este exemplo, vou usar um prompt placeholder:
            prompt_completo = f"""
            Você é um analista de dados em Python. Recebeu o seguinte DataFrame (df):
            Primeiras linhas:
            {df.head().to_string()}
            Tipos de dados das colunas:
            {df.dtypes.to_string()}
            Pergunta do usuário: "{question}"
            Gere apenas o código Python necessário para responder à pergunta, com comentários claros e interpretação usando st.write() ou st.markdown().
            Use bibliotecas já importadas: pandas as pd, matplotlib.pyplot as plt, streamlit as st, io, seaborn as sns, numpy as np.
            Não use exec() ou eval(). Não leia/escreva arquivos. Feche figuras plt com plt.close(fig).
            Exemplo de média: media = df['col'].mean(); st.write(f"Média: {media}"); st.markdown("Interpretação...")
            Exemplo de histograma: fig, ax = plt.subplots(); sns.histplot(df['col'], ax=ax); st.pyplot(fig); plt.close(fig); st.markdown("Interpretação...")
            Sua resposta (apenas o bloco de código Python):
            """

            with st.spinner("Gerando código com IA..."):
                response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt_completo}], temperature=0.1)
                raw_code = response.choices[0].message.content
                code = re.sub(r"^```(python)?", "", raw_code.strip(), flags=re.MULTILINE)
                code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE)

            with st.expander("Código gerado pela IA:"):
                st.code(code, language="python")
            st.warning("⚠️ **Atenção:** O código abaixo foi gerado por uma IA. Revise antes de executar.")
            
            code_editor_key_area = f"code_editor_{current_widget_prefix}_{question[:15].replace(' ','_')}"
            with st.expander("Edite o código se desejar (AVANÇADO):"):
                code_editado = st.text_area("Edite o código Python aqui:", code, height=300, key=code_editor_key_area)

            run_code_key_check = f"run_code_checkbox_{current_widget_prefix}_{question[:15].replace(' ','_')}"
            run_code = st.checkbox("Sim, entendo os riscos e desejo executar o código.", key=run_code_key_check)
            
            if run_code and code_editado.strip():
                try:
                    allowed_builtins = {k: v for k, v in __builtins__.__dict__.items() if k in ["print", "len", "range", "list", "dict", "str", "int", "float", "bool", "True", "False", "None", "max", "min", "sum", "abs", "round", "all", "any", "isinstance", "getattr", "hasattr", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "sorted", "zip", "map", "filter", "enumerate", "reversed", "set", "tuple"]}
                    safe_globals = {"__builtins__": allowed_builtins, "pd": pd, "plt": plt, "sns": sns, "np": np, "st": st, "df": df.copy(), "io": io}
                    compile(code_editado, "<string>", "exec") # Validar sintaxe antes
                    exec(code_editado, safe_globals)
                except Exception as e:
                    st.error(f"Erro ao executar o código: {e}")
                    with st.expander("Detalhes do Erro (Traceback)"):
                        st.text(traceback.format_exc())
            elif run_code and not code_editado.strip():
                 st.error("O campo de código está vazio.")

elif uploaded_file is None and st.session_state.last_uploaded_file_id is None:
    st.info("Por favor, faça o upload de um arquivo para começar a análise.")

st.markdown("---_---")
st.markdown("Desenvolvido como um protótipo. Use com cautela.")



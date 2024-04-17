import streamlit as st
import langchain_helper as lch
import sql_helper as sql
import textwrap
import tempfile
from pathlib import Path
import os

st.set_page_config(layout="wide", initial_sidebar_state = "expanded", page_title= "Prompt Lab", page_icon= ":brain:")

if 'upload' not in st.session_state:
    st.session_state.upload = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'loader' not in st.session_state:
    st.session_state.loader = "PyPDF"
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 100000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 100
if 'overwrite' not in st.session_state:
    st.session_state.overwrite = True
    
def submit_contexto (upload):
    if upload:
        temp_dir = tempfile.mkdtemp()
        uploaded_file_path = os.path.join(temp_dir, upload.name)
        st.session_state.collection_name = "LAB_" + upload.name
        with open(uploaded_file_path, "wb") as f:
            f.write(upload.getvalue())
        lch.create_vector_from_pdf(st.session_state.collection_name, 
                                   uploaded_file_path,
                                   st.session_state.loader,
                                   st.session_state.chunk_size,
                                   st.session_state.chunk_overlap,
                                   st.session_state.overwrite
                                   )
        contexto_container.write(f":green[Upload de arquivo {st.session_state.collection_name} bem sucedido!]")

lab_collections_result = sql.get_lab_collections()
lab_collections = []

st.title(":brain: Prompt Lab", anchor=False)
results_container = st.container(border=True)
with results_container:   
    st.subheader("Resultados:", divider="grey", anchor=False)

for l in lab_collections_result:
    lab_collections.insert(0, l[0])

with st.sidebar:   
    st.header(":gear: Configurações", divider="grey", anchor=False)
    contexto_container = st.container(border=True)
    
    with contexto_container:
        st.subheader(":blue_book: Contexto:", divider="grey", anchor=False)
        st.session_state.contexto = st.selectbox(
        'Selecione um contexto do PgVector:',
        (lab_collections),  
        index= None)
         
        if not st.session_state.contexto:
            with st.expander(":three_button_mouse: Upload de novo contexto", expanded=False):
                st.session_state.upload = st.file_uploader("Upload PDF", type="pdf")
                st.session_state.loader = st.selectbox(
                    "PDF Loader", 
                    ("PyPDFLoader", "PyPDFLoader - ExtractImages", 
                     "UnstructuredPDFLoader", "UnstructuredPDFLoader - ElementsMode", 
                     "PyPDFium2Loader", "PDFMinerLoader", 
                     "PyMuPDFLoader", "PDFPlumberLoader"),
                    index= 0) 
                st.session_state.chunk_size = st.number_input("Particionamento (Caracteres)", value= st.session_state.chunk_size)
                st.session_state.chunk_overlap = st.number_input("Sobreposição (Caracteres)", value= st.session_state.chunk_overlap)
                st.session_state.overwrite = st.checkbox("Sobreescrever arquivo de mesmo nome", st.session_state.overwrite)
                st.button(label=":arrow_up: Upload de arquivo", on_click = submit_contexto, args=(st.session_state.upload,))
    
    with st.form(key="my_form_question"):
        st.subheader(":question: Prompt:", divider="grey", anchor=False)
        prompt = st.text_area(
        label="Prompt de comando", key="enviar prompt")
        submit_query_button = st.form_submit_button(label=":arrow_forward: Enviar Prompt")
        
if st.session_state.contexto:
    with results_container:
        with st.expander(":three_button_mouse: Resultado do particionamento", expanded=False):
            st.write(sql.get_documents_by_collection_name(st.session_state.contexto))





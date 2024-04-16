import streamlit as st
import langchain_helper as lch
import textwrap
import tempfile
from pathlib import Path
import os

collection_name = ""

with st.sidebar:   
    st.title("Agente Inteligente PDF")
    with st.form(key="my_form_file", clear_on_submit=True):
        uploaded_file = st.file_uploader("File upload", type="pdf")
        submit_button_for_file = st.form_submit_button(label="Processar arquivo")   
    with st.form(key="my_form_question"):
        query = st.text_area(
            label="Prompt de comando", key="enviar prompt"
        )
        submit_button_for_question = st.form_submit_button(label="Enviar Prompt")
    
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

if submit_button_for_file and uploaded_file:
    collection_name = uploaded_file.name
    docs = lch.create_vector_from_pdf(collection_name, path)
    st.write(f"Arquivo {collection_name} processado com sucesso. Envie prompt para analisá-lo")

if submit_button_for_question and query:
    response, docs = lch.get_response_from_query(collection_name, query)
    st.subheader("Resposta:")
    st.text(textwrap.fill(response["answer"]))




import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Agente Inteligente")

with st.sidebar:   
    index_name = st.sidebar.text_input(label="Nome do Agente", max_chars=150, key="index_name") 
    with st.form(key="my_form_file"):
        pdf_url = st.text_input(label="Caminho do Arquivo", max_chars=150)
        submit_button_for_file = st.form_submit_button(label="Carregar Arquivo")

    with st.form(key="my_form_site"):
        site_url = st.text_input(label="URL do Site", max_chars=400)
        submit_button_for_site = st.form_submit_button(label="Carregar Site")

    with st.form(key="my_form_yt_video"):
        youtube_url = st.text_input(label="URL do Video", max_chars=150)
        submit_button_for_yt_video = st.form_submit_button(label="Carregar Video")

    with st.form(key="my_form_question"):
        query = st.text_area(
            label="Me pergunte algo!", max_chars=150, key="query"
        )
        submit_button_for_question = st.form_submit_button(label="Enviar Pergunta")
    

if submit_button_for_file and pdf_url and index_name:
    docs = lch.create_vector_from_pdf(pdf_url)
    lch.embed_and_store_document_splits(docs, index_name)
    st.write("Arquivo processado com sucesso!")

if submit_button_for_site:
    docs = lch.create_vector_from_html_url(site_url)
    lch.embed_and_store_document_splits(docs, index_name)
    st.write("Site processado com sucesso!")

if submit_button_for_yt_video:
    docs = lch.create_vector_from_yt_url(youtube_url)
    lch.embed_and_store_document_splits(docs, index_name)
    st.write("Vídeo processado com sucesso!")

if submit_button_for_question and query:
    response, docs = lch.get_response_from_query(index_name, query)
    st.subheader("Resposta:")
    st.text(textwrap.fill(response["answer"]))



import streamlit as st
import langchain_helper as lch
import sql_helper as sql
import tempfile
from pathlib import Path
import os

# CONFIGS PÁGINA
st.set_page_config(layout="wide", initial_sidebar_state = "expanded", page_title= "Prompt Lab", page_icon= ":brain:")

# VARIÁVEIS DE SESSÃO
    #CONTEXTO
if 'contexto' not in st.session_state:
    st.session_state.contexto = None
if 'tipo_contexto' not in st.session_state:
    st.session_state.tipo_contexto = "PDF"
if 'upload' not in st.session_state:
    st.session_state.upload = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'loader' not in st.session_state:
    st.session_state.loader = "PyPDF"
if 'dpi' not in st.session_state:
    st.session_state.dpi = 500
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 100000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 100
if 'overwrite' not in st.session_state:
    st.session_state.overwrite = True
    
    #RETRIEVER BASE
if 'retrieval_active' not in st.session_state:
    st.session_state.retrieval_active = False
if 'query_retriever_base' not in st.session_state:
    st.session_state.query_retriever_base = ""
if 'retriever_type' not in st.session_state:
    st.session_state.retriever_type = None
if 'base_retriever' not in st.session_state:
    st.session_state.base_retriever = None
if 'base_retriever_result' not in st.session_state:
    st.session_state.base_retriever_result = None
if 'search_type' not in st.session_state:
    st.session_state.search_type = "similarity"
if 'retriever_top_k' not in st.session_state:
    st.session_state.retriever_top_k = 10
if 'child_chunk_size' not in st.session_state:
    st.session_state.child_chunk_size = 1000
if 'child_chunk_overlap' not in st.session_state:
    st.session_state.child_chunk_overlap = 100
    
    #RETRIEVER CONTEXTUAL
if 'contextual_retriever_type' not in st.session_state:
    st.session_state.contextual_retriever_type = None
if 'ctx_retriever' not in st.session_state:
    st.session_state.ctx_retriever = None
if 'ctx_retriever_result' not in st.session_state:
    st.session_state.ctx_retriever_result = None
if 'contextual_retriever_temperature' not in st.session_state:
    st.session_state.contextual_retriever_temperature = True
if 'contextual_retriever_openai_model_name' not in st.session_state:
    st.session_state.contextual_retriever_openai_model_name = True
    
    #RETRIEVER MULTIQUERY
if 'mutiquery_retriever_type' not in st.session_state:
    st.session_state.mutiquery_retriever_type = None
if 'multiquery_retriever' not in st.session_state:
    st.session_state.multiquery_retriever = None
if 'multiquery_retriever_result' not in st.session_state:
    st.session_state.multiquery_retriever_result = None
    
    #QUERY
if 'query' not in st.session_state:
    st.session_state.query = None
if 'prompt_digitado' not in st.session_state:
    st.session_state.prompt_digitado = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None
if 'enforce_json' not in st.session_state:
    st.session_state.enforce_json = False
if 'json_schema' not in st.session_state:
    st.session_state.json_schema = """{
    "type": "function",
    "function": {
        "name": "invoice",
        "description": "An invoice document",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_number": {
                    "description": "The invoice document alphanumeric number",
                    "type": "string"
                },
                "delivery_address": {
                    "description": "Delivery address for the invoice, also called Consignee. Sometimetimes it will be the same as the Billing address",
                    "type": "string"
                },
                "billing_address": {
                    "description": "Billing address for the invoice. Sometimetimes it will be the same as the Delivery or Consignee address",
                    "type": "string"
                },
                "favorite_food": {
                    "description": "The persons favorite food",
                    "type": "string"
                }
            }
        }
    }
}"""
if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = """Please examine carefully and provide responses based on the following user instructions:
Document:
{context}
User Instructions: 
{question}"""
if 'openai_model_name' not in st.session_state:
    st.session_state.openai_model_name = "gpt-3.5-turbo"
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'chain_type' not in st.session_state:
    st.session_state.chain_type = "create_stuff_documents_chain"
    
    #RESULT FINAL 
if 'result_final' not in st.session_state:
    st.session_state.result_final = None
    
# FUNÇÕES
    #SUBMIT CONTEXTO    
def submit_pdf_vetorizado (input_collection):
    if st.session_state.upload:
        temp_dir = tempfile.mkdtemp()
        uploaded_file_path = os.path.join(temp_dir, st.session_state.upload.name)
        st.session_state.collection_name = input_collection
        with open(uploaded_file_path, "wb") as f:
            f.write(st.session_state.upload.getvalue())
        lch.create_vector_from_pdf(st.session_state.collection_name, 
                                   uploaded_file_path,
                                   st.session_state.loader,
                                   st.session_state.dpi,
                                   st.session_state.chunk_size,
                                   st.session_state.chunk_overlap,
                                   st.session_state.overwrite
                                   )
        contexto_container.write(f":green[Upload de arquivo {st.session_state.collection_name} bem sucedido!]")
    else: contexto_container.write(":red[Selecione um arquivo antes!]")

def submit_texto (input_collection):
    if st.session_state.upload and st.session_state.upload != "":
        st.session_state.collection_name = input_collection
        lch.create_vector(st.session_state.collection_name, 
                            st.session_state.upload,
                            True,
                            st.session_state.chunk_size,
                            st.session_state.chunk_overlap,
                            st.session_state.overwrite
                            )
        contexto_container.write(":green[Upload do textp bem sucedido!]")
    else: contexto_container.write(":red[Defina o texto antes!]")

    #QUERY
def submit_query():
    if (st.session_state.prompt_digitado and st.session_state.prompt_digitado != ""
    and st.session_state.prompt_template != ""):
        st.session_state.query = st.session_state.prompt_digitado
        query_container.write(":green[Query definida com sucesso!]")
    

    #RETRIEVER BASE
def get_base_retriever():
    if st.session_state.collection_name and st.session_state.query:
        if st.session_state.retriever_type == "Vectorstore":
            st.session_state.base_retriever = lch.get_retriever_vectorstore(
                st.session_state.collection_name, 
                st.session_state.search_type, st.session_state.retriever_top_k)
        if st.session_state.retriever_type == "ParentDocument":
            st.session_state.base_retriever = lch.get_retriever_parent_document(
                st.session_state.collection_name,
                st.session_state.child_chunk_size, st.session_state.child_chunk_overlap)
        
def submit_base_retriever():
    if st.session_state.query_retriever_base and st.session_state.query_retriever_base != "" and st.session_state.contexto:
        result_retriever = st.session_state.base_retriever.get_relevant_documents(
            st.session_state.query_retriever_base)
        st.session_state.base_retriever_result = []
        for r in result_retriever:
            st.session_state.base_retriever_result.append({getattr(r, "page_content")})
        retriever_container.write(f":green[Retriever base {st.session_state.retriever_type} testado com sucesso!]")
    else:
        retriever_container.write(":red[Defina um contexto e query antes!]")
    
    #RETRIEVER CONTEXTUAL
def get_contextual_retriever():   
    if st.session_state.contextual_retriever_type == "Contextual compression - LLMChainExtractor":
        st.session_state.ctx_retriever = lch.get_retriever_ctx_comp_llmchainextractor(
            st.session_state.base_retriever, 
            st.session_state.contextual_retriever_temperature, 
            st.session_state.contextual_retriever_openai_model_name)
    if st.session_state.contextual_retriever_type == "Contextual compression - LLMChainFilter":
        st.session_state.ctx_retriever = lch.get_retriever_ctx_comp_llmchainfilter(
            st.session_state.base_retriever,
            st.session_state.contextual_retriever_temperature, 
            st.session_state.contextual_retriever_openai_model_name)    
    if st.session_state.contextual_retriever_type == "Contextual compression - EmbeddingsFilter":
        st.session_state.ctx_retriever = lch.get_retriever_ctx_comp_embeddingsfilter(
            st.session_state.base_retriever, 
            st.session_state.contextual_retriever_temperature)
        
def submit_contextual_retriever():
    if st.session_state.query and st.session_state.query != "" and st.session_state.contexto:
        st.session_state.ctx_retriever_result = st.session_state.ctx_retriever.get_relevant_documents(
            st.session_state.query
        )
        retriever_container.write(f":green[Retriever contextual {st.session_state.retriever_type} testado com sucesso!]")
    else:
        retriever_container.write(":red[Defina um contexto e query antes!]")
        
    #SUBMIT FINAL
def submit_final():
    if ((st.session_state.base_retriever or not st.session_state.retrieval_active) and 
            st.session_state.openai_model_name and 
            st.session_state.temperature and
            st.session_state.prompt_template and
            st.session_state.query and
            st.session_state.chain_type):
        if st.session_state.retrieval_active:
            docs = None
            base_rtv = st.session_state.base_retriever
        else:
            docs = st.session_state.contexto
            base_rtv = None
        st.session_state.result_final = lch.get_response_from_prompt_and_retriever(
                docs = docs, 
                base_retriever= base_rtv, 
                contextual_retriever= None, 
                openai_model_name= st.session_state.openai_model_name, 
                temperature= st.session_state.temperature, 
                prompt_template=  st.session_state.prompt_template, 
                prompt= st.session_state.query,
                retriever_prompt= st.session_state.query_retriever_base, 
                chain_type= st.session_state.chain_type,
                json_schema= st.session_state.json_schema,
                enforce_json= st.session_state.enforce_json,
                chat_history= st.session_state.chat_history
                )
        st.sidebar.write(":green[Query executada com sucesso!]")
    else:
        st.sidebar.write(":red[Dados obrigatórios incompletos!]")

 
# MAIN
# BUSCA COLLECTIONS       
lab_collections_result = sql.get_lab_collections()
lab_collections = []
for l in lab_collections_result:
    lab_collections.insert(0, l[0])

#TÍTULO E CONTAINERS
st.title(":brain: Prompt Lab", anchor=False)
results_final_container = st.container(border=True)
with results_final_container:   
    st.subheader("Resultados Finais:", divider="grey", anchor=False)
    st.button(":brain: Executar", on_click= submit_final)
results_container = st.container(border=True)
with results_container:   
    st.subheader("Resultados Parciais:", divider="grey", anchor=False)

#SIDEBAR
with st.sidebar:   
    st.header(":gear: Configurações", divider="grey", anchor=False)
    #CONTAINERS SIDEBAR
    contexto_container = st.container(border=True)
    query_container = st.container(border=True)
    retriever_container = st.container(border=True)
    
    #CONTEXTO
    with contexto_container:
        st.subheader(":blue_book: Contexto:", divider="grey", anchor=False)
        
        st.selectbox(
        ":open_file_folder: Selecione uma coleção do PgVector:",
        (lab_collections), index=None , 
        key='collection_name')
        
        if not st.session_state.collection_name:
            st.session_state.base_retriever_result = None
            st.session_state.ctx_retriever_result = None
            st.session_state.multiquery_retriever_result = None
            with st.expander(":three_button_mouse: Upload de novo contexto", expanded=False):
                st.selectbox(
                    'Selecione o tipo de contexto:',
                    ("PDF", "Texto"), key='tipo_contexto')
                #PDF
                if st.session_state.tipo_contexto == "PDF":
                    st.session_state.upload = st.file_uploader("Upload PDF", type="pdf")
                    if st.session_state.upload:
                        input_collection = st.text_input(label= "Nome da coleção",value= st.session_state.upload.name)
                    else:
                        input_collection = ""
                    st.session_state.loader = st.selectbox(
                        "PDF Loader", 
                        ("PyPDFLoader", 
                        "UnstructuredPDFLoader", "UnstructuredPDFLoader - ElementsMode", "PyMuPDFLoader - FindTables", 
                        "PyMuPDFLoader", "PyTesseract - OCR", "PubLayNet - OCR"),
                        index= 0) 
                    if st.session_state.loader == "PyTesseract - OCR" or st.session_state.loader == "PubLayNet - OCR":
                        st.session_state.dpi = st.number_input("Densidade de pixels", value= st.session_state.dpi)
                #TEXTO
                if st.session_state.tipo_contexto == "Texto":
                    input_collection = st.text_input(label= "Nome da coleção")
                    st.text_area(key="upload",
                    label="Digite o contexto")
                
                #PARTICIONAMENTO
                st.session_state.chunk_size = st.number_input("Particionamento (Caracteres)", value= st.session_state.chunk_size)
                st.session_state.chunk_overlap = st.number_input("Sobreposição (Caracteres)", value= st.session_state.chunk_overlap)
                st.session_state.overwrite = st.checkbox("Sobreescrever coleção de mesmo nome", st.session_state.overwrite)
                
                #SUBMIT
                if input_collection and input_collection != "":
                    if st.session_state.tipo_contexto == "PDF":
                        st.button(label=":arrow_up: Salvar novo arquivo", on_click = submit_pdf_vetorizado, args= ("LAB_" + input_collection,))
                    if st.session_state.tipo_contexto == "Texto":
                        st.button(label=":arrow_up: Salvar novo texto", on_click = submit_texto, args= ("LAB_" + input_collection,))
        else:
            st.session_state.contexto = sql.get_documents_by_collection_name(st.session_state.collection_name) 
            
#QUERY            
    with query_container:
        st.subheader(":question: Query:", divider="grey", anchor=False)
        st.selectbox(
            "Modelo Open AI", 
            ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"),
            key= "openai_model_name")
        st.selectbox(
            "Tipo de Chain", 
            ("create_stuff_documents_chain", "create_structured_output_runnable"),
            key= "chain_type") 
        st.session_state.temperature = st.number_input("Temperatura", value= st.session_state.temperature)           
        if st.session_state.chain_type == "create_history_aware_retriever":
            if not st.session_state.retrieval_active:
                st.write(":red[Obrigatório selecionar um retriever para esta chain!]")
            st.session_state.chat_history = f"""{st.session_state.result_final["callback_llm"]["full_prompt"]}
            System_response: {st.session_state.result_final["result"]}"""
            st.session_state.chat_history = st.text_area(
            label="Histórico de chat", value = st.session_state.chat_history)
        if st.session_state.chain_type == "create_structured_output_runnable":
            st.session_state.json_schema = st.text_area(
            label="JSON Schemas para output", value= st.session_state.json_schema)
            st.checkbox("Formato JSON obrigatório", key="enforce_json")
            if not st.session_state.prompt_digitado or st.session_state.prompt_digitado == "":
                st.session_state.prompt_digitado = "Use the given format to extract the information from the document. If you can't find any of the information, use None as it's value." 
        st.session_state.prompt_template = st.text_area(
        label="Template do prompt", value = st.session_state.prompt_template)
        st.session_state.prompt_digitado = st.text_area(
        label="Prompt de comando", value = st.session_state.prompt_digitado)
        submit_query()
        checkbox_retriver = st.checkbox("Ativar Retrival vetorial", key="retrieval_active")
        if checkbox_retriver:
            st.session_state.query_retriever_base = st.session_state.query

#RETRIEVER BASE        
    if st.session_state.retrieval_active:
        with retriever_container:
            st.subheader(":mag: Retriever:", divider="grey", anchor=False)
            with st.expander(":three_button_mouse: Retriever Base", expanded=False):
                st.session_state.retriever_type = st.selectbox(
                'Selecione o Retriever base:',
                ("Vectorstore", "nenhum"))
                #VECTORSTORE
                if st.session_state.retriever_type == "Vectorstore":
                    st.session_state.search_type = st.selectbox(
                        'Tipo de busca:',
                        ("similarity", "mmr"))
                    st.session_state.retriever_top_k = st.number_input("Top K (Documentos selecionados)", value= st.session_state.retriever_top_k)
                    
                st.session_state.query_retriever_base = st.text_area(
                label="Prompt retriver base", value= st.session_state.query_retriever_base)
            get_base_retriever()
            st.button(label=":test_tube: Testar retriever base", on_click = submit_base_retriever)
    else:
        st.session_state.base_retriever_result = None
        st.session_state.ctx_retriever_result = None
        st.session_state.multiquery_retriever_result = None
        st.session_state.base_retriever = None
        st.session_state.ctx_retriever = None
        st.session_state.multiquery_retriever = None

#CONTAINER RESULTADO FINAL       
with results_final_container:
    if st.session_state.result_final:
        with st.expander(":three_button_mouse: Resultado Final", expanded=True):
            st.write(st.session_state.result_final["result"])
        with st.expander(":three_button_mouse: Tokens Open AI", expanded=False):
            st.write(st.session_state.result_final["callback_open_ai"])
        with st.expander(":three_button_mouse: Prompt Completo", expanded=False):
            st.write(st.session_state.result_final["callback_llm"]["full_prompt"])
                    
#CONTAINER RESULTADOS PARCIAIS
    # CONTEXTO        
if st.session_state.contexto and st.session_state.collection_name:
    with results_container:
        with st.expander(":three_button_mouse: Contexto Selecionado", expanded=False):
            st.write(f":blue_book: Coleção {st.session_state.collection_name}:")
            st.write(st.session_state.contexto)

    # QUERY        
if st.session_state.query:
    with results_container:
        with st.expander(":three_button_mouse: Query Selecionada", expanded=False):
            st.write(":question: Query:")
            st.write(f'\"{st.session_state.query}\"')

    # RETRIEVERS        
if st.session_state.base_retriever:
    with results_container:
        with st.expander(":three_button_mouse: Retriever Base Selecionado", expanded=False):
            st.write(f":mag: Retriver Base: {st.session_state.retriever_type}")
            st.write(f":question: Prompt do retriever: {st.session_state.query_retriever_base}")
            if st.session_state.base_retriever_result:
                st.write(":test_tube: Resultado Teste:")
                st.write(st.session_state.base_retriever_result)
    





import json
from PIL import Image, ImageEnhance, ImageFilter
from typing import List
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import BaseCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
import langchain.document_loaders as document_loaders
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_structured_output_runnable, create_history_aware_retriever
from langchain.docstore.document import Document
import fitz
from pydantic import BaseModel, Field
import pytesseract
import layoutparser as lp
from langchain import hub
import streamlit as st
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\LucasBreder\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" // somente localmente  


from dotenv import load_dotenv
import os

load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

# CONNECTION_STRING = PGVector.connection_string_from_db_params(
#         driver=os.getenv("PGVECTOR_DRIVER"),
#         host=os.getenv("PGVECTOR_HOST"),
#         port=int(os.getenv("PGVECTOR_PORT")),
#         database=os.getenv("PGVECTOR_DATABASE"),
#         user=os.getenv("PGVECTOR_USER"),
#         password=os.getenv("PGVECTOR_PASSWORD"),
# )

CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=st.secrets["PGVECTOR_DRIVER"],
        host=st.secrets["PGVECTOR_HOST"],
        port=int(st.secrets["PGVECTOR_PORT"]),
        database=st.secrets["PGVECTOR_DATABASE"],
        user=st.secrets["PGVECTOR_USER"],
        password=st.secrets["PGVECTOR_PASSWORD"],
)

def load_pdf(file_path, pdf_loader, dpi):
    
    if pdf_loader == "UnstructuredPDFLoader":
        loader = document_loaders.UnstructuredPDFLoader(file_path)
    if pdf_loader == "UnstructuredPDFLoader - ElementsMode":
        loader = document_loaders.UnstructuredPDFLoader(file_path, mode="elements")
    if pdf_loader == "PyPDFLoader":
        loader = document_loaders.PyPDFLoader(file_path)
    if pdf_loader == "PyMuPDFLoader - FindTables":
        return load_tables(file_path)
    if pdf_loader == "PyMuPDFLoader":
        loader = document_loaders.PyMuPDFLoader(file_path)
    if pdf_loader == "PyTesseract - OCR":
        return load_pytesseract(file_path, dpi)                     
    return loader.load()

def load_pytesseract(file_path, dpi):
    doc = fitz.open(file_path)
    docs = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = ImageEnhance.Color(img).enhance(0)
        img = ImageEnhance.Contrast(img).enhance(2)
        img = img.filter(ImageFilter.MedianFilter)
        img = img.filter(ImageFilter.GaussianBlur(radius = 0.5))
        page_text = pytesseract.image_to_string(img, lang="eng+fra+por+spa+deu")
        docs.append(Document(page_content=page_text))
        
    return docs
    
def load_tables(file_path):
    doc = fitz.open(file_path)
    docs = []
    for p in doc.pages():
        tabs = p.find_tables(strategy = "line")
        for t in tabs:
            docs.append(Document(page_content=t.to_pandas()))
    return docs


def create_vector_from_pdf(collection_name, file_path, pdf_loader_type, dpi, chunk_size, chunk_overlap, overwrite):          
    data = load_pdf(file_path, pdf_loader_type, dpi)

    return create_vector(collection_name, data, False, chunk_size, chunk_overlap, overwrite)
    
def create_vector(collection_name, data, create_docs, chunk_size, chunk_overlap, overwrite):          
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if create_docs:
        text = text_splitter.create_documents([data])
    else:
        text = data
    
    docs = text_splitter.split_documents(text)

    return PGVector.from_documents(
        documents=docs, 
        embedding=embeddings, 
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=overwrite,)

def get_retriever_vectorstore(collection_name,search_type, retriever_top_k):
    store = PGVector(
    collection_name= collection_name,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    )
    return store.as_retriever(search_type=search_type, search_kwargs={"k": retriever_top_k})
    
def get_retriever_parent_document(collection_name, child_chunk_size, child_chunk_overlap):
    store = PGVector(
    collection_name=collection_name,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    )
    docstore = InMemoryStore()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)
    
    return ParentDocumentRetriever(
        vectorstore=store,
        docstore=docstore,
        child_splitter=child_splitter
        )

def get_retriever_ctx_comp_llmchainextractor(base_retriever, contextual_retriever_temperature, contextual_retriever_openai_model_name):    

        llm = ChatOpenAI(temperature=contextual_retriever_temperature, 
                     model = contextual_retriever_openai_model_name,
                    openai_api_key=openai_api_key,)
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
        base_compressor = compressor, base_retriever = base_retriever)
    
def get_retriever_ctx_comp_llmchainfilter(base_retriever, contextual_retriever_temperature, contextual_retriever_openai_model_name):    
    llm = ChatOpenAI(temperature=contextual_retriever_temperature, 
                    model = contextual_retriever_openai_model_name,
                openai_api_key=openai_api_key,)
    _filter = LLMChainFilter.from_llm(llm)
    return ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=base_retriever)

def get_retriever_ctx_comp_embeddingsfilter(base_retriever, contextual_similarity_threshold):    
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=contextual_similarity_threshold)
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=base_retriever)

class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

def get_retriever_multiquery(base_retriever, multiquery_retriever_temperature,multiquery_retriever_openai_model_name,parser_key):
    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    llm = ChatOpenAI(temperature=multiquery_retriever_temperature, 
                    model = multiquery_retriever_openai_model_name,
                openai_api_key=openai_api_key,)

    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
    
    return MultiQueryRetriever(
    retriever=base_retriever, llm_chain=llm_chain, parser_key=parser_key)


class CustomHandler(BaseCallbackHandler):
    response_llm = {}
    def on_llm_start(
        self, serialized, prompts, **kwargs,):
        self.response_llm["full_prompt"] = "\n".join(prompts)

def get_response_from_prompt_and_retriever(docs, base_retriever, 
        contextual_retriever, openai_model_name, temperature, 
        prompt_template, prompt, retriever_prompt, chain_type,
        json_schema, enforce_json, chat_history):
    handler = CustomHandler()
    llm = ChatOpenAI(
        temperature=temperature,
        model=openai_model_name,
        openai_api_key=openai_api_key,
        callbacks = [handler]
    )
    prompt_template_obj = PromptTemplate( template=prompt_template, input_variables=['context', 'query'] )
    
    if docs:
        final_context = []
        for d in docs:
            final_context.append(Document(page_content=d[0]))
    else:
        if contextual_retriever:
            final_context = contextual_retriever.get_relevant_documents(
            retriever_prompt)
        else:
            final_context = base_retriever.get_relevant_documents(
            retriever_prompt)

    with get_openai_callback() as cb:
        if chain_type == "create_stuff_documents_chain":
            chain = create_stuff_documents_chain(llm, prompt_template_obj)
            result = chain.invoke({"context": final_context, "question": prompt})
        
        if chain_type == "create_structured_output_runnable":
            chain = create_structured_output_runnable(output_schema = json.loads(json_schema),
                                                      llm = llm, prompt = prompt_template_obj,
                                                        enforce_function_usage= enforce_json, 
                                                        mode="openai-tools"
                                                      )
            result = chain.invoke({"context": final_context, "question": prompt})
        
        if chain_type == "create_history_aware_retriever":
            rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
            chain = create_history_aware_retriever(llm,base_retriever, rephrase_prompt)
            
            result = chain.invoke({"input": prompt, "chat_history": chat_history})
           
    return {"result": result, "callback_open_ai" : cb, "callback_llm": handler.response_llm}
    
        
    
    


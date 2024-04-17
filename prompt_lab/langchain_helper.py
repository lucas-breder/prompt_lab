from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import langchain.document_loaders as document_loaders

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.getenv("PGVECTOR_DRIVER"),
        host=os.getenv("PGVECTOR_HOST"),
        port=int(os.getenv("PGVECTOR_PORT")),
        database=os.getenv("PGVECTOR_DATABASE"),
        user=os.getenv("PGVECTOR_USER"),
        password=os.getenv("PGVECTOR_PASSWORD"),
)


def create_vector_from_pdf(collection_name, file_path, pdf_loader, chunk_size, chunk_overlap, overwrite):
    
    if pdf_loader == "UnstructuredPDFLoader":
        loader = document_loaders.UnstructuredPDFLoader(file_path)
    if pdf_loader == "UnstructuredPDFLoader - ElementsMode":
        loader = document_loaders.UnstructuredPDFLoader(file_path, mode="elements")
    if pdf_loader == "PyPDFLoader":
        loader = document_loaders.PyPDFLoader(file_path)
    if pdf_loader == "PyPDFLoader - ExtractImages":
        loader = document_loaders.PyPDFLoader(file_path, extract_images=True)
    if pdf_loader == "PyPDFium2Loader":
        loader = document_loaders.PyPDFium2Loader(file_path)
    if pdf_loader == "PDFMinerLoader":
        loader = document_loaders.PDFMinerLoader(file_path)
    if pdf_loader == "PyMuPDFLoader":
        loader = document_loaders.PyMuPDFLoader(file_path)
    if pdf_loader == "PDFPlumberLoader":
        loader = document_loaders.PDFPlumberLoader(file_path)     
              
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)

    return PGVector.from_documents(
        documents=docs, 
        embedding=embeddings, 
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=overwrite,)

def get_response_from_query(collection_name, query, openai_model_name="gpt-3.5-turbo"):
    
    store = PGVector(
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(
        temperature=0.7,
        model=openai_model_name,
        openai_api_key=openai_api_key,
    )

    prompt_template = """
    Please examine carefully and provide responses to the following user inquiries regarding this task:
    Context:
    {context}
    User question: 
    {query}
    Respond to the user using JSON format:
    """

    QA_PROMPT = PromptTemplate( template=prompt_template, input_variables=['context', 'query'] )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type_kwargs={"prompt": QA_PROMPT},
        verbose=True
    )

    return qa_chain({"query": query})



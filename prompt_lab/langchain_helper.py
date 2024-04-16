from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

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


def create_vector_from_pdf(collection_name, file_path):

    CR_Template_Normal = PyPDFLoader(file_path).load()
    text_splitter_CR_Template_Normal = CharacterTextSplitter(chunk_overlap=100)
    CR_Template_Normal_content = text_splitter_CR_Template_Normal.split_documents(CR_Template_Normal)

    PGVector.from_documents(
        documents=CR_Template_Normal_content, 
        embedding=embeddings, 
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,)


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



from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Redis
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def create_vector_from_yt_url(video_url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(video_url, language="pt")
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        docs = text_splitter.split_documents(transcript)

        return docs
    
    except:
         print("ERROR: Could not create/save embeddings")

def create_vector_from_html_url(url: str):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        docs = text_splitter.split_documents(data)
        return docs
    except Exception as e:
         print("ERROR: Could not create/save embeddings.Details:\n" + str(e) + "\n") 

def create_vector_from_pdf(filePath: str):
    try:
        loader = PyPDFLoader(filePath)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        docs = text_splitter.split_documents(data)
        return docs
    except Exception as e:
         print("ERROR: Could not create/save embeddings.Details:\n" + str(e) + "\n")

def embed_and_store_document_splits(splits, indexName) -> Redis:
    try:
        print("Saving chunk with " + str(len(splits)) + " splits to Vector DB (Redis)..." )

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        db = Redis.from_documents(
            splits,
            embeddings, 
            redis_url="redis://localhost:6379",  
            index_name=indexName
        ) 
        return db

    except:
         print("ERROR: Could not create/save embeddings")

def get_response_from_query(indexName, query, k=4):
    db = Redis.from_existing_index(
        embedding=embeddings, 
        schema=[],
        redis_url="redis://localhost:6379", 
        index_name=indexName)
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key,
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """Você é um assistente da empresa Elogroup que responde perguntas sobre conteudos carregados de diversas maneiras. 
                Sao elas: transcricoes de videos do youtube, carregamento de documentos pdfs e carregamento de conteudo por url de site

        Responda a seguinte pergunta: {pergunta}
        Procurando nas seguintes transcrições: {docs}

        Use somente informação da transcrição para responder a pergunta, e sempre que for se referir a uma empresa utilize o nome Elogroup. Se você não sabe, responda
        com "Eu não sei".

        Suas respostas devem ser bem detalhadas e verbosas.
        """,
            )
        ]
    )

    chain = LLMChain(llm=llm, prompt=chat_template, output_key="answer")
    response = chain({"pergunta": query, "docs": docs_page_content})

    return response, docs

if __name__ == "__main__":
    #docs = create_vector_from_yt_url("https://www.youtube.com/watch?v=Z2SGE3_2Grg&list=PLyqOvdQmGdTTRYIm2jPsirBviWc2k0rdt")
    #docs = create_vector_from_html_url("https://aws.amazon.com/pt/what-is/neural-network/#:~:text=Uma%20rede%20neural%20%C3%A9%20um,camadas%2C%20semelhante%20ao%20c%C3%A9rebro%20humano.")
    #docs = create_vector_from_pdf("/Users/macbookpro/Downloads/macbook-pro-16inch-nov23-BR03406036-info.pdf")
    #db = embed_and_store_document_splits(docs, 'teste01')
    indexName = "teste01"
    response, docs = get_response_from_query(indexName, "o que e rede neural ?")
    print(response)
    
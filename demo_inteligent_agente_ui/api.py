#to run the server: uvicorn api:app --reload
#to explorer swagger ui: http://127.0.0.1:8000/docs

from fastapi import FastAPI
import langchain_helper as lch

app = FastAPI()

@app.post("/process-file")
async def process_file(pdf_url: str, index_name: str):
    docs = lch.create_vector_from_pdf(pdf_url)
    lch.embed_and_store_document_splits(docs, index_name)
    return {"message": "Arquivo processado com sucesso!"}

@app.post("/process-site")
async def process_site(site_url: str, index_name: str):
    docs = lch.create_vector_from_html_url(site_url)
    lch.embed_and_store_document_splits(docs, index_name)
    return {"message": "Site processado com sucesso!"}

@app.post("/process-video")
async def process_video(youtube_url: str, index_name: str):
    docs = lch.create_vector_from_yt_url(youtube_url)
    lch.embed_and_store_document_splits(docs, index_name)
    return {"message": "Vídeo processado com sucesso!"}

@app.get("/get-response/{index_name}/{query}")
async def get_response(index_name: str, query: str):
    response, docs = lch.get_response_from_query(index_name, query)
    return {"answer": response["answer"]}
from fastapi import FastAPI
from rag_qa import qa
app = FastAPI()


@app.get("/query", summary='Query with llamaindex+openai')
def root(question:str):
    return qa(question)
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
import os.path


colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

PERSIST_DIR = "storage"
if not os.path.exists(f'{PERSIST_DIR}/docstore.json'):
    documents = SimpleDirectoryReader("rag_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

def qa(question:str):
    query_engine = index.as_query_engine(node_postprocessors=[colbert_reranker])
    response = query_engine.query(question.strip())
    response.response=response.response.strip()
    return response
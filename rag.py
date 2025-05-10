from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import List, Tuple
import torch

class Rag:
    def __init__(self, faiss_index_path: str = "faiss_index"):
        self.faiss_index_path = faiss_index_path
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if os.path.exists(self.faiss_index_path):
            self.vector_store = FAISS.load_local(
                self.faiss_index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

    def load_file(self, path: str) -> List[Document]:
        file = TextLoader(path)
        return file.load()
    
    def chunks(self, data: List[Document]) -> List[Document]:
        if not data:
            raise ValueError("No documents provided for chunking")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", "#", "##", "**Q:**"]
        )
        return splitter.split_documents(data)
    
    def get_len(self, data: List[Document]) -> int:
        if not data or not isinstance(data, list):
            raise ValueError("Invalid document list")
        return len(data[0].page_content)
    
    def store_in_faiss(self, documents: List[Document], force_recreate: bool = False) -> FAISS:
        if not documents:
            raise ValueError("No documents provided for FAISS storage")
        
        if os.path.exists(self.faiss_index_path) and not force_recreate:
            self.vector_store = FAISS.load_local(
                self.faiss_index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.vector_store.save_local(self.faiss_index_path)
        
        return self.vector_store
    
    def retriever(self, query: str) -> List[Tuple[Document, float]]:

        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Run store_in_faiss first.")
        
        return self.vector_store.similarity_search_with_relevance_scores(query=query, k=3)

if __name__ == "__main__":
    rag = Rag(faiss_index_path="faiss_index")
    
    query = "What is your product?"
    results = rag.retriever(query)
    print(len(results))
    # for i, (result, score) in enumerate(results):
    #     print(f"Result {i+1} (score: {score:.4f}): {result.page_content[:100]}... (Source: {result.metadata['source']})")

    context_tex = "\n\n######################################\n\n".join([doc.page_content for doc, _socre in results])
    print(context_tex)
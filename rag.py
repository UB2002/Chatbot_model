from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from typing import List, Tuple
import google.generativeai as genai
from langchain_core.embeddings import Embeddings


class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str = None):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            result = [genai.embed_content(model=self.model_name, content=text, task_type="RETRIEVAL_DOCUMENT")["embedding"] for text in texts]
            return result
        except Exception as e:
            raise ValueError(f"Error embedding documents with Gemini: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(model=self.model_name, content=text, task_type="RETRIEVAL_QUERY")["embedding"]
            return result
        except Exception as e:
            raise ValueError(f"Error embedding query with Gemini: {str(e)}")

class Rag:
    def __init__(self, faiss_index_path: str = "/app/faiss_index", embedding_model_name: str = "models/text-embedding-004", document_path: str = None):
        self.faiss_index_path = faiss_index_path
        self.embedding_model = GeminiEmbeddings(model_name=embedding_model_name)
        self.vector_store = None
        
        if os.path.exists(self.faiss_index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {self.faiss_index_path}")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Will recreate index...")
                self.vector_store = None
        
        if self.vector_store is None and document_path:
            try:
                if not os.path.exists(document_path):
                    raise FileNotFoundError(f"Document file not found: {document_path}")
                documents = self.load_file(document_path)
                chunks = self.chunks(documents)
                self.store_in_faiss(chunks, force_recreate=True)
                print(f"Created new FAISS index from {document_path}")
            except Exception as e:
                print(f"Error initializing FAISS index from {document_path}: {e}")
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
            try:
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Recreating index...")
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
                self.vector_store.save_local(self.faiss_index_path)
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
    try:
        rag = Rag(faiss_index_path="faiss_index")
        
        # Example: Load and store documents
        documents = rag.load_file("dataset/FAQ.markdown")  # Replace with actual file path
        chunks = rag.chunks(documents)
        rag.store_in_faiss(chunks, force_recreate=True)  # Force recreate for new embeddings
        
        query = "What is your product?"
        results = rag.retriever(query)
        print(f"Retrieved {len(results)} results")
        
        context_text = "\n\n######################################\n\n".join([doc.page_content for doc, _score in results])
        print(context_text)
    except Exception as e:
        print(f"Error: {str(e)}")
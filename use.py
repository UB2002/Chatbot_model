from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
import os
from rag import Rag
import google.generativeai as genai

class GeminiRAGChatbot:
    def __init__(self, faiss_index_path="faiss_index", document_path: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.rag = Rag(faiss_index_path=faiss_index_path, document_path=document_path)
        self.history = []
    
    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc, _ in docs])
    
    def retrieve(self, query: str) -> List:
        return self.rag.retriever(query)
    
    def generate_answer(self, query: str, context: str) -> str:
        try:
            prompt = f"""Answer the question based on the provided context. If you cannot find the answer in the context, 
say that you don't know but try to provide general information related to the question.

Context: {context}

Question: {query}

Answer:"""
            
            response = self.model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                answer = response.text
            else:
                answer = response.parts[0].text if hasattr(response, 'parts') else str(response)
            self.history.append({"role": "user", "parts": [query]})
            self.history.append({"role": "model", "parts": [answer]})
            
            return answer
        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            raise e
    
    def chat(self, query: str) -> Dict[str, Any]:
        try:
            retrieved_docs = self.retrieve(query)
            context = self.format_docs(retrieved_docs)
            answer = self.generate_answer(query, context)
            
            return {
                "answer": answer,
                "sources": [{"content": doc.page_content,
                            "source": doc.metadata.get("source", "Unknown"),
                            "score": score}
                            for doc, score in retrieved_docs]
            }
        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "error": str(e)
            }
    
    def clear_history(self):
        self.history = []

if __name__ == "__main__":
    try:
        chatbot = GeminiRAGChatbot()
        
        query = "What is your product?"
        print(f"Sending query: {query}")
        
        response = chatbot.chat(query)
        
        print(f"\nQuestion: {query}\n")
        print(f"Answer: {response['answer']}\n")
        
        if "sources" in response and response["sources"]:
            print("Sources:")
            for i, source in enumerate(response["sources"]):
                print(f"Source {i+1} (score: {source['score']:.4f}):")
                print(f"From: {source['source']}")
                print(f"Content preview: {source['content'][:150]}...\n")
        else:
            print("No sources returned.")
        
        if "error" in response:
            print(f"Error details: {response['error']}")
    except Exception as e:
        print(f"Uncaught exception: {str(e)}")
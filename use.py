from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
import os
from rag import Rag
import google.generativeai as genai

class GeminiRAGChatbot:
    def __init__(self, faiss_index_path="faiss_index"):
        # Load API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize Gemini 1.5 Flash model
        print("Initializing Gemini 1.5 Flash model...")
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize RAG component
        self.rag = Rag(faiss_index_path=faiss_index_path)
        
        # Initialize conversation history
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
            
            print("Sending prompt to Gemini Flash...")
            
            # Generate response using the content generation API
            response = self.model.generate_content(prompt)
            
            # Extract text content from response
            if hasattr(response, 'text'):
                answer = response.text
            else:
                # Alternative way to access response text
                answer = response.parts[0].text if hasattr(response, 'parts') else str(response)
            
            print(f"Received response from Gemini Flash (length: {len(answer)} chars)")
            
            # Update history with actual user query and answer
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
            print(f"Error in chat method: {str(e)}")
            # Make sure to include the 'answer' key even in error cases
            return {
                "answer": f"An error occurred: {str(e)}",
                "error": str(e)
            }
    
    def clear_history(self):
        self.history = []

if __name__ == "__main__":
    try:
        print("Initializing GeminiRAGChatbot...")
        chatbot = GeminiRAGChatbot()
        
        query = "What is your product?"
        print(f"Sending query: {query}")
        
        print("Retrieving and generating response...")
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
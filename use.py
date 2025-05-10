from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
import os
from rag import Rag
import torch
class QwenRAGChatbot:
    def __init__(self, model_id="Qwen/Qwen3-0.6B", faiss_index_path="faiss_index"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=self.access_token).to(self.device)
        self.rag = Rag(faiss_index_path=faiss_index_path)
        self.history = []
    
    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc, _ in docs])
    
    def retrieve(self, query: str) -> List:
        return self.rag.retriever(query)
    
    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""Answer the question based on the provided context. If you cannot find the answer in the context, 
say that you don't know but try to provide general information related to the question.

Context:
{context}

Question: {query}

Answer:"""
        
        messages = self.history + [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=1024)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Update history with the actual user query (not the augmented prompt)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
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
            return {"error": str(e)}
            
    def clear_history(self):
        self.history = []


if __name__ == "__main__":
    chatbot = QwenRAGChatbot()
    
    query = "What is your product?"
    response = chatbot.chat(query)
    
    print(f"Question: {query}\n")
    print(f"Answer: {response['answer']}\n")
    print("Sources:")
    for i, source in enumerate(response.get("sources", [])):
        print(f"Source {i+1} (score: {source['score']:.4f}):")
        print(f"From: {source['source']}")
        print(f"Content preview: {source['content'][:150]}...\n")
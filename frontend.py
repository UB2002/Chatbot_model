import streamlit as st
from use import QwenRAGChatbot  # assuming your main logic is in use.py

# Initialize chatbot (load once)
@st.cache_resource
def load_chatbot():
    return QwenRAGChatbot()

chatbot = load_chatbot()

st.title("🔍 Qwen RAG Chatbot")

# Input from the user
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Generating response..."):
        response = chatbot.chat(user_query)

    if "error" in response:
        st.error(f"Error: {response['error']}")
    else:
        st.markdown("### 🧠 Answer:")
        st.write(response["answer"])

        st.markdown("### 📚 Sources:")
        for i, source in enumerate(response.get("sources", [])):
            st.markdown(f"**Source {i+1}** (score: {source['score']:.4f})")
            st.markdown(f"- **From:** {source['source']}")
            st.markdown(f"- **Content preview:** {source['content'][:300]}...")

# Button to clear chat history
if st.button("Clear history"):
    chatbot.clear_history()
    st.success("History cleared.")

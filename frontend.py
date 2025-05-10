import streamlit as st
from use import GeminiRAGChatbot  # Import from your use.py file
import time

# Set page configuration
st.set_page_config(
    page_title="Gemini RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better appearance with black background and white text
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #333333;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #ffffff;
        color: #ffffff;
    }
    .score-badge {
        background-color: #ffffff;
        color: #000000;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .answer-box {
        background-color: #222222;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 6px solid #ffffff;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #ffffff;
    }
    .stSpinner > div {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chatbot (load once)
@st.cache_resource
def load_chatbot():
    with st.spinner("Initializing Gemini 1.5 Flash model..."):
        return GeminiRAGChatbot()

# Title and description
st.markdown("<h1 class='main-header'>🔍 Gemini RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("""
This chatbot uses Google's Gemini 1.5 Flash model with Retrieval-Augmented Generation (RAG) 
to provide accurate answers with source information from your knowledge base.
""")

# Try to load the chatbot
try:
    chatbot = load_chatbot()
except Exception as e:
    st.error(f"Failed to initialize the chatbot: {str(e)}")
    st.stop()

# Store conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a clear history button
if st.button("🧹 Clear Chat History", use_container_width=True):
    try:
        chatbot = load_chatbot()
        chatbot.clear_history()
        st.success("✅ Chat history has been cleared!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from the user
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = chatbot.chat(user_query)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    # Display answer
                    answer_placeholder = st.empty()
                    answer_placeholder.markdown(f"<div class='answer-box'>{response['answer']}</div>", unsafe_allow_html=True)
                    
                    # Display sources if available
                    if "sources" in response and response["sources"]:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(
                                    f"""<div class='source-box'>
                                        <strong>Source {i+1}</strong> <span class='score-badge'>Score: {source['score']:.4f}</span>
                                        <br><strong>From:</strong> {source['source']}
                                        <br><strong>Preview:</strong> {source['content'][:300]}...
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No specific sources found for this query.")
                
                # Store assistant response in history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.get("answer", "Error generating response")
                })
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
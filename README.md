# Gemini RAG-Powered Multi-Agent Q&A Chatbot

This project implements a Retrieval-Augmented Generation (RAG) powered chatbot using Google's Gemini 1.5 Flash model, integrated with a FAISS vector store for efficient document retrieval and a Streamlit-based web interface for user interaction. The system is designed to answer queries based on a small document collection, with an agentic workflow to handle different types of queries.

## Architecture

The chatbot consists of three main components, each implemented in a separate Python file:

1. **RAG Component (`rag.py`)**

   - Handles document ingestion, chunking, and storage in a FAISS vector store.
   - Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
   - Retrieves the top 3 relevant document chunks for a given query.

2. **Chatbot Logic (`use.py`)**

   - Integrates the RAG component with Google's Gemini 1.5 Flash model.
   - Manages conversation history and generates answers based on retrieved context.
   - Returns answers along with source information and relevance scores.

3. **Web Interface (`frontend.py`)**
   - Provides a user-friendly interface using Streamlit.
   - Displays chat history, answers, and source information with relevance scores.
   - Includes features like clearing chat history and custom styling for a black-themed UI.

## Key Design Choices

- **RAG with FAISS**: FAISS was chosen for its efficiency in similarity search, enabling fast retrieval of relevant document chunks.
- **Gemini 1.5 Flash**: Selected for its cost-effectiveness and performance in natural language generation, suitable for a demo-scale project.
- **Streamlit UI**: Used for rapid development of a responsive web interface, with custom CSS for a polished look.
- **Modular Design**: Separating RAG, chatbot logic, and frontend into distinct files enhances maintainability and scalability.
- **Error Handling**: Comprehensive error handling ensures robustness, with meaningful error messages displayed to users.
- **Environment Variables**: API keys are managed via a `.env` file for security.

## Prerequisites

- Python 3.8+
- A Google Gemini API key (set as `GEMINI_API_KEY` in a `.env` file)
- GitHub repository: [https://github.com/UB2002/Chatbot_model](https://github.com/UB2002/Chatbot_model)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/UB2002/Chatbot_model.git
   cd Chatbot_model
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your Gemini API key:
     ```plaintext
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

1. **Prepare Documents**:

   - Place your text documents (e.g., FAQs, product specs) in a directory.
   - Update the `rag.py` script to load these documents using the `load_file` method.

2. **Run the Streamlit App**:

   ```bash
   streamlit run frontend.py
   ```

3. **Interact with the Chatbot**:
   - Open your browser to the URL provided by Streamlit (typically `http://localhost:8501`).
   - Type your question in the chat input box.
   - View the answer, retrieved context, and source information with relevance scores.
   - Use the "Clear Chat History" button to reset the conversation.

## Running the CLI (Optional)

To test the chatbot logic without the web interface:

```bash
python use.py
```

## This will run a sample query ("What is your product?") and print the answer and sources to the console.

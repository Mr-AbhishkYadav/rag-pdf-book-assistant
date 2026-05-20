# rag-pdf-book-assistant
A Retrieval-Augmented Generation(RAG) application built with python,streamlit and Langchain. It allows users to upload PDF books, intelligently processes and chunks the text, stores the embeddings in a Chroma vector database, and leverages Mistral AI to answer user queries with strict contextual accuracy.

# 📚 RAG PDF Book Assistant

An intelligent, context-aware document assistant that allows you to have a conversation with any PDF book or document. This project utilizes **Retrieval-Augmented Generation (RAG)** to ensure that the AI answers questions based *only* on the contents of the uploaded text, eliminating hallucinations.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **LLM & Embeddings:** Mistral AI (`mistral-small-latest` & `mistral-embed`)
* **Vector Store:** Chroma DB
* **Document Loader:** PyPDF


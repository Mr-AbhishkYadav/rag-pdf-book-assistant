import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            try:
                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # Split Document into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                # CRITICAL FIX: Ensure chunks actually contain valid text content
                # Filter out empty or broken documents before passing to embeddings
                valid_chunks = [
                    doc for doc in chunks 
                    if doc.page_content and len(doc.page_content.strip()) > 0
                ]

                if not valid_chunks:
                    st.error("The uploaded PDF yielded no readable text content.")
                else:
                    # Initialize Embeddings explicitly passing the API key
                    embeddings = MistralAIEmbeddings(
                        model="mistral-embed", 
                        mistral_api_key=MISTRAL_API_KEY
                    )

                    # Create vector store (Persists automatically in newer Chroma versions)
                    vectorstore = Chroma.from_documents(
                        documents=valid_chunks,
                        embedding=embeddings,
                        persist_directory="chroma_db"
                    )
                    
                    st.success("Vector database created successfully!")
            
            except Exception as e:
                st.error(f"An error occurred during database creation: {e}")
            
            finally:
                # Clean up temporary file safely
                if os.path.exists(file_path):
                    os.remove(file_path)

if os.path.exists("chroma_db"):
    embeddings = MistralAIEmbeddings(
        model="mistral-embed", 
        mistral_api_key=MISTRAL_API_KEY
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-latest", # Recommended stable alias for mistral-small
        mistral_api_key=MISTRAL_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Searching and generating answer..."):
            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

            st.write("### AI Answer")
            st.write(response.content)
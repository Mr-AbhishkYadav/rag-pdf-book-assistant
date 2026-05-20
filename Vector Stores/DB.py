from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()
embeddings = MistralAIEmbeddings(model="mistral-embed")
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings_model=MistralAIEmbeddings()

vectorstore=Chroma.from_documents(
    documents= docs,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)
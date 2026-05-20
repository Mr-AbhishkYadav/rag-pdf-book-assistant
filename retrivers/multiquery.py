# from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# from langchain.chat_models import init_chat_model
# from dotenv import load_dotenv

# load_dotenv()

# docs = [
#     Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
#     Document(page_content="Gradient descent minimizes the loss function."),
#     Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
#     Document(page_content="Neural networks use gradient descent for training."),
#     Document(page_content="Support Vector Machines are supervised learning algorithms.")
# ]

# embeddings= HuggingFaceEmbeddings()

# vectorstore= Chroma.from_documents(docs,embeddings)

# retriever=vectorstore.as_retriever()

# llm=init_chat_model("mistral-small-2506")

# multiquery=MultiQueryRetriever(
#     retriever="retreiver",
#     llm=llm
# )

# query="what is gradient descent?"

# docs= multiquery.invoke(query)

# print("--------by_multiquery--------")
# for doc in docs:
#     print(doc.page_content)


from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv



load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()


llm = ChatMistralAI(model="mistral-small-latest")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)
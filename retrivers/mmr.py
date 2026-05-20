from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings= HuggingFaceEmbeddings()

vectorstore= Chroma.from_documents(docs,embeddings)
similarity_retriver= vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)
print("----------by_similarity_retrivers---------")
similarity_docs=similarity_retriver.invoke("what is gradient descent?")
for docs in similarity_docs:
    print(docs.page_content)

mmr_retrivers=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
print("-------by_mmr_retrivers-------")
mmr_docs=mmr_retrivers.invoke("what is gradient descent?")
for docs in mmr_docs:
   print(docs.page_content)
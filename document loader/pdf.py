from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
data= PyPDFLoader("document loader\GRU.pdf")
docs=data.load()
spliter=TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
chunks=spliter.split_documents(docs)
print(chunks[0].page_content)
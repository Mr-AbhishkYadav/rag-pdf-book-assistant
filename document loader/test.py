from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
data= TextLoader("document loader/notes.txt")
spliter= CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)
docs= data.load()
chunks=spliter.split_documents(docs)
# print(len(chunks)) 
for i in chunks:
    print(i.page_content)
    print()
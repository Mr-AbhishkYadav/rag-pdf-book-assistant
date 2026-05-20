from langchain_community.document_loaders import WebBaseLoader

url="https://www.apple.com/apple-watch-ultra-3/"
data=WebBaseLoader(url)
docs=data.load()
print(len(docs))
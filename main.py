# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# load_dotenv()

# # data= PyPDFLoader("document loader/Fundamentals Of Deep Learning PDF.pdf")
# # docs= data.load()

# # spliter=RecursiveCharacterTextSplitter(
# #     chunk_size=1000,
# #     chunk_overlap=200
# # )

# # chunks=spliter.split_documents(docs)

# template=ChatPromptTemplate.from_messages([
#     ("system","you are an AI that summarizes the text"),
#     ("human","{data}")
# ])

# # prompt=template.format_messages(data=docs[0].page_content)

# model = init_chat_model("mistral-small-2506")

# # result=model.invoke(prompt)
# # print(result.content)




from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model=MistralAIEmbeddings(model="mistral-embed")

vectorstore=Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistral-small-2506")

#prompt template
prompt=ChatPromptTemplate.from_messages(
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

print("RAG system created")

print("press 0 to exit")

while True:
    query=input("You : ")
    if query == "0":
        break

    docs=retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })

    response=llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")
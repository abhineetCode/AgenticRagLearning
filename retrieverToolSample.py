import os
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool

df = pd.read_csv("realistic_restaurant_reviews.csv")

#Step1: create a document out of teh csv file and split in the array (chunk) for indexing into our vectorstore
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)


#Step2: Use an in-memory vector store and ollama embeddings
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

#What is Vector:
#ML models cannot interpret information intelligibly in their raw format and require low-dimensional numerical data 
# as input. Therefore, it is necessary to convert the data into a numerical format.
#ML models use neural network embeddings to convert real-word information into numerical representations called vectors. 
# Vectors are numerical values that represent information in a multi-dimensional array. 
# They help ML models to find similarities among sparsely distributed items.
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embedding_model
)
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

#Retriever class returns Documents given a text unstructured query.
#It is more general than a vector store. 
#A retriever does not need to be able to store documents, only to return (or retrieve) it. 
#Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # Number of documents to return
)
#response = retriever.invoke("what is the best restaurant in town with gluten-free options?")
#print(response)
#print(len(response))

#Step3: Create a retriever tool using LangChain's prebuilt create_retriever_tool
#The create_retriever_tool function is used to create a Tool instance with the custom retriever, 
#a name, a description, a document prompt, a document separator, and an argument schema.
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_restaurant_review",
    "Search and return review for restaurants in the town.",
)
#response = retriever_tool.invoke({"query": "what is the best restaurant in town with gluten-free options?"})
#print(response)
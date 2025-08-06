from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from uuid import uuid4
import os

file_path = "agile-service-management-guide.pdf"
db_location = "./chrome_agileManagement_db"

#PyPDFLoader loads one Document object per PDF page
loader = PyPDFLoader(file_path)
docs = loader.load()
add_documents = not os.path.exists(db_location)
#print(len(docs))
#print(f"{docs[0].page_content[:200]}\n")
#print(docs[0].metadata)

#STEP2:  Splitting the document into chunks for indexing
#We use the RecursiveCharacterTextSplitter, which will recursively split the document 
#using common separators like new lines until each chunk is the appropriate size.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)
if add_documents:
    documents = text_splitter.split_documents(docs)

#print(len(all_splits))

#STEP3: Use an in-memory vector store and ollama embeddings
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

#What is Vector:
#ML models cannot interpret information intelligibly in their raw format and require low-dimensional numerical data 
# as input. Therefore, it is necessary to convert the data into a numerical format.
#ML models use neural network embeddings to convert real-word information into numerical representations called vectors. 
# Vectors are numerical values that represent information in a multi-dimensional array. 
# They help ML models to find similarities among sparsely distributed items.
vector_store = Chroma(
    collection_name="agile_management_guide",
    persist_directory=db_location,
    embedding_function=embedding_model
)

if add_documents:
    # Generate unique IDs for each document
    # Using uuid4() to ensure each document has a unique identifier
    # This is important for retrieval and management of documents in the vector store
    vector_store.add_documents(documents=documents, ids=[str(uuid4()) for _ in range(len(documents))])

#Retriever class returns most relvent information from our knowledge base (Documents) given a text unstructured query.
#It is more general than a vector store. 
#A retriever does not need to be able to store documents, only to return (or retrieve) it. 
#Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Number of documents to return
)
#response = retriever.invoke("How to bring agility to service management?")
#print(response)
#print(len(response))

#STEP4: Create a retriever tool using LangChain's prebuilt create_retriever_tool
#The create_retriever_tool function is used to create a Tool instance with the custom retriever, 
#a name, a description, a document prompt, a document separator, and an argument schema.
retriever_tool = create_retriever_tool(
    retriever,
    "agile_service_management_guide",
    "Search and return learning for agile.",
)
#response = retriever_tool.invoke({"query": "How to bring the agile mindset in individuals"})
#print(response)
#https://lilianweng.github.io/posts/2023-06-23-agent/

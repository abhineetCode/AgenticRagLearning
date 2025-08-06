import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import OpenAIEmbeddings

#https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#2-create-a-retriever-tool

os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"
os.environ["LANGCHAIN_PROJECT"] = "myFirstlanggraph"

llm = ChatGroq(groq_api_key = "", model_name = "Gemma2-9b-It")

#webbaseloader
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
#We'll start by fetching the content of the pages using WebBaseLoader utility:
docs=[WebBaseLoader(url).load() for url in urls]
#print(docs)
#print(docs[0][0].page_content.strip()[:1000])

#Split the fetched documents into smaller chunks for indexing into our vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    length_function=len,
)
doc_splits = text_splitter.split_documents(docs_list)

#print(doc_splits[0].page_content.strip())

#Use an in-memory vector store and OpenAI embeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

#Create a retriever tool using LangChain's prebuilt create_retriever_tool
from langchain_community.tools import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)
retriever_tool.invoke({"query": "types of reward hacking"})

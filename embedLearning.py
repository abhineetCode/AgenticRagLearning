from langchain_ollama import OllamaEmbeddings

#Embedding models are algorithms trained to encapsulate information into dense representations 4
# in a multi-dimensional space. Data scientists use embedding models to enable ML models to 
# comprehend and reason with high-dimensional data. These are common embedding models used in ML applications.
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

#The base Embeddings class in LangChain provides two methods: 
            #one for embedding documents and 
            #one for embedding a query. 
#The former takes as input multiple texts, while the latter takes a single text. 

#Embed a list of texts or documents to create vector representations.
embeddings = embedding_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings))
print(len(embeddings[0]))

# Embed single query: Embed a single piece of text for the purpose of comparing to other embedded pieces of texts.
embedded_query = embedding_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])

# What is embedding?
# Embedding enables machine learning models to find similar objects i.e. they allow models to learn complex patterns 
# and relationships in the data. For instance, in natural language processing (NLP), words with similar meanings 
# will have similar embeddings to easily understand the relationships between different words and 
# categories instead of just analysing each word in isolation thus, generate more coherent and contextually 
# relevant responses to user prompts and questions.

# Embeddings are not only used for text data, but can also be applied to a wide range of data types, 
# including images, graphs, and more. 
# # Depending on the type of data you’re working with, different types of embeddings can be used.

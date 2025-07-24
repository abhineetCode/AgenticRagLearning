activate by running below two in terminal

    python -m venv venv
    
    venv/Scripts/activate

install required packages from requirement.txt.

pip install -r .\requirement.txt

go to smith.langchain.com to create an account and create an api key.

go to console.groq.com to create an account and create an api key


# AgenticRagLearning

**What is Generative AI?**

Generative AI is a type of artificial intelligence that creates new content – such as text, images, or audio – based on patterns learned from existing data.

At the heart of it there is LLM like GPT-4, claude and huge volume  of data like Wikipedia, dictionary etc.

**What is AI Agent?**

Ai Agent is a program that takes input, thinks and acts to complete a task using tools, memory and knowledge.
It is autonomous but task specific and does not span multiple or evolving goal.

**What is Agentic AI?**

Agentic AI is a system where one or more AI agent work autonomously, often over long or complex tasks, making decision, using tools and other agent to reach a  goal.
Multi step reasoning
Multi step planning
Works on a complex goal autonomously

There are many Agentic AI framework
-	Agno
-	Crew AI
-	Langgraph
-	Microsoft Autogen

System Type	        GenAI (LLM only)	                    AI Agent	                                            Agentic AI
Task Capability	    Answer based on Pretrained knowledge.	Takes input, decides and completes a task	            Handles multi step goals with planning and coordination
Tool usage	        No external Tool	                    Uses tool to completes and task	                        Uses multiple tools, may call other agents
Autonomous decision	No decision making	                    Makes decision to complete the task	                    Plans, decides, and adapts overtime
			
**What is context widow?**

Context is a models's memory that has the converstion history or the external document or data it is processing.

Its a amount of text as a token, that model remember at any one time.LLM have limited context window or context length.

When you interact with LLM, the model takes teh conversation history and the prompts as a context window and passed to the LLM to predict the next relevent word and generate the coherent responses. for example, In a chatbot, the context allows the model remembers the previous question and answer, maintaining a consistent conversation flow. When summarizing a long document, the context helps the model grasp the overall meaning and relationship between different section.

Context Enggineering: Technique like Retrieval Augmented Generation (RAG) are used to provide LLMs with additional context from external sources, further improving the quality of responses.

Generally speaking, increasing an LLM's context window size translates to increased accuracy, fewer hallucinations, more coheent model responses, longer conversations and improved ability to analyze longer sequences of data. However, increased context length is not without tradeoffs: it often entails increased computational power requirements - and there fore increased costs and a potential increse in vulnerability to adversarial attacks.

**What is embedding?**

Embedding enables machine learning models to find similar objects i.e. they allow models to learn complex patterns and relationships in the data. For instance, in natural language processing (NLP), words with similar meanings will have similar embeddings to easily understand the relationships between different words and categories instead of just analysing each word in isolation thus, generate more coherent and contextually relevant responses to user prompts and questions.
Embeddings are not only used for text data, but can also be applied to a wide range of data types, including images, graphs, and more. Depending on the type of data you’re working with, different types of embeddings can be used.

What is Vector?

ML models cannot interpret information intelligibly in their raw format and require low-dimensional numerical data as input. Therefore, it is necessary to convert the data into a numerical format.
ML models use neural network embeddings to convert real-word information into numerical representations called vectors. Vectors are numerical values that represent information in a multi-dimensional array. They help ML models to find similarities among sparsely distributed items. 

“dad”=[0.1548,0.4848,…,1.864]

“mom”=[0.8785,0.8974,…,2.794] 

**Embeddings methods**
1.	Frequency based method
   
They are based on the idea that the importance of the significance or a word can be inferred from how frequently it occurs in the text. 
One such embedding is called TF-IDF: Term Frequency Inverse Document Frequency
TF-IDF highlights word that are frequent within a specific document, but are rare across the entire corpus. For example, in a document about coffee, TF-IDF would emphasize words like espresso or cappuccino, which might appear often in that document but rarely in others, about different topic.

2. Prediction based embeddings

They capture semantic relationship and contextual information between words. For example, in the sentences “The Dog is barking loudly” and the “The Dog is wagging its tail.” In prediction-based model would learn to associate Dog with words like bark, wag, and tail.
They excel at separating words with closed meaning and can manage the various senses in which a word may be used.

Now there are various models for generating words embeddings.

1.	Word2Vec: that was developed by Google in 2013. It has two main architectures
a.	CBOW (Continuous bags of words). CBOW predicts a target word based on its surrounding context words.
b.	Skip-Gram, does opposite – predicting context words given a target word.
2.	GLOVE: Global vectors for word representation, created by Standford university in 2014. This uses co-occurrences statistics to create word vectors.


3. Contextual based,

The representation of word changes based on surrounding context, so for example in transformer model, word “bank” would have different representation in the sentences.” I am going to the bank to deposit money” and “I am sitting at the bank of the river.” 	


**What are embedding models?**

Embedding models are algorithms trained to encapsulate information into dense representations in a multi-dimensional space. Data scientists use embedding models to enable ML models to comprehend and reason with high-dimensional data. These are common embedding models used in ML applications.

1. Principal component analysis
   
Principal component analysis (PCA) is a dimensionality-reduction technique that reduces complex data types into low-dimensional vectors. It finds data points with similarities and compresses them into embedding vectors that reflect the original data. While PCA allows models to process raw data more efficiently, information loss may occur during processing.

2. Singular value decomposition
   
Singular value decomposition (SVD) is an embedding model that transforms a matrix into its singular matrices. The resulting matrices retain the original information while allowing models to better comprehend the semantic relationships of the data they represent. Data scientists use SVD to enable various ML tasks, including image compression, text classification, and recommendation. 

3. Word2Vec
   
Word2Vec is an ML algorithm trained to associate words and represent them in the embedding space. Data scientists feed the Word2Vec model with massive textual datasets to enable natural language understanding. The model finds similarities in words by considering their context and semantic relationships.
There are two variants of Word2Vec—Continuous Bag of Words (CBOW) and Skip-gram. CBOW allows the model to predict a word from the given context, while Skip-gram derives the context from a given word. While Word2Vec is an effective word embedding technique, it cannot accurately distinguish contextual differences of the same word used to imply different meanings. 

4. BERT
   
BERT is a transformer-based language model trained with massive datasets to understand languages like humans do. Like Word2Vec, BERT can create word embeddings from input data it was trained with. Additionally, BERT can differentiate contextual meanings of words when applied to different phrases. For example, BERT creates different embeddings for ‘play’ as in “I went to a play” and “I like to play.” 

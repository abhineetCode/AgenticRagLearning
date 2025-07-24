# AgenticRagLearning

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


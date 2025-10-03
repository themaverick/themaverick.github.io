# NV-Embed

first off, let us start with the question 'What exactly is NV-Embed?' and since you people are here you probably know that it is a text **Embedding Model**. Which loosely means that it generates embeddings(_vector of numbers_) for the text fed to it. Let us define a more precise definition of embedders in a way that will be helpful for us.
>_Embedding Models are functions which output vectors of numbers(*embeddings*) when fed with an input. The objective of these models is to represent high-level(human understandable) data in a form which can be interprated and processed by machines and programs._
>
This high-level data ranges from images to videos to audio and much more. I'll write more about embedding models in a separate blog.\
Just like every other sub-field of NLP, Transformers have started to dominate here too. The first transformer based SOTA embedding model was [BERT](https://arxiv.org/abs/1810.04805) (_Bidirectional Encoder Representations from Transformers_) which has Encoder backbone and is trained using 2 objectives.
1. <ins>MLM (Masked Language Modeling)</ins>
2. <ins>NSP (Next Sentence prediction)</ins>

discussion on these objectives is beyond the scope of this blog.\
After the success of Decoder-based LLMs, people started exploring their utility in Encoding text. There were a few reasons why people believed that this would work:
1. LLMs are trained on **huge amount of textual data** and their size is many times than that of encoder based embedding models.
2. LLMs are **more efficient** in terms of using data during training, In BERT only 15% of tokens in a sequence are masked at a time whereas an LLM tries to predict all of the tokens iteratively in a sequence. <!--can i plot a graph which compares this efficiency. like at what length of input does casual training becomes more efficient than mlm.-->

However, despite these promising characteristics of LLMs, there several problems too.

1. They have causal mask (one-sided attention) and hence only the final token has the information from all of the tokens of the sequence.
2. They are optimized for predicting the next token and not for retaining information from the sequence. Although the prior has strong relation with the latter.\
    Try to understand it this way, when we are trying to predict the next token, we have to focus more on the fluency, grammatical aspect, punctuation marks, conjunctions etc. whereas for an embedding model, the semantic aspect is more important. eg. certain words which drastically change the meaning of sentence should have a high influence on the embeddings but it might not change the structure of the sentence and hence is not that useful for LLMs.

[**NV-Embed**](https://arxiv.org/pdf/2405.17428) model smartly takes care of these problems and successfully produces a SOTA Embedding model based on the [e5-mistral-7B-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) ,i.e, instruct mistral 7B model trained on synthetic data with the standard contrastive objective function.\
First, Nvidia folks get rid of the causal mask so that the representations with respect to each token has the access to the entire sequence. Now you may think that the quality of embeddigs should degrade as the llm is not _used to_ having bidirectional attention and might require some training to _warm-up_ the attention links but the reason of choosing Mistral was that it is known to have used bidirectional attention during some part of its training and hence the performance largely remains constant. for detailed analysis check out [LLM2Vec](https://arxiv.org/pdf/2404.05961). Now, to overcome the difference in their objectives(problem 2), they trained the model on text pairs with a contrastive objective function. Specifically, they trained the model with a _2 stage contrastive instrution tuning method_.<br><br>
There are several downstream tasks which are used to quantify the performance of an embedding model. Some of them are:<br>
1. **Semantic Textual Similarity (STS)**: 
Pairs of sentences are given a similarity score and it's correlation with human annotators is computed.

2. **Classification**: 
A logistic regression model is trained on the embeddings of the training set of a supervied sentence classification dataset which is then evaluated on the test set and the accuracy is reported.

3. **Clustering**: 
Here, K-means clustering algorithm is applied to the embeddings with number of clusters equal to the number of gold labels. [V-measure](https://aclanthology.org/D07-1043.pdf) for the dataset is reported.

4. **Retrieval**: 
The dataset consists of a corpus of documents and a set of queries where each query is mapped to relevant documents. Everything is embedded and the top similar documents to each query, on the basis of cosine similarity, are evaluated. nDCG@10 is computed and reported.

5. **Reranking**: 
Similar to retrieval but the main metric reported is Mean Average Precision(MAP) and the number of corpus documents is much smaller and each query has it.

## Training Setup
The most common training for embedding models is contrastive(InfoNCE) where for similar sentences, the embeddings from the model are forced towards greater similarity and for dissimilar sentences the embeddings are forced apart from each other.<br>



$$  
L = \frac{1}{B} \sum_{i=1}^{B}
- \log 
\frac{
    \exp\big( \text{sim}(f(x_i), f(x_i^+)) / \tau \big)
}{
    \sum_{j=1}^{B} \mathbf{1}_{[j \neq i]} \, \exp\big( \text{sim}(f(x_i), f(x_j)) / \tau \big)
}  
$$

where:

- $f(x_i)$ = embedding of sentence $x_i$
- $x_i^+$ = positive example corresponding to anchor $x_i$
- $sim(f_i, f_j) = f_i · f_j$ = cosine similarity
- $\tau > 0$ = temperature parameter
- Denominator sums over all other sentences in the batch (negatives)

This particular objective function is the in-batch variant of the InfoNCE loss where all of the other sentences in a batch are treated as negatives and hence make the whole training much more efficient. As the embeddings for all of the sentences can be calculated in a single forward pass once and more negatives give stronger gradients for the model to learn.

### The Issue

This configuration of using in-batch negatives is not suitable for all of the discussed downstream tasks. Take a guess for which of the tasks it might fail.

While training for classification, a batch will have multiple instances(pairs) belonging to one class and using in-batch negatives will mislead the model by pushing their embeddings apart. Infact, this same issue will be faced in clustering as well weheras, for retrieval task having in-batch negatives is helpful. As in retrieval tasks, the class of the sentences is not relevant, it depends whether the corpus is mapped to the query and this is the reason that the training process is divided into 2 parts.

1. **Retrieval Training**:
First, the model is trained on retrieval datasets with in-batch negatives, allowing the model to converge quickly.

2. **Non-retrieval Training**:
The model is then trained on non-retrieval datasets with no in-batch negatives.



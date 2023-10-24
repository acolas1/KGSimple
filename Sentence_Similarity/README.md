# Sentence Similarity
Package to calculate the similarity score between two sentences.
## Installation
>sentence-similarity prefers python 3.6 or higher.
```python
pip install sentence-similarity
```
## Examples
### Using Transformers
```python
from sentence_similarity import sentence_similarity
sentence_a = "paris is a beautiful city"
sentence_b = "paris is a grogeous city"
```
#### Supported Models
You can access some of the official model through the `sentence_similarity` class. However, you can directly type the HuggingFace's model name such as `bert-base-uncased` or `distilbert-base-uncased` when instantiating a `sentence_similarity`.

> See all the available models at [huggingface.co/models](https://huggingface.co/transformers/pretrained_models.html).
```python
model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')
```
BERT is bidirectional, the [CLS] is encoded including all representative information of all tokens through the multi-layer encoding procedure. The representation of [CLS] is individual in different sentences. 
Set embedding_type to `cls_token_embedding`, To compute the similarity score between two sentences based on [CLS] token. 
> paper link (https://arxiv.org/pdf/1810.04805.pdf)

```python
score=model.get_score(sentence_a,sentence_b,metric="cosine")
print(score)
```
Available metric are euclidean, manhattan, minkowski, cosine score.

### Using Sentence Transformers
```python
from sentence_similarity import sentence_similarity
sentence_a = "paris is a beautiful city"
sentence_b = "paris is a grogeous city"
```
#### Supported Models
You can access all the pretrained models of `Sentence-Transformers`

> See all the available models at [sbert/models](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models).
```python
model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='sentence_embedding')
```
Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity.
Set embedding_type to `sentence_embedding` (default embedding_type), To compute the similarity score between two sentences based on sbert. 
> paper link (https://arxiv.org/pdf/1908.10084.pdf)
```python
score=model.get_score(sentence_a,sentence_b,metric="cosine")
print(score)
```
Available metric are euclidean, manhattan, minkowski, cosine score.

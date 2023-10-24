'''
from sentence_similarity import sentence_similarity
sentence_a = "paris is a beautiful city"
sentence_b = "paris is a grogeous city"
model=sentence_similarity('distilbert-base-uncased','cls_token_embedding')
model.get_score(sentence_a,sentence_b,metric="cosine")
'''

from .sentence_similarity import sentence_similarity
__version__='1.0.0'




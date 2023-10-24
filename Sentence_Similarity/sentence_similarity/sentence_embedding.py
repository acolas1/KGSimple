from sentence_transformers import SentenceTransformer
from .similarity_functions import *

function_dispatcher = { 'cosine' : cosine, 'euclidean' : euclidean,'manhattan':manhattan,'minkowski':minkowski}

class sentence_embedding():
    '''load the pretrained model'''
    def __init__(self,model_name):
        self.model = SentenceTransformer(model_name)  

    '''compute the similarity score between two sentences'''
    def sentence_similarity(self,sentence1,sentence2,metric="cosine"):
        try:
            sentence1=self.model.encode(sentence1)
            sentence2=self.model.encode(sentence2)
            score=function_dispatcher[metric](sentence1.tolist(),sentence2.tolist())
            return score
        except Exception as e:
            print(e)
            return None














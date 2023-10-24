from transformers import AutoTokenizer, AutoModel
from .similarity_functions import *

function_dispatcher = { 'cosine' : cosine, 'euclidean' : euclidean,'manhattan':manhattan,'minkowski':minkowski}

class cls_token_embedding():
    '''load the pretrained model and tokenizer'''
    def __init__(self,model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    '''calculate the embeddings for the given sentence'''
    def calculate_embedding(self,model,tokenizer,sentence):
        token=tokenizer(sentence, return_tensors="pt")
        output=model(**token, output_attentions=True)
        return output
    
    '''compute the similarity score between two sentences'''
    def sentence_similarity(self,sentence1,sentence2,metric="cosine"):
        try:
            sentence1=self.calculate_embedding(self.model,self.tokenizer,sentence1)
            sentence2=self.calculate_embedding(self.model,self.tokenizer,sentence2)
            score=function_dispatcher[metric](sentence1[0][0][0].tolist(),sentence2[0][0][0].tolist())
            return score
        except Exception as e:
            print(e)
            return None














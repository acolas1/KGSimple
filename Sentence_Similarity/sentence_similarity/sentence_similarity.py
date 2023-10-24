from .sentence_embedding import *
from .cls_token_embedding import *
import logging
logger = logging.getLogger(__name__)

class sentence_similarity():
    '''load the pretrained model'''
    def __init__(self,model_name,embedding_type='sentence_embedding'):
        if  embedding_type.lower().strip() == "sentence_embedding":
            self.model=sentence_embedding(model_name)
        elif embedding_type.lower().strip() == "cls_token_embedding":
            self.model=cls_token_embedding(model_name)
        else:
            logger.warning('WARNING: embedding_type should be either sentence_embedding or cls_token_embedding. Using the default embedding_type (sentence_embedding) ...')
            self.model=sentence_embedding(model_name)
    
    '''compute the similarity score between two sentences'''
    def get_score(self,sentence1,sentence2,metric="cosine"):
        try:
           if metric not in ['cosine','euclidean','manhattan','minkowski']:
                logger.warning('WARNING: '+str(metric)+' metric not exist. Available metric are cosine, euclidean, manhattan, minkowski')
                return None
           result=self.model.sentence_similarity(sentence1,sentence2,metric)
           return result
        except Exception as e:
           print(e)
           return None
















import pytest
from get_data import input_value
import sys
sys.path.append("..")
from sentence_similarity import sentence_similarity

#load the model
cls_model=sentence_similarity('distilbert-base-uncased','cls_token_embedding')
sent_emb_model=sentence_similarity('distilbert-base-uncased','sentence_embedding')

ip1,ip2=input_value()
testdata1=[(ip1,ip1,0),(ip2,ip2,0),(ip1,ip2,0.8)]
testdata2=[(ip1,ip1,0),(ip2,ip2,0),(ip1,ip2,1.5)]

#check the minkowski distance
@pytest.mark.parametrize("sentence1, sentence2, output",testdata1)
def test_minkowski_distance_cls_token_embedding(sentence1,sentence2,output):
   score=cls_model.get_score(sentence1,sentence2,metric="minkowski")
   score=round(score,2)
   assert output-0.5 < score < output+0.5


@pytest.mark.parametrize("sentence1, sentence2, output",testdata2)
def test_minkowski_distance_sentence_embedding(sentence1,sentence2,output):
   score=sent_emb_model.get_score(sentence1,sentence2,metric="minkowski")
   score=round(score,2)
   assert output-0.5 < score < output+0.5
   
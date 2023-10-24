import pytest
from get_data import input_value
import sys
sys.path.append("..")
from sentence_similarity import sentence_similarity

#load the model
cls_model=sentence_similarity('distilbert-base-uncased','cls_token_embedding')
sent_emb_model=sentence_similarity('distilbert-base-uncased','sentence_embedding')

ip1,ip2=input_value()
testdata=[(ip1,ip1,0.99),(ip2,ip2,0.99),(ip1,ip2,0.95)]

#check the cosine score
@pytest.mark.parametrize("sentence1, sentence2, output",testdata)
def test_cosine_score_cls_token_embedding(sentence1,sentence2,output):
   score=cls_model.get_score(sentence1,sentence2,metric="cosine")
   score=round(score,2)
   assert output-0.25 < score < output+0.25


@pytest.mark.parametrize("sentence1, sentence2, output",testdata)
def test_cosine_score_sentence_embedding(sentence1,sentence2,output):
   score=sent_emb_model.get_score(sentence1,sentence2,metric="cosine")
   score=round(score,2)
   assert output-0.25 < score < output+0.25
   
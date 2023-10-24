import textstat
import sys
from scipy.special import expit
import evaluate

class SaliencyBERTScore:
    def __init__(self, lmscorer = "bertscore", lang="en"):
        self.bertscore = evaluate.load(lmscorer)
        self.lang = lang


    def calc_BERT_score(self, predictions, references, sigmoid):
        results = self.bertscore.compute(predictions=predictions, references=references, lang=self.lang)
        if sigmoid:
            results = expit(results)
        return results

    def score_batched(self, generated_text, source_text=None, sigmoid=False, printing=False, **kwargs):
        gen_score, source_score = None, None
        bert_score = self.calc_BERT_score(generated_text, source_text, sigmoid)
        f1 = bert_score['f1']
        
        if printing:
            print("scores: ", str(f1))
        return {"scores": f1}

    def score(self, generated_text, source_text=None, sigmoid=False, printing=False, **kwargs):
        gen_score, source_score = None, None
        bert_score = self.calc_BERT_score([generated_text], [source_text], sigmoid)
        f1 = bert_score['f1']
        
        if printing:
            print("scores: ", str(f1))
        return {"scores": f1}

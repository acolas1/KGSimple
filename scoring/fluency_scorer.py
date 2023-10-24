#based on SLOR scoring (https://aclanthology.org/K18-1031.pdf)
import math
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import torch
from transformers import GPT2Tokenizer
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from scipy.special import expit

class FluencyScorer:
    def __init__(self, batch_size=1, reduce="mean", log=True, laplace_smooth=False, prob_dict_path=None):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.reduce = reduce
        self.log = log
        self.laplace_smooth = laplace_smooth
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.scorer = LMScorer.from_pretrained("gpt2", device=self.device, batch_size=batch_size)
        self.idf_df = pd.read_csv(prob_dict_path, ',', encoding='utf-8')
        self.freq_dict = pd.Series((self.idf_df.frequency.values), index=self.idf_df.token).to_dict()
        self.num_tokens = self.idf_df.total.values[0]        
        
    def unigram_score(self, sentences):
        if self.freq_dict is None:
            raise Exception("Probability dictionary is not defined.") 
        unigram_scores = []
        for sent in sentences:
            unigram_prob = 1
            for token in word_tokenize(sent.lower()):
                if token in self.freq_dict:
                    if self.laplace_smooth:
                        curr_unigram_prob = (self.freq_dict[token]+1)/(self.num_tokens+len(self.freq_dict))
                    else:
                        curr_unigram_prob = self.freq_dict[token]/self.num_tokens
    
                

                else:
                    if self.laplace_smooth:
                        curr_unigram_prob = (1/(self.num_tokens+len(self.freq_dict)))
                    else:
                        curr_unigram_prob = 1
                # unigram_prob += curr_unigram_prob
                
                
                if self.log:
                    unigram_prob +=np.log(curr_unigram_prob)
                else:
                    unigram_prob *= curr_unigram_prob
            uni_score = unigram_prob/len(word_tokenize(sent))
            unigram_scores.append(uni_score)
        return unigram_scores
    
    def SLOR_score(self, sentence_list, lm_score, unigram_score):
        SLOR_scores = []
        for i in range(len(sentence_list)):
            SLOR_score = lm_score[i]-unigram_score[i]
            if self.log:
                SLOR_score = math.exp(lm_score[i]-unigram_score[i])
            SLOR_scores.append(SLOR_score)
        return SLOR_scores
        
    def score_batched(self, generated_texts, source_texts=None, printing=False, **kwargs):
        sources_SLOR_score, generateds_SLOR_score = None, None
        if source_texts:
            sources_lm_prob_scores = self.scorer.sentence_score(source_texts, reduce=self.reduce, log=self.log)
            sources_unigram_scores = self.unigram_score(source_texts)
            sources_SLOR_score = self.SLOR_score(source_texts, sources_lm_prob_scores, sources_unigram_scores)



        generateds_lm_prob_scores = self.scorer.sentence_score(generated_texts, reduce=self.reduce, log=self.log)
        generateds_unigram_scores = self.unigram_score(generated_texts)
        generateds_SLOR_score = self.SLOR_score(generated_texts, generateds_lm_prob_scores, generateds_unigram_scores)
       
        if printing:
            print("[source_sents]", source_texts)
            print("[source_lm]", sources_lm_prob_scores)
            print("[source_unigram]", sources_unigram_scores)
            print("[source_scores]", sources_SLOR_score)
            print("[generated_sents]", generated_texts)
            print("[generated_lm]", generateds_lm_prob_scores)
            print("[generated_unigram]", generateds_unigram_scores)
            print("[generated_scores]", generateds_SLOR_score)
        return {"scores": generateds_SLOR_score, "source_scores": sources_SLOR_score}

    def score(self, generated_text, source_text=None, printing=False, **kwargs):
        # sources_lm_prob_score = scorer.sentence_score(source_list, reduce="mean")
        
        sources_SLOR_score, generateds_SLOR_score = None, None
        if source_text:
            source_list = [source_text]
            sources_lm_prob_scores = self.scorer.sentence_score(source_list, reduce=self.reduce, log=self.log)
            sources_unigram_scores = self.unigram_score(source_list)
            sources_SLOR_score = self.SLOR_score(source_list, sources_lm_prob_scores, sources_unigram_scores)
       
        
        
        generateds_list = [generated_text]
        generateds_lm_prob_scores = self.scorer.sentence_score(generateds_list, reduce=self.reduce, log=self.log)
        generateds_unigram_scores = self.unigram_score(generateds_list)
        generateds_SLOR_score = self.SLOR_score(generateds_list, generateds_lm_prob_scores, generateds_unigram_scores)
       
        if printing:
            print("[source_sents]", source_text)
            print("[source_lm]", sources_lm_prob_scores)
            print("[source_unigram]", sources_unigram_scores)
            print("[source_scores]", sources_SLOR_score)
            print("[generated_sents]", generated_text)
            print("[generated_lm]", generateds_lm_prob_scores)
            print("[generated_unigram]", generateds_unigram_scores)
            print("[generated_scores]", generateds_SLOR_score)
        return {"scores": generateds_SLOR_score, "source_scores": sources_SLOR_score}
        
        
    
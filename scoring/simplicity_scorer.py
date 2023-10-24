import textstat
import sys
from scipy.special import expit

class SimplicityTextScore:
    def __init__(self):
        pass

    def calc_FRE(self, text, sigmoid):
        min_val = -30
        score = textstat.flesch_reading_ease(text)
        scaled_score = (score - min_val) / (121.22 - min_val)
        # Clamp scaled_score to the range [0, 1]
        scaled_score = max(0, min(scaled_score, 1))
        
        if sigmoid:
            scaled_score = expit(scaled_score)
            
        return scaled_score
        
    
    
    def calc_FKGL(self, text, sigmoid):
        score = max(0,textstat.flesch_kincaid_grade(text))
        if sigmoid:
            score = expit(score)
        return score

    def score_batched(self, generated_texts, source_texts=None, sigmoid=False, printing=False, **kwargs):
        gen_score, source_score = [],[]
        
        for text in generated_texts:
            gen_score.append(self.calc_FRE(text, sigmoid))
        
            
        if source_texts:
            for text in source_texts:
                source_score.append(self.calc_FRE(text, sigmoid))
         
        if printing:
            print("score: ", gen_score)
            print("source_score: ", source_score)
        return {"scores": gen_score, "source_scores": source_score}
    
    def score(self, generated_text, source_text=None, sigmoid=False, printing=False, **kwargs):
        gen_score, source_score = None, None
        
        gen_score = self.calc_FRE(generated_text, sigmoid)
            
        if source_text:
            source_score = self.calc_FRE(source_text, sigmoid)
         
        if printing:
            print("score: ", gen_score)
            print("source_score: ", source_score)
        return {"scores": gen_score, "source_scores": source_score}

    
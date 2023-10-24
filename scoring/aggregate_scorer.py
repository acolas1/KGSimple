import torch, time, numpy as np

class ScorerWrapper:
    def __init__(self, scorers, scoring_method="logsum", batch_size=1):
        assert scoring_method in ["product", "logsum"], "Unrecognized `scoring_method`"
        
        self.scorers = scorers
        self.scoring_method = scoring_method

        # if self.scoring_method == "logsum":
        #     self.score_func = logsum_score
        # elif self.scoring_method == "product":
        #     self.score_func = product_score
            
        if batch_size > 1:
            exec("self.score_func = {}".format(self.scoring_method+"_"+"score_batched"))
        else:
            exec("self.score_func = {}").format(self.scoring_method+"_"+"score")
        self.batch_size = batch_size
    def get_score_names(self):
        return [s["name"] for s in self.scorers]
        
    def score_batched(self, input_texts=None, generated_texts=None, old_kgs=None, new_kgs=None, dels_ents=None, partial=False, printing=False, timings=False, extras={}, progress=False):
        assert len(input_texts) == len(generated_texts) == len(old_kgs) == len(new_kgs) == len(dels_ents), "Data lengths don't match"
        
        data_list = []
        for inp, gen, old_kg, new_kg, del_ents in zip(input_texts, generated_texts, old_kgs, new_kgs, dels_ents):
            data_list.append({"inp": inp, "gen": gen, "old_kg": old_kg, "new_kg": new_kg, "del_ents": del_ents})

        if len(data_list) == 0:
            progress = False
       
        for batch in batcher(data_list, batch_size=self.batch_size, progress=progress):
            batch_inputs = [instance_dict["inp"] for instance_dict in batch]
            batch_gens = [instance_dict["gen"] for instance_dict in batch]
            batch_old_kgs = [instance_dict["old_kg"] for instance_dict in batch]
            batch_new_kgs = [instance_dict["new_kg"] for instance_dict in batch]
            batch_dels_ents = [instance_dict["del_ents"] for instance_dict in batch]
            batch_scores = self.score_func(self.scorers, batch_inputs, batch_gens, batch_old_kgs, batch_new_kgs, batch_dels_ents)
            for score_type, scores in batch_scores.items():
                if type(scores) in [torch.Tensor, np.array, np.ndarray]:
                    batch_scores[score_type] = scores.tolist()

        if printing:
            print("[total]", all_outputs["total_scores"])
        return batch_scores
    
    def score(self, input_text=None, generated_text=None, old_kg=None, new_kg=None, del_ents=None):
        aggregate_score = self.score_func(self.scorers, input_text, generated_text, old_kg, new_kg, del_ents)
        return aggregate_score
        

    def __call__(self, graphs, input_text, generated_text, **kwargs):
        return self.score(graphs, input_text, generated_text, **kwargs)

def batcher(iterator, batch_size=4, progress=False):
        if progress:
            iterator = tqdm.tqdm(iterator)

        batch = []
        for elem in iterator:
            batch.append(elem)
            if len(batch) == batch_size:
                final_batch = batch
                batch = []
                yield final_batch
        if len(batch) > 0: # Leftovers
            yield batch
            
#INPUT TEXT SHOULD BE GOLDEN_SEN IN OUR MODEL
def product_score_batched(scorers, input_texts, generated_texts, old_kgs=None, new_kgs=None, dels_ents=None):
    scorer_returns = {}
    total_scores = np.ones((len(generated_texts)))
    for scorer in scorers:
        if scorer['name'] == "brevity_pen":
            scores = scorer['model'].score_batched(old_kgs,new_kgs)
        elif scorer['name'] == "hallucination_new":
            scores =  scorer['model'].score_new_entity_batched(generated_texts, new_kgs)
        elif scorer['name'] == "hallucination_del":
            scores =  scorer['model'].score_del_entity_batched(generated_texts, dels_ents)
        else:
            scores = scorer['model'].score_batched(generated_texts,input_texts)

        if scorer['sign'] == 1:
            total_scores *= np.array(scores['scores'])
        else:
            total_scores *= (1-np.array(scores['scores']))
        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
    scorer_returns['total_scores'] = total_scores
    return scorer_returns

#INPUT TEXT SHOULD BE GOLDEN_SEN IN OUR MODEL
def product_score(scorers, input_text, generated_text, old_kg=None, new_kg=None, del_ents=None):
    scorer_returns = {}
    total_scores = 1
    #calc all scores
    print("[input_text]", input_text)
    print("[generated_text]", generated_text)
    print("[old_kg]", old_kg)
    print("[new_kg]", new_kg)
    print("[del_ents]", del_ents)
    for scorer in scorers:
        if scorer['name'] == "brevity_pen":
            scores = scorer['model'].score(old_kg,new_kg)
        elif scorer['name'] == "hallucination_new":
            scores =  scorer['model'].score_new_entity(generated_text, new_kg)
        elif scorer['name'] == "hallucination_del":
            scores =  scorer['model'].score_del_entity(generated_text, del_ents)
        else:
            scores = scorer['model'].score(generated_text,input_text)
        
        if scorer['sign'] == 1:
            total_scores *= np.array(scores['scores'])
        else:
            total_scores *= (1-np.array(scores['scores']))
        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
    scorer_returns['total_scores'] = total_scores
    return scorer_returns

# def logsum_score(scorers, graphs,input_text, generated_text):
#     scorer_returns == {}

#     for scorer in scorers:
#         scores = scorer['model'].score(graphs, input_text, generated_text, **extras)
#         weight = scorer.get("weight", 1.0)
#         scores["scores"] = np.clip(scores["scores"], 0.0001, 0.9999)
#         if scorer['sign'] == 1:
#             total_scores += weight*np.log(np.array(scores['scores']))
#         else: # It's a binary penalty
#             total_scores += np.log(1-np.array(scores["scores"]))

#         scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})

#     scorer_returns['total_scores'] = total_scores
#     return scorer_returns

    

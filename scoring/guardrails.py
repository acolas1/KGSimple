import os
import json
import numpy as np
import pandas as pd
import torch
import random
import re
from collections import defaultdict

from transformers import BartTokenizer, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import sys
sys.path.append('../')

from utils import *
from GAP.data_relations_as_nodes import GAPDataloader, EventDataset, WebNLGDataset
from GAP.data_relations_as_nodes import evaluate_bleu, get_t_emb_dim
from tqdm import tqdm, trange

from rake_nltk import Rake
import spacy
# import yake
from evaluate import load
from sentence_similarity import sentence_similarity
from GAP.modeling_gap_type import GAPBartForConditionalGeneration as GAP_Type_model
from GAP.modeling_gap import GAPBartForConditionalGeneration as GAP_model
    
class GraphReductionPenalty:
    def __init__(self, min_ratio=0.5):
        self.min_ratio = min_ratio
        
    
    def graph_overlap(self, old_graph, new_graph):
        '''
        compression score
        ## make sure the graph does not get to small (0.50, current round comparison)
        ## currently checking number of edges
        ''' 
        if len(old_graph) == 0:
            return 0.0
        return float(len(new_graph))/len(old_graph)
    
    def score_batched(self, source_graphs, generated_graphs, sigmoid=False, printing=False, **kwargs):
        assert len(source_graphs) == len(generated_graphs), "Numberof graphs are not equal."
        
        reduction_penalties = []
        for i in range(len(source_graphs)):
            graph_overla_ratio = self.graph_overlap(source_graphs[i], generated_graphs[i])
            reduction_penalty = 1.0 if graph_overla_ratio < self.min_ratio else 0.0
            reduction_penalties.append(reduction_penalty)
        
        if printing:
            print("reduction: ", reduction_penalties)
        return {"scores": reduction_penalties}

    def score(self, source_graph, generated_graph, sigmoid=False, printing=False, **kwargs):
        graph_overla_ratio = self.graph_overlap(source_graph, generated_graph)
        reduction_penalty = 1.0 if graph_overla_ratio < self.min_ratio else 0.0
        if printing:
            print("reduction: ", reduction_penalty)
        return {"scores": reduction_penalty}


'''
entity checker (in the new sentence)
## don't mention entities that were deleted
## don't mention too many extra entities
''' 
class NERInaccuracyPenalty:
    def __init__(self, spacy_model="en_core_web_sm"):
        common_ents = ["one", "united states", "army", "navy", "day"]

        self.common_ents = set([cent.lower() for cent in common_ents])
        self.word2num = {}
        self.black_list_types = set(["ORDINAL", "WORK_OF_ART", "EVENT", "PRODUCT", "LAW", "LANGUAGE"])
        self.number_words_to_remove = set(["the", "a", "an", "at", "of", "in", "than", "several", "few", "only", "about", "another", "least", "most", "last", "first", "early", "earlier",
                                           "later", "over", "fewer", "row", "every", "late", "ago", "only", "about", "around", "within", "more", "less"])

        self.string2digits = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                              "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
                              "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "a hundred": 100, "hundred": 100, "a thousand": 1000,
                              "thousand": 1000}

        self.string2digits = {k: str(v) for k, v in self.string2digits.items()}
        self.digits2string = {v: k for k, v in self.string2digits.items()}
        self.spacy_model = spacy.load("en_core_web_sm")

    def common_ents_no_problem(self, ent_text):
        return ent_text in self.common_ents

    def clean_entity_text(self, ent_text):
        ent_text = ent_text.lower().replace("-", " ").replace('"', '').strip().replace("'s", "")
        if ent_text[:4] == "the ":
            ent_text = ent_text[4:]
        return ent_text.strip()

    def singular(self, ent_text):
        if len(ent_text) == 0:
            return ent_text
        if ent_text[-1] == "s":
            return ent_text[:-1]
        else:
            return ent_text

    def quantifier_cleaning(self, quantifier_text):
        words = nltk.tokenize.word_tokenize(quantifier_text.lower())
        words = sorted([w for w in words if len(w) >= 2 and w not in self.number_words_to_remove])
        return set(words)

    def quantifier_matching(self, quantifier, entity_list):
        quantifier_clean = self.quantifier_cleaning(quantifier)
        entity_list_clean = [self.quantifier_cleaning(ent) for ent in entity_list]
        return any([quantifier_clean in ent2_clean for ent2_clean in entity_list_clean])

    def remove_graph_entities(self, sen_ents, graph_ents):

        ent_set = set([self.clean_entity_text(e) for e in graph_ents])
        finals = []
        
        for ent_new in sen_ents:
            raw_entity_lower = ent_new["text"].lower()
            entity_text = self.clean_entity_text(ent_new["text"])
            if self.common_ents_no_problem(entity_text): # The entity is too common and could added anywhere
                continue
            if entity_text in ent_set or self.singular(entity_text) in ent_set: # Exact match with some entity
                continue
            if entity_text in ent_set or self.singular(entity_text).lower() in ent_set or raw_entity_lower in ent_set:
                # Sometimes the NER model won't tag the exact same thing in the original paragraph, but we can just do string matching
                continue
            # Starting the entity-specific matching
            if ent_new["type"] in ["DATE", "CARDINAL", "MONEY", "PERCENT"]:
                # For dates:
                # a subset match is allowed: "several months" -> "months", "only a few weeks" -> "a few weeks"
                quantifier_clean = self.quantifier_cleaning(ent_new["text"])
                if self.quantifier_matching(ent_new["text"], graph_ents):
                    continue
                
                if ent_new["type"] == "CARDINAL":
                    if raw_entity_lower in self.string2digits and self.string2digits[raw_entity_lower] in ent_set:
                        continue # They wrote "nineteen" instead of 19
                    elif raw_entity_lower in self.digits2string and self.digits2string[raw_entity_lower] in ent_set:
                        continue # They wrote 19 instead of "nineteen"

            if ent_new["type"] in ["GPE", "NORP"]:
                #loose pattern matching for nationality
                pattern_n = r"^" + entity_text + r"[a-zA-Z]*n$"
                found = False
                for t in ent_set:
                    if re.search(pattern_n, t):
                        found = True
                        break
                    if t in entity_text:
                        found = True
                        break
                if found:
                    continue
                
                if entity_text+"n" in ent_set or entity_text[:-1] in ent_set:
                    # If you say india instead of indian, or indian instead of india.
                    # Definitely doesn't work with every country, could use a lookup table
                    continue
            if ent_new["type"] in ["ORG", "PERSON", "DATE", "CARDINAL"]:
                # Saying a smaller thing is fine: Barack Obama -> Obama. University of California, Berkeley -> University of California
                if any([entity_text in ent_text2 for ent_text2 in ent_set]):
                    continue
            finals.append(ent_new)
        return finals


    def extract_entities(self, text):
        doc = self.spacy_model(text)
        return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]


    def score_new_entity_batched(self, generated_texts, new_graphs, printing=False, **kwargs):
        '''
        entity checker (in the new sentence)
        ## don't mention too many extra entities
        ''' 
        new_scores = []
        sens_ents = []
        new_graphs_ents = []
        assert len(generated_texts) == len(new_graphs), "Number of texts and graphs are not equal."
        for i in range(len(generated_texts)):
            ## extract entities for the new graphs
            sen_ents = self.extract_entities(generated_texts[i])
            sens_ents.append(sen_ents)
            new_graph_ents = set()
            for edge in new_graphs[i]:
                new_graph_ents.add(edge[0])
                new_graph_ents.add(edge[2])
            new_graphs_ents.append(new_graph_ents)
            ## check if there are new entities from the sentence comparing with graph
            new_ents = self.remove_graph_entities(sen_ents, new_graph_ents)
            ## ratio of entities that are in the sentence but not in the graph
            if len(new_graph_ents) > 0:
                new_scores.append(len(new_ents) / len(new_graph_ents))
            else:
                new_scores.append(0)
            
                
        if printing:
            print("[scores]", new_scores)
            print("[sen_entities]", sens_ents)
            print("[graph_entities]", new_graphs_ents)
            
        return {"scores": new_scores, "sen_entities": sens_ents, "graph_entities": new_graphs_ents}
    
    def score_new_entity(self, generated_text, new_graph, printing=False, **kwargs):
        '''
        entity checker (in the new sentence)
        ## don't mention too many extra entities
        ''' 
        ## extract entities for the new graphs
        sen_ents = self.extract_entities(generated_text)
        new_graph_ents = set()
        for edge in new_graph:
            new_graph_ents.add(edge[0])
            new_graph_ents.add(edge[2])
        ## check if there are new entities from the sentence comparing with graph
        new_ents = self.remove_graph_entities(sen_ents, new_graph_ents)
        ## ratio of entities that are in the sentence but not in the graph
        if len(new_graph_ents) > 0:
            new_score = len(new_ents) / len(new_graph_ents)
        else:
            new_score = 0
            
        if printing:
            print("[scores]", new_score)
            print("[sen_entities]", sen_ents)
            print("[graph_entities]", new_graph_ents)
            
        return {"scores": new_score, "sen_entities": sen_ents, "graph_entities": new_graph_ents}

    
    def score_del_entity_batched(self, generated_texts, dels_ents, printing=False, **kwargs):
        '''
         entity checker (in the new sentence)
         ## don't mention entities that were deleted
         ''' 
        del_scores = []
        sens_ents = []
        assert len(generated_texts) == len(dels_ents), "Number of texts and length of del ents graph list are not equal."
        for i in range(len(generated_texts)):     
            ## extract entities from new sentence with NER module
            sen_ents = self.extract_entities(generated_texts[i])
            ## check if delete entities are included in the sentence
            del_set = set([self.clean_entity_text(e) for e in dels_ents[i]])
            sen_set = set([e['text'] for e in sen_ents])
            del_score = len(del_set.intersection(sen_set)) > 0
            sens_ents.append(sen_ents)
            del_scores.append(del_score)
        if printing:
            print("[scores]", del_scores)
            print("[sen_entities]", sens_ents)
            print("[del_entities]", dels_ents)
            
        return {"scores": del_scores, "sen_entities": sens_ents, "del_entities": dels_ents}
    
    def score_del_entity(self, generated_text, del_ents, printing=False, **kwargs):
        '''
         entity checker (in the new sentence)
         ## don't mention entities that were deleted
         ''' 
        ## extract entities from new sentence with NER module
        sen_ents = self.extract_entities(generated_text)
        ## check if delete entities are included in the sentence
        del_set = set([self.clean_entity_text(e) for e in del_ents])
        sen_set = set([e['text'] for e in sen_ents])
        del_score = len(del_set.intersection(sen_set)) > 0

        if printing:
            print("[scores]", del_score)
            print("[sen_entities]", sen_ents)
            print("[del_ents]", del_ents)
            
        return {"scores": del_score, "sen_entities": sen_ents, "del_entities": del_ents}

    
#     def score(self, generated_text, del_ents, new_graph, source_text=None, sigmoid=False, printing=False, **kwargs):
#         '''
#         entity checker (in the new sentence)
#         ## don't mention entities that were deleted
#         ## don't mention too many extra entities
#         ''' 
#         ## extract entities for the new graphs
#         new_graph_ents = set()
#         for edge in new_graph:
#             new_graph_ents.add(edge[0])
#             new_graph_ents.add(edge[2])
#         ## extract entities from new sentence with NER module
#         sen_ents = self.extract_entities(generated_text)

#         ## calculate delete and new entities scores
#         res = self.score_entity(new_graph_ents, del_ents, sen_ents)
#         return res


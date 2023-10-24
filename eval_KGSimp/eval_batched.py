#### read in result files, format, run eval functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import logging

import random
import numpy as np
import torch
import pandas as pd
from ast import literal_eval
import sys
 
# setting path
sys.path.append('/blue/daisyw/acolas1/KGSimplification/')

from eval_utils import *
from scoring.guardrails import *
from scoring.saliency_scorer import SaliencyBERTScore
from scoring.simplicity_scorer import SimplicityTextScore

def eval_batched(original_graphs, original_texts, generated_graphs, generated_texts, operations, batch_size, fluency_scorer, sim_scorer, const_parser, classifier):
    ## for entity extraction
    ner_check = NERInaccuracyPenalty()
    
    ## run eval functions in eval_utils.py for each graph in current batch
    print("batch size: " + str(batch_size))

    all_graphs_scores = []
    for i in range(0, len(original_graphs), batch_size):
        original_graphs_bs = original_graphs[i:i+batch_size]
        original_texts_bs = original_texts[i:i+batch_size]
        generated_graphs_bs = generated_graphs[i:i+batch_size]
        generated_texts_bs = generated_texts[i:i+batch_size]
        operations_bs = operations[i:i+batch_size]

        ## text evals
        text_lengths_bs = calc_text_length(generated_texts_bs)
        compression_ratios_bs = calc_compression_ratio(original_texts_bs, generated_texts_bs)
        syllables_counts_bs =  calc_syllables_count(generated_texts_bs)
        syllables_ratios_bs = calc_syllables_ratio(original_texts_bs, generated_texts_bs)
        fluency_scores_bs, saliency_scores_bs = calc_text_metrics(original_texts_bs, generated_texts_bs, fluency_scorer, sim_scorer)
        
        contistuency_heights_bs, contistuency_diameters_bs = calc_semantic_tree_metrics(generated_texts_bs,const_parser)
        
        ## graph evals:
        graph_lens_bs, graph_ratios_bs = calc_graph_stats(original_graphs_bs, generated_graphs_bs)
        ent_overlap_ratios_bs, rel_overlap_ratios_bs, graph_added_counts_bs, graph_deleted_counts_bs = comp_graph_eval(original_graphs_bs, generated_graphs_bs)

        ## graph-text evals:
        text_overlap_ratios_bs, text_added_counts_bs, text_deleted_counts_bs = entity_text_eval(original_texts_bs, generated_texts_bs, ner_check)
        accept_merge_ratios_bs, accept_replace_ratios_bs, accept_deletion_ratios_bs, rej_merge_ratios_bs, rej_replace_ratios_bs, rej_deletion_ratios_bs = calc_operation_stats(operations_bs)

        ## roberta-classification score (mhd add)
        roberta_scores_bs = roberta_score(classifier, generated_texts_bs)
        
 

        ## append batch results to global result
        for batch_idx, global_idx in enumerate(range(i, i+len(original_graphs_bs))):
            graph_dict = dict()
            graph_dict['text_lengths'], graph_dict['syllables_counts'], graph_dict['fluency_scores'], graph_dict['sim_scores'], \
            graph_dict['compression_ratios'], graph_dict['syllables_ratios'], graph_dict['contistuency_heights'], graph_dict['contistuency_diameters'], \
            graph_dict['graph_lens'], graph_dict['graph_ratios'], graph_dict['graph_ent_overlap_ratios'], graph_dict['graph_rel_overlap_ratios'], graph_dict['graph_added_counts'], graph_dict['graph_deleted_counts'], \
            graph_dict['accept_merge_ratios'], graph_dict['accept_replace_ratios'], graph_dict['accept_deletion_ratios'], \
            graph_dict['rej_merge_ratios'], graph_dict['rej_replace_ratios'], graph_dict['rej_deletion_ratios'], \
            graph_dict['text_overlap_ratios'], graph_dict['text_added_counts'], graph_dict['text_deleted_counts'], graph_dict['roberta_score'] = text_lengths_bs[batch_idx], syllables_counts_bs[batch_idx], fluency_scores_bs[batch_idx], saliency_scores_bs[batch_idx], \
            compression_ratios_bs[batch_idx], syllables_ratios_bs[batch_idx], contistuency_heights_bs[batch_idx], contistuency_diameters_bs[batch_idx], \
            graph_lens_bs[batch_idx], graph_ratios_bs[batch_idx], ent_overlap_ratios_bs[batch_idx], rel_overlap_ratios_bs[batch_idx], graph_added_counts_bs[batch_idx], graph_deleted_counts_bs[batch_idx], \
            accept_merge_ratios_bs[batch_idx], accept_replace_ratios_bs[batch_idx], accept_deletion_ratios_bs[batch_idx], rej_merge_ratios_bs[batch_idx], rej_replace_ratios_bs[batch_idx], rej_deletion_ratios_bs[batch_idx], \
            text_overlap_ratios_bs[batch_idx], text_added_counts_bs[batch_idx], text_deleted_counts_bs[batch_idx], roberta_scores_bs[batch_idx]
        
            all_graphs_scores.append(graph_dict)
        # break

    return all_graphs_scores



def eval_baselines_batched(original_texts, generated_texts, batch_size, fluency_scorer, sim_scorer, const_parser, classifier):
    ## for entity extraction
    ner_check = NERInaccuracyPenalty()
    
    ## run eval functions in eval_utils.py for each graph in current batch
    print("batch size: " + str(batch_size))

    
    all_graphs_scores = []
    print("FULL GEN LENGTH: ", len(generated_texts))
    for i in range(0, len(original_texts), batch_size):
        original_texts_bs = original_texts[i:i+batch_size]
        generated_texts_bs = generated_texts[i:i+batch_size]
        
        print("CURR GEN TEXT: " , len(generated_texts_bs))
        print("BATCH SIZE: ", batch_size)
        ## text evals
        text_lengths_bs = calc_text_length(generated_texts_bs)
        compression_ratios_bs = calc_compression_ratio(original_texts_bs, generated_texts_bs)
        syllables_counts_bs =  calc_syllables_count(generated_texts_bs)
        syllables_ratios_bs = calc_syllables_ratio(original_texts_bs, generated_texts_bs)
        fluency_scores_bs, saliency_scores_bs = calc_text_metrics(original_texts_bs, generated_texts_bs, fluency_scorer, sim_scorer)

        contistuency_heights_bs, contistuency_diameters_bs = calc_semantic_tree_metrics(generated_texts_bs,const_parser)
        ## graph-text evals:
        text_overlap_ratios_bs, text_added_counts_bs, text_deleted_counts_bs = entity_text_eval(original_texts_bs, generated_texts_bs, ner_check)

        ## roberta-classification score (mhd add)
        roberta_scores_bs = roberta_score(classifier, generated_texts_bs)

        ## append batch results to global result
        for batch_idx, global_idx in enumerate(range(i, i+len(original_texts_bs))):
            graph_dict = dict()
            graph_dict['text_lengths'], graph_dict['syllables_counts'], graph_dict['fluency_scores'], graph_dict['sim_scores'], \
            graph_dict['compression_ratios'], graph_dict['syllables_ratios'], graph_dict['contistuency_heights'], graph_dict['contistuency_diameters'], \
            graph_dict['text_overlap_ratios'], graph_dict['text_added_counts'], graph_dict['text_deleted_counts'], graph_dict['roberta_score'] = text_lengths_bs[batch_idx], syllables_counts_bs[batch_idx], fluency_scores_bs[batch_idx], saliency_scores_bs[batch_idx], \
            compression_ratios_bs[batch_idx], syllables_ratios_bs[batch_idx], contistuency_heights_bs[batch_idx], contistuency_diameters_bs[batch_idx], \
            text_overlap_ratios_bs[batch_idx], text_added_counts_bs[batch_idx], text_deleted_counts_bs[batch_idx], roberta_scores_bs[batch_idx]
        
            all_graphs_scores.append(graph_dict)
        # break
    return all_graphs_scores
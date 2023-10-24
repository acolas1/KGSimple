#### read in result files, format, run eval functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# setting path
sys.path.append('../../')

import logging

import random
import numpy as np
import torch
import pandas as pd
from ast import literal_eval
import stanza
import sacrebleu.tokenizers.tokenizer_13a as tok

from eval_utils import *
from eval_batched import *
from cli import args

sys.path.append('/blue/daisyw/acolas1/KGSimplification/')
from scoring.saliency_scorer import SaliencyBERTScore
from scoring.fluency_scorer import FluencyScorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def find_last_accept(x):
    # print(len(x))
    for i in range(len(x) - 1, -1, -1):
        cur_graph = set([tuple(lst) for lst in x[i][0]])
        prev_graph = set([tuple(lst) for lst in x[i][3]])
        if x[i][2] == 'START' or (x[i][2].split('_')[1] == 'ACCEPTED' and cur_graph != prev_graph):
            # if i <= 2:
            #     print(i)
            return x[i]

def find_best_accept(x):
    max_score = 0
    cur_stat_id = 0
    for i in range(len(x)):
        cur_graph = set([tuple(lst) for lst in x[i][0]])
        prev_graph = set([tuple(lst) for lst in x[i][3]])
        if x[i][2] == 'START' or (x[i][2].split('_')[1] == 'ACCEPTED' and cur_graph != prev_graph):
            if max_score <= x[i][-1]:
                max_score = x[i][-1]
                cur_stat_id = i
    # if cur_stat_id <= 2:
    #     print(cur_stat_id)
    return x[cur_stat_id]

def extract_operations(x):
    op_stats = []
    for i in range(1, len(x)): # ignore first start states
        ## fix the operation stats here (post-fix)
        ## if the original graph == initial graph
        ## the operation will be considered as fail (rejected)
        op_stat = x[i][2].split('_')
        cur_graph = set([tuple(lst) for lst in x[i][0]])
        prev_graph = set([tuple(lst) for lst in x[i][3]])
        if op_stat[-1] == 'ACCEPTED' and cur_graph == prev_graph:
            # op_stat[-1] = 'REJECTED'
            op_stat[-1] = 'NOT_FOUND'

        op_stats.append(op_stat)
    return op_stats


def eval():
    eval_mod = args.eval_mod
    eval_batch_size = args.eval_batch_size
    output_dir = args.output_dir
    dataset_dir = args.dataset_dir

    graphs_file = args.graphs_file
    test_file = os.path.join(output_dir, dataset_dir, eval_mod.split('-')[0], graphs_file)
 
    ## read in the output file
    ## format: no header, each row is the queue of processing on one graph
    ## each cell contains:
    ## if accepted: (newly_generated_graph, newly_generated_sentence, op + stat, previous_accepted_graph, previous_graph_score, current_graph_score)
    ## if rejected: (previous_accepted_graph, newly_generated_sentence, op + stat, newly_generated_graph, current_graph_score, previous_graph_score)
    ## ******** some graphs do not have accepted operations, keep that in mind !!!!
    stats_df = pd.read_csv(test_file, encoding='utf8', names=['op_' + str(i) for i in range(51)])
    
    fluency_scorer = FluencyScorer(1, log=False, laplace_smooth=True, prob_dict_path="../../data/wiki/enwiki/enwiki_terms_with_punc.csv")
    sim_scorer = SaliencyBERTScore()
    classifier = pipeline("sentiment-analysis", model = "textattack/roberta-base-CoLA")
    
    const_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    
    for col in stats_df.columns:
        stats_df[col] = stats_df[col].apply(literal_eval)

    # print(stats_df.op_0.head())
    
    ## get original graphs
    original_graphs = list(map(lambda x: x[0], stats_df.op_0))
    original_texts = list(map(lambda x: x[1], stats_df.op_0))
    # print(original_graphs[0])

    for index, text in enumerate(original_texts):
        original_texts[index] = tok.Tokenizer13a()(text.strip()).lower()
        
    generated_graphs = []
    generated_texts = []
    operations = []
    ## for greater than prev
    ## generated_graphs and generated_texts are the last accepted ones
    if eval_mod == 'greater_than_prev' or eval_mod == 'greater_than_zero-last':
        ## find the last accepted graph, otherwise, use the original graph
        last_accept_stats = stats_df.apply(lambda row: find_last_accept(row), axis=1)
        print(len(last_accept_stats))
        generated_graphs = list(map(lambda x: x[0], last_accept_stats))
        generated_texts = list(map(lambda x: x[1], last_accept_stats))
        
        for index, text in enumerate(generated_texts):
            generated_texts[index] = tok.Tokenizer13a()(text.strip()).lower()

    elif eval_mod == 'greater_than_zero-best':
        best_accept_stats = stats_df.apply(lambda row: find_best_accept(row), axis=1)
        print(len(best_accept_stats))
        generated_graphs = list(map(lambda x: x[0], best_accept_stats))
        generated_texts = list(map(lambda x: x[1], best_accept_stats))
        
        for index, text in enumerate(generated_texts):
            generated_texts[index] = tok.Tokenizer13a()(text.strip()).lower()
    
    operations = stats_df.apply(lambda row: extract_operations(row), axis=1)
    print(len(original_graphs), len(original_texts), len(generated_graphs), len(generated_texts), len(operations))

    
    ## batch compute scores for each graph
    all_graphs_scores = eval_batched(original_graphs, original_texts, generated_graphs, generated_texts, operations, eval_batch_size, fluency_scorer, sim_scorer, const_parser, classifier)
    print(len(all_graphs_scores))

    ## save to csv file
    all_graphs_scores_df = pd.DataFrame(all_graphs_scores)
    if eval_mod == 'greater_than_zero-last' or eval_mod == 'greater_than_zero-best':
        score_file = test_file.replace('.csv', '-' + eval_mod.split('-')[1] + '-eval_score.csv')
    else:
        score_file = test_file.replace('.csv', '-eval_score.csv')
    # print(score_file)
    all_graphs_scores_df.to_csv(score_file, encoding='utf8')


if __name__ == '__main__':
    # eval_mod = 'greater_than_prev'
    # eval_batch_size = 1
    # output_dir = '../out/output/'
    # dataset_dir = 'WebNLG_Simple'

    # graphs_file = 'webnlg_out_condition_greater_than_prev.csv'
    
    # eval(test_file=os.path.join(output_dir, dataset_dir, eval_mod, graphs_file), eval_mod=eval_mod.split('-')[0], eval_batch_size=eval_batch_size)
    eval()

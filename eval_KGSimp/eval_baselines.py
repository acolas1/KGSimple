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

def eval():
    eval_mod = args.eval_mod ## model + eval type
    eval_batch_size = args.eval_batch_size
    output_dir = args.output_dir
    dataset_dir = args.dataset_dir

    graphs_file = args.graphs_file
    # if dataset_dir == 'EditTS': ## 4 mods for EditTS
    if dataset_dir != 'Generated':
        original_file = os.path.join(output_dir, eval_mod + '-golden_generated.txt')
        generated_file = os.path.join(output_dir, dataset_dir, graphs_file)
    else:
        original_file = os.path.join(output_dir, eval_mod + '-golden_generated.txt')
        generated_file = os.path.join(output_dir, eval_mod + '-golden_generated.txt')

    fluency_scorer = FluencyScorer(1, log=False, laplace_smooth=True, prob_dict_path="../../data/wiki/enwiki/enwiki_terms_with_punc.csv")
    sim_scorer = SaliencyBERTScore()
    classifier = pipeline("sentiment-analysis", model = "textattack/roberta-base-CoLA")
    
    const_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    original_texts = []
    generated_texts = []

    ## read in original text file
    with open(original_file, 'r') as o_f:
        Lines = o_f.readlines()
        cnt = 0
        # Strips the newline character
        for line in Lines:
            # print("Line{}: {}".format(cnt, line.strip()))
            if cnt % 2:
                text = tok.Tokenizer13a()(line.strip()).lower()
                original_texts.append(text)
            cnt += 1
    ## read in the generated text file
    ## format: each line is the generated sentence
    ## need to read in the original sentence as well (EditTS has 3, complex, simple, generated; after preprocess it's the same as others)
    # Using readlines()
    if dataset_dir == 'EditTS':
        with open(generated_file, 'r') as gf:
            Lines = gf.readlines()
            cnt = 0
            # Strips the newline character
            for line in Lines:
                cnt += 1
                if cnt % 8 == 3:
                    # print("Line{}: {}".format(cnt, line.strip()))
                    text = tok.Tokenizer13a()(line.strip()).lower()
                    generated_texts.append(text)
    elif dataset_dir == 'Generated':
        generated_texts = original_texts
    else:
        with open(generated_file, 'r') as gf:
            Lines = gf.readlines()
            # cnt = 0
            # Strips the newline character
            for line in Lines:
                # cnt += 1
                # print("Line{}: {}".format(count, line.strip()))
                text = tok.Tokenizer13a()(line.strip()).lower()
                generated_texts.append(text)

    print(len(original_texts), len(generated_texts))
    # exit()
    
    ## batch compute scores for each graph
    all_graphs_scores = eval_baselines_batched(original_texts, generated_texts, eval_batch_size, fluency_scorer, sim_scorer, const_parser, classifier)
    print(len(all_graphs_scores))

    ## save to csv file
    all_graphs_scores_df = pd.DataFrame(all_graphs_scores)
    score_file = generated_file.replace('.txt', '-eval_score.csv')
    # print(score_file)
    all_graphs_scores_df.to_csv(score_file, encoding='utf8')


if __name__ == '__main__':
    eval()
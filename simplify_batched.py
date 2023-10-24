import os
import math
import json
import numpy as np
import pandas as pd
import torch
import random
from collections import defaultdict
import csv

from transformers import BartTokenizer, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import *

from scoring.fluency_scorer import FluencyScorer
from scoring.saliency_scorer import SaliencyBERTScore
from scoring.simplicity_scorer import SimplicityTextScore
from scoring.guardrails import *
from scoring.aggregate_scorer import ScorerWrapper

from GAP.data_relations_as_nodes import GAPDataloader, EventDataset, WebNLGDataset
from GAP.data_relations_as_nodes import evaluate_bleu, get_t_emb_dim
from tqdm import tqdm, trange

from rake_nltk import Rake
# import yake
from evaluate import load
from sentence_similarity import sentence_similarity
from GAP.modeling_gap_type import GAPBartForConditionalGeneration as GAP_Type_model
from GAP.modeling_gap import GAPBartForConditionalGeneration as GAP_model
#T5
from T5.modeling_t5 import T5ForConditionalGeneration as T5

bertscore = load("bertscore")
## sentence model for merge
phrase_model = sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')

## for sentence checking
ner_check = NERInaccuracyPenalty()

def run(args, logger):
    #load in model for graph-to-text and tokenizer
    checkpoint = args.model_path
    tokenizer_path = args.tokenizer_path
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
    if args.type_encoding:
        t_emb_dim = get_t_emb_dim(args)
        model = GAP_Type_model.from_pretrained(checkpoint,t_emb_dim=t_emb_dim)
    else:
        if 't5' in args.model_path:
            model = T5.from_pretrained(checkpoint)
        else:
            model = GAP_model.from_pretrained(checkpoint)
    
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))    
        
    # Here let's put all the scorers and make a "score" function for each. 
    scores = []
    if args.fluency_scorer:
        scores.append({"name": "fluency", "model": FluencyScorer(1, log=False, laplace_smooth=True, prob_dict_path=args.prob_dict_path), "sign": 1, "weight": 1.0})
    if args.simple_scorer:
        scores.append({"name": "simple_text_score", "model": SimplicityTextScore(), "sign": 1, "weight": 1.0})
    if args.saliency_scorer:
        scores.append({"name": "saliency_bert", "model": SaliencyBERTScore(), "sign": 1, "weight": 1.0})
    if args.brevity_scorer:
        scores.append({"name": "brevity_pen", "model": GraphReductionPenalty(min_ratio=0.6), "sign": -1, "weight": 1.0})
    if args.hallucination_new_scorer:
        scores.append({"name": "hallucination_new", "model": NERInaccuracyPenalty(), "sign": -1, "weight": 1.0})
    if args.hallucination_del_scorer:
        scores.append({"name": "hallucination_del", "model": NERInaccuracyPenalty(), "sign": -1, "weight": 1.0})
   
    bs = args.predict_batch_size
    scorer = ScorerWrapper(scores, scoring_method="product", batch_size=bs)
    ## read in graphs and golden sentences
    source_file = args.format_file + '.source'
    # target_file = args.format_file + '.target'
    graphs = read_formatdata(os.path.join(args.data_path, 'format', source_file))
    print("Num of graphs: ", len(graphs))
    print("Num of applied operations: ", args.num_operations)
    print("Temperature/SA: ", args.use_temperature, args.tinit, args.cr)
    
    ## golden sentences will be the translation of the original graph
    if os.path.exists(args.golden_file):
        print("exists")
        golden_sentences = []
        with open(args.golden_file) as f:
            while True:
                in_graph = f.readline().strip()
                out_text = f.readline().strip()
                if not out_text: break
                golden_sentences.append(out_text)
    else:
        print("does not exist")
        ## golden sentences will be the translation of the original graph
        golden_sentences = graph2sen(logger, args, graphs, model, tokenizer)
        outfile = open(args.golden_file,'a')
        for in_,out in zip(graphs,golden_sentences):
            outfile.write(str(in_)+"\n")
            outfile.write(out+"\n")
        print("Saved to {}".format(str(args.golden_file)))
    print("--------------------------------------------------------------")
    ## load enwiki TF-IDF for words
    idf_path = args.idf_path
    idf_df = pd.read_csv(idf_path, ',', encoding='utf-8')
    idf_dict = pd.Series(idf_df.idf.values, index=idf_df.token).to_dict()

    ## load complex-simple dictionary
    dict_path = args.dict_path
    complex_dict = load_complex_dict(dict_path, '\t')
    
    ## load rules
    rule_path = args.rule_path
    rule_list = []
    with open(rule_path) as f:
        for line in f:
            rule, conf = line[:-1].split('\t')
            head, tail = rule.split(' => ')
            ## extract the two relations in head
            head1, head2 = head.split('#SEP#')
            head1 = head1.split('|')
            head2 = head2.split('|')
            tail = tail.split('|')
            ## add to rule list
            rule_list.append((head1, head2, tail, conf))

    ## apply operations
    # sample_ids = random.sample(range(len(graphs)), 50)
    print('---------------Apply simplify operations on graphs------------------')
    sim_score = defaultdict(list)
    # for i in range(len(graphs_dataset.data)):
    
    graph_states = [[[graph, gold_sen, "START", [], 0, 0]] for graph, gold_sen in zip(graphs, golden_sentences)]
    
    print("batch size: " + str(bs))
    for i in range(0, len(graphs), bs):
        golden_sen = golden_sentences[i:i+bs]
        current_batch = graphs[i:i+bs]
        for op_step in range(args.num_operations):
            graph_states_batched = [item[-1] for item in graph_states[i:i+bs]]
            triples_batch, scores = [], []
            for graph_state in graph_states_batched:
                triples_batch.append(graph_state[0])
                scores.append(graph_state[-1])
   
            distribution = [args.delete, args.replace, args.merge]
            
            operations = operation_sample_batched(distribution, len(current_batch))

            new_triples_batch, del_ents_batch = [],[]
            for op_idx in range(len(current_batch)):
                ## select the operation function
                operation = operations[op_idx]
               
                ## Delete: delete the least-centralized node (edge)
                if operation == "delete":
                    new_triples, del_ents = graph_delete(triples_batch[op_idx], idf_dict, tokenizer)
                    new_triples_batch.append(new_triples)
                    del_ents_batch.append(del_ents)
                    print("New graph after delete: {}".format(new_triples))
                    
                ## Replace:  replace all possible complex node (edge)
                elif operation == "replace":
                    new_triples, del_ents = graph_replace(triples_batch[op_idx], idf_dict, complex_dict, tokenizer)
                    new_triples_batch.append(new_triples)
                    del_ents_batch.append(del_ents)
                    print("New graph after replace: {}".format(new_triples))
                ## merge one pair of possible edges
                elif operation == "merge":
                    new_triples, del_ents = graph_merge(triples_batch[op_idx], idf_dict, tokenizer)
                    new_triples_batch.append(new_triples)
                    del_ents_batch.append(del_ents)
                    print("New graph after merging: {}".format(new_triples))
            generated = graph2sen(logger, args, new_triples_batch, model, tokenizer)
            ## evaluation
            scorer_returns = scorer.score_batched(golden_sen, generated, triples_batch, new_triples_batch, del_ents_batch)
            comb_scores = scorer_returns['total_scores']
            # score
            # checking evaluation scores
            for batch_idx, curr_i in enumerate(range(i,i+len(current_batch))):
                if "greater_zero" in args.pass_condition:
                    if comb_scores[batch_idx] > 0:
                    ## (newly_generated_graph, newly_generated_sentence, op + stat, previous_accepted_graph, previous_graph_score, current_graph_score)
                        graph_states[curr_i].append([new_triples_batch[batch_idx], generated[batch_idx], operations[batch_idx]+"_ACCEPTED", triples_batch[batch_idx], graph_states[curr_i][-1][-1], comb_scores[batch_idx]])
                    else:
                    ## (previous_accepted_graph, newly_generated_sentence, op + stat, newly_generated_graph, current_graph_score, previous_graph_score)
                        graph_states[curr_i].append([triples_batch[batch_idx], generated[batch_idx], operations[batch_idx]+"_REJECTED", new_triples_batch[batch_idx], comb_scores[batch_idx], graph_states[curr_i][-1][-1]])   
                        
                elif "greater_prev" in args.pass_condition:
                    if comb_scores[batch_idx] > graph_states[curr_i][-1][-1]:
                        graph_states[curr_i].append([new_triples_batch[batch_idx], generated[batch_idx], operations[batch_idx]+"_ACCEPTED", triples_batch[batch_idx], graph_states[curr_i][-1][-1], comb_scores[batch_idx]])
                    elif args.use_temperature and comb_scores[batch_idx] > 0:
                        sample_prob = math.exp((comb_scores[batch_idx] - graph_states[curr_i][-1][-1])/(args.tinit - args.cr*op_step))
                        sample_prob = min(max(0, sample_prob), 1)
                        if sample_prob > random.random():
                            graph_states[curr_i].append([new_triples_batch[batch_idx], generated[batch_idx], operations[batch_idx]+"_ACCEPTED", triples_batch[batch_idx], graph_states[curr_i][-1][-1], comb_scores[batch_idx]])
                        else:
                            graph_states[curr_i].append([triples_batch[batch_idx], generated[batch_idx], operations[batch_idx]+"_REJECTED", new_triples_batch[batch_idx], comb_scores[batch_idx], graph_states[curr_i][-1][-1]])       
                    else:
                        graph_states[curr_i].append([triples_batch[batch_idx], generated[batch_idx], operations[batch_idx]+"_REJECTED", new_triples_batch[batch_idx], comb_scores[batch_idx], graph_states[curr_i][-1][-1]])         
                else:
                    raise Exception("Invalid pass condition given.")
               
            #     j -= 1
            print('************ COMPLETED ONE SAMPLED OPERATION FOR BATCH ************')
        print('************ BATCH FULLY COMPLETE ************')
    print('******************************            ******************************            FINISHED            ******************************            ******************************')
    with open(args.output_csv, "w") as f:
        wr = csv.writer(f)
        wr.writerows(graph_states)

if __name__ == '__main__':
    # data_path = '/home/UPSA/MK-Simple/data/GAP'
    # data_type = 'test'
    # format_file = data_type + '-large.format'
    run(data_path, format_file)

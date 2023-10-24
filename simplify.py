import os
import json
import numpy as np
import pandas as pd
import torch
import random
from collections import defaultdict

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

bertscore = load("bertscore")
## sentence model for merge
phrase_model = sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')

## for sentence checking
ner_check = NERInaccuracyPenalty()

def run(args, logger):
    #load in model for graph-to-text and tokenizer
    checkpoint = args.model_path
    tokenizer_path = args.tokenizer_path
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
    if args.type_encoding:
        t_emb_dim = get_t_emb_dim(args)
        model = GAP_Type_model.from_pretrained(checkpoint,t_emb_dim=t_emb_dim)
    else:
        model = GAP_model.from_pretrained(checkpoint)
    
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))    
        
    # Here let's put all the scorers and make a "score" function for each. 
    scores = [{"name": "fluency", "model": FluencyScorer(1, log=True, laplace_smooth=True, prob_dict_path="data/wiki/enwiki/enwiki_terms_with_punc.csv"), "sign": 1, "weight": 1.0},
           {"name": "simple_text_score", "model": SimplicityTextScore(), "sign": 1, "weight": 1.0},
           {"name": "saliency_bert", "model": SaliencyBERTScore(), "sign": 1, "weight": 1.0},
           {"name": "brevity_pen", "model": GraphReductionPenalty(min_ratio=0.6), "sign": -1, "weight": 1.0},
           {"name": "hallucination_new", "model": NERInaccuracyPenalty(), "sign": -1, "weight": 1.0},
           {"name": "hallucination_del", "model": NERInaccuracyPenalty(), "sign": -1, "weight": 1.0}
           ]
 
    
    scorer = ScorerWrapper(scores, scoring_method="product")
    ## read in graphs and golden sentences
    source_file = args.format_file + '.source'
    # target_file = args.format_file + '.target'
    graphs = read_formatdata(os.path.join(args.data_path, 'format', source_file))
    print("Num of graphs: ", len(graphs))
    ## golden sentences will be the translation of the original graph
    if os.path.exists(args.golden_file):
        print("exists")
        golden_sentences = []
        with open(args.golden_file) as f:
            while True:
                in_graph = f.readline().strip()
                out_text = f.readline().strip()
                golden_sentences.append(out_text)
                if not out_text: break
    else:
        print("does not exist")
        ## golden sentences will be the translation of the original graph
        golden_sentences = graph2sen(logger, args, graphs, model, tokenizer)
        outfile = open(args.golden_file,'a')
        for in_,out in zip(graphs,golden_sentences):
            print(in_)
            print(out)
            outfile.write(str(in_)+"\n")
            outfile.write(out+"\n")
        print("Saved to {}".format(str(args.golden_file)))
        
    ## load enwiki TF-IDF for words
    idf_path = 'data/wiki/enwiki/enwiki_terms.csv'
    idf_df = pd.read_csv(idf_path, ',', encoding='utf-8')
    idf_dict = pd.Series(idf_df.idf.values, index=idf_df.token).to_dict()

    ## load complex-simple dictionary
    dict_path = 'data/SimplePPDB/SimplePPDB'
    complex_dict = load_complex_dict(dict_path, '\t')

    ## load rules
    rule_path = 'data/rules/format-wikidata2019-hierarchy-map.txt'
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
    graph_states = [[[graph, "START", 0]] for graph in graphs]

    #TODO: here have to batch things
    for i in range(len(graphs)):
        golden_sen = golden_sentences[i]
        for j in range(args.num_operations):
            triples, _, score = graph_states[i][-1]
            distribution = [args.delete, args.replace, args.merge]
            
            #TODO: here have to batch things
            operation = operation_sample(distribution)
            # triples = graphs[i]
                

            #TODO: here have to batch things
            ## select the operation function
            print("original graph: {}".format(triples))
            
            #TODO: here have to batch things
            ## Delete: delete the least-centralized node (edge)
            if operation == "delete":
                new_triples, del_ents = graph_delete(triples, idf_dict, tokenizer)
                print("New graph after delete: {}".format(new_triples))

            #TODO: here have to batch things
            ## Replace:  replace all possible complex node (edge)
            if operation == "replace":
                new_triples, del_ents = graph_replace(triples, idf_dict, complex_dict, tokenizer)
                print("New graph after replace: {}".format(new_triples))

            #TODO: here have to batch things
            ## merge one pair of possible edges
            if operation == "merge":
                new_triples, del_ents = graph_merge(triples, idf_dict, tokenizer)
                print("New graph after merging: {}".format(new_triples))
            
            generated = graph2sen(logger, args, new_triples, model, tokenizer)[0]
            ## evaluation
            #TODO: here have to batch things
            scorer_returns = scorer.score(golden_sen, generated, triples, new_triples, del_ents)
            #TODO: here have to batch things
            comb_score = scorer_returns['total_scores'].item(0)
            
            

            #score
            ## checking evaluation scores
            ## add to queue if satisfy condition
            #TODO: here have to batch things
            if comb_score > 0:
                graph_states[i].append((new_triples, operation, comb_score))
            else:
                j -= 1
            # raise

            print('****')
        
        # break
    print('******************************')
    
    ## print average similarity ascores
    # print('BERT score of new sentences, precision: {}, recall: {}, f1: {}'.format(Average(sim_score['precision']), Average(sim_score['recall']), Average(sim_score['f1'])))


if __name__ == '__main__':
    # data_path = '/home/UPSA/MK-Simple/data/GAP'
    # data_type = 'test'
    # format_file = data_type + '-large.format'
    run(data_path, format_file)

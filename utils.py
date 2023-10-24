from __future__ import unicode_literals, print_function, division

import re
import torch
import os
import collections
import numpy as np
import random
import copy

from nltk.corpus import wordnet as wn
import smart_open
smart_open.open = smart_open.smart_open
# import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
import math
from rake_nltk import Rake as RAKE
import networkx as nx
import nltk

import evaluate

#TODO: change back when using webnlg
# from GAP.data_relations_as_nodes import GAPDataloader, EventDataset, WebNLGDataset
from T5.data import EventDataset
from T5.data import WebNLGDataLoader
from T5.data import GAPDataLoader

def operation_sample_batched(probs, num_samples):
    """
    Sample from `choices` with probability according to `probs`
    """
    choices = ["delete", "replace", "merge"]
    operation_list = []
    for i in range(num_samples):
        np_probs = np.array(probs)
        np_probs /= np_probs.sum()
        operation_list.append(np.random.choice(choices, p=np_probs))
    return operation_list

def operation_sample(probs):
    """
    Sample from `choices` with probability according to `probs`
    """
    choices = ["delete", "replace", "merge"]
    np_probs = np.array(probs)
    np_probs /= np_probs.sum()
    return np.random.choice(choices, p=np_probs)
        
def tokenize_text(text):
    text = text.replace('–', ' - ')
    text = text.replace('—', ' - ')
    text = text.replace(u'\u2212', ' - ')
    text = text.replace(u'\u2044', ' - ')
    text = text.replace(u'\xd7', ' x ')
    text = text.replace(u'>\u200b<', u'> <')
    text = text.replace(u'\xa0', u' ')
    # text = text.replace('><ent', '> <ent')
    # text = re.sub(">+(\S)+<ent", r'> \1 <', text)
    text = re.sub('([|.,!?():;&\+\"\'/-])', r' \1 ', text)
    text = text.replace('><e', '> <e')
    text = re.sub('\s{2,}', ' ', text)
    text = text.replace('<entity_','<ENT_')
    return text

def normalize_relations(relation):
    relation = re.sub("([a-z])([A-Z])","\g<1> \g<2>",relation)
    relation = relation.replace('_', ' ')
    return relation

def format_data(triples):
    if type(triples) is not list:
        triples = [triples]
        
    triples_text = ''

    #TODO: Change H,R,T back to S,P,O for WebNLG
    for triple in triples:
        new_triple = []
        subject = triple[0].lower()
        relation = normalize_relations(triple[1]).lower()
        object_ = triple[2].lower()
        new_triple.append('<H>')
        new_triple.append(subject)
        new_triple.append('<R>')
        new_triple.append(relation)
        new_triple.append('<T>')
        new_triple.append(object_)
        new_triple_text = ' '.join(new_triple)
        triples_text = triples_text + " " + new_triple_text
    triples_data_bart = triples_text.strip()
    return triples_data_bart

def preprocessData():
    return


######## 
#### sample nodes(words) for different operations
########


## load models
# print('loading glove')
# glove_model300 = api.load('glove-wiki-gigaword-300')
# print('loading word2vec')
# word2vec = api.load('word2vec-google-news-300')
# our_word2vec = Word2Vec.load(config['dataset'] + '/Word2vec/word2vec.model') #word2vec_src

## special token index
SOS_token = 1
EOS_token = 2
PAD_token = 0
UNK_token = 3

## average of a list
def Average(lst):
    return sum(lst) / len(lst)

## build TF-IDF dictionary
def getIDF(phrase2count, N):
	idf = {}
	for k,v in phrase2count.items():
		idf[k] = math.log2(N/v)
	return idf

## get word IDF value from dictionary
def get_idf_value(idf, word):
    if idf is None:
        return 1
    else:
        if word in idf:
            return idf[word]
        return 0

def phrase_sim(model, ph1, ph2, mode):
    return model.get_score(ph1, ph2, metric=mode)

def phrase_idf(phrase, idf_dict):
    phrase_list = phrase.split(' ')
    ph_idf = float(sum(idf_dict.get(w, 0) for w in phrase_list)) / float(len(phrase_list))
    return ph_idf

## match word by word for those order-changed phrases
def wordmatch(phrase, all_text):
    ## remove '(' and ')' in phrases
    phrase = phrase.replace('(', '').replace(')', '').replace(',', '')
    ## separate into words
    words = phrase.split(' ')
    phrase_len = len(words)
    ## match by word
    phrase_cnt = 0
    for sen in all_text:
        if sum(1 for p in words if p in sen) / phrase_len >= 0.5:
            phrase_cnt += 1
    return phrase_cnt

def load_complex_dict(dict_path, sep):
    complex_dict = dict()
    with open(dict_path, 'r') as f:
        for line in f:
            _, conf, _, complex_w, simple_w = line.strip().split(sep)
            if complex_w in complex_dict and complex_dict[complex_w][1] > conf:
                continue
            else:
                complex_dict[complex_w] = (simple_w, conf)
    return complex_dict

## build vocab dictionary (word count)
def getvocab(phrases, all_text, min_freq_out, phrase2count):
    for phrase in phrases:
        ## if there is comma, seperate and check first phrase
        check_phrases = [phrase]
        if ', ' in phrase:
            check_phrases = phrase.split(', ')
        elif ',' in phrase:
            check_phrases = phrase.split(',')
        phrase_len = len(check_phrases)
        if phrase == 'front-engine, front-wheel-drive layout':
            print(check_phrases)
        ## count each phrase, if all words are matched, count as one match sentence
        phrase_cnt = 0
        for sen in all_text:
            if sum(1 for p in check_phrases if p in sen) / phrase_len > 0.7:
                phrase_cnt += 1
        if phrase not in phrase2count and phrase_cnt:
            phrase2count[phrase] = phrase_cnt
        else:
            phrase_cnt = wordmatch(phrase, all_text)
            if phrase not in phrase2count and phrase_cnt:
                phrase2count[phrase] = phrase_cnt
            # if ',' in phrase:
            elif phrase not in phrase2count:
                phrase2count[phrase] = 1
                # print("phrase error: {}".format(phrase))

    # for k,v in phrase2count.items():
    # 	if v >= min_freq_out:
    # 		outputvocab.append(k)
    return phrase2count

## read in format data
def read_formatdata(data_file):
    ## read triples into list
    graphs = []
    with open(os.path.join(data_file)) as f:
            for line in f:
                line = line[:-1]
                ## read in triples for each graph
                triples = list(map(lambda x: tuple(x.split('|sep|')), line.split('|triple|')[1:]))
                graphs.append(triples)
                # print(triples)
    return graphs

## diameter of a graph
def graph_diameter(graph):
    G = nx.Graph(graph)
    return nx.diameter(G)

## degree centrality of nodes in a graph
## normalized by the maximum degree in this graph
def graph_degree(graph):
    G = nx.Graph(graph)
    return nx.degree_centrality(G)

## entity frequency in a graph (sorted)
def entity_freq(graph):
    ent_dict = collections.defaultdict(int)
    for s, r, o in graph:
        ent_dict[s] += 1
        ent_dict[o] += 1
    return ent_dict

## leverage entity freq to find the most central node(s) (if exist) in a graph
def central_node(graph):
    ent_dict = entity_freq(graph)
    max_degree = max(ent_dict.values())
    ## remove two edges with central node will decrease its degree by two
    ## we still want the central noBERT_scorede to be central after merge
    if max_degree >= 4:
        return [k for k in ent_dict if ent_dict[k] == max_degree], ent_dict
    else:
        return [], ent_dict

## BERT score
def BERT_score(predictions, references, lang):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang=lang)
    return results

## translate graph(s) to sentence
## graph, GAP model, tokenizer
@torch.no_grad()
def graph2sen(logger, args, graphs, model, tokenizer):
    ## format graph as input to GAP
    format_graph = [format_data(g) for g in graphs]
    input_data = format_graph
    # print(input_data[0])
    ## dataset with only new & original sentences
    dataset = EventDataset(logger, args, input_data, tokenizer, "val")
    # dataloader = WebNLGDataLoader(args, dataset, "val")
    dataloader = GAPDataLoader(args, dataset, "val")
    ## predict(translate) two graphs to sentences
    ###### why only prediction[0] (first batch result) is kept?????
    predictions = []
    ## clear memory
    model.eval()
    for i, batch in enumerate(dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        if 't5' in args.model_path:
            outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=args.num_beams,
                                 length_penalty=args.length_penalty,
                                 max_length=args.max_output_length,
                                 early_stopping=True,)
        else:
            outputs = model.generate(input_ids=batch[0],
                                        attention_mask=batch[1],
                                        input_node_ids=batch[2],
                                        node_length=batch[3],
                                        adj_matrix=batch[4],
                                        num_beams=args.num_beams,
                                        length_penalty=args.length_penalty,
                                        repetition_penalty=args.rep_penalty,
                                        max_length=args.max_output_length,
                                        early_stopping=True,)

        # Convert ids to tokens
        for output in outputs:
            pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            pred = re.sub(r'\b([a-zA-Z]) ?\. ?([a-zA-Z])\b', r'\1.\2', pred)
            pred = re.sub(r'\s+:', ':', pred)
            pred = re.sub(r'\s+;', ';', pred)
            pred = re.sub(r'\s*-\s*', '-', pred)
            pred = re.sub(r'\s+\)', ')', pred)
            pred = re.sub(r'\(\s+', '(', pred)

            predictions.append(pred.strip())
    # print(len(predictions))
    return predictions

## graph vs sentence score
def proxy_score(logger, args, graph, golden_sen, model, tokenizer, lang):
    ## convert graph to sentence with GAP model
    new_sen = graph2sen(logger, args, [graph], model, tokenizer)[0]
    print("New sentence (translate): {}".format(new_sen))
    return BERT_score([new_sen], [golden_sen], lang)


######## action functions
def sampling_graph(args, graph, subgraph):
    '''
    loop through graph components, and call sampling function
    graph: original graph, list of (e1, r, e2)
    subgraph: target subgraph, list of (e1, r, e2)
    return possible graphs
    '''
    res_graphs = []
    for e1, r, e2 in subgraph:
        ## sampling for each component
        res_graph.append(sampling_single(args, graph, e1))
        res_graph.append(sampling_single(args, graph, r))
        res_graph.append(sampling_single(args, graph, e2))
    return res_graphs
        
def sampling_single(args, graph, component):
    """
    sample replacement for give graph component (entity/relation)
    args: configration of the model
    graph: original graph
    component: target component, entity/relation/combined triple (word/sentence)
    """
    cand_graphs = []
    #### sample with text
    
    return cand_graphs

def add(args, graph, subgraph):
    return

def replace(graph, replace_w, simple_w):
    """
    replace an entity / relation to implify
    graph: original graph, list of (e1, r, e2)
    replace_w: the word of an edge/node that will be replaced, e/r
    simple_w: the simplified word for that edge/node
    return possible graphs
    """
    new_graph = []
    for i, edge in enumerate(graph):
        if replace_w in edge:
            new_graph.append(tuple(map(lambda x: x.replace(replace_w, simple_w), edge)))
        else:
            new_graph.append(edge)

    return new_graph

def delete(graph, target):
    """
    delete an entity
    graph: original graph, list of (e1, r, e2)
    target: the entity to delete
    return graph after delete
    """
    new_graph = [edge for edge in graph if target not in edge]
    
    return new_graph

def merge(graph, target):
    """
    merge two relations into one: (e1, h1, e2), (e2, h2, e3) --> (e1, t, e3)
    graph: original graph, list of (e1, r, e2)
    target: edge ids, edge pair idf, edge relations
    return graph after merge
    """
    #### merge into a sentence?
    # idx_list, tail, conf, r1, r2, h1, h2 = target
    idx_list, pair_idf, r1, r2, merge_mod = target
    ## get the subject and object of the new triple
    ## get the deleted entity
    if merge_mod == 0:
        subj, obj = graph[idx_list[0]][0], graph[idx_list[1]][2]
        del_ent = graph[idx_list[1]][0]
    elif merge_mod == 1:
        subj, obj = graph[idx_list[0]][2], graph[idx_list[1]][2]
        del_ent = graph[idx_list[1]][0]
    elif merge_mod == 2:
        subj, obj = graph[idx_list[0]][0], graph[idx_list[1]][0]
        del_ent = graph[idx_list[1]][2]
    elif merge_mod == 3: ## reduce one edge, keep same entities
        subj, obj = graph[idx_list[0]][0], graph[idx_list[0]][2]
        del_ent = ''
    ## remove orignal triples
    new_graph = [graph[i] for i in range(len(graph)) if i not in idx_list]
    ## add the merge triple
    # new_graph.append([subj, tail[0], obj])
    new_graph.append([subj, r1 + '|concat|' + r2, obj])
    return new_graph, del_ent

def transtive_edges(graph, central_ents):
    edge_pairs = []
    for i in range(len(graph)):
        edge1 = graph[i]
        for j in range(i + 1, len(graph)):
            edge2 = graph[j]
            if edge1[0] == edge2[2] and edge1[0] not in central_ents:
                edge_pairs.append([edge2[1], edge1[1], [j, i], 0])
            elif edge2[0] == edge1[2] and edge2[0] not in central_ents:
                edge_pairs.append([edge1[1], edge2[1], [i, j], 0])
    return edge_pairs

def head_or_tail_share_edges(graph, central_ents, freq_dict):
    edge_pairs = []
    for i in range(len(graph)):
        edge1 = graph[i]
        for j in range(i + 1, len(graph)):
            edge2 = graph[j]
            ## head = head or tail = tail, check central node
            ## no separate edge/graph after merge
            if edge2[0] == edge1[0]:
                if edge1[0] not in central_ents or freq_dict[edge2[2]] >= 2 or freq_dict[edge1[2]] >= 2:
                    edge_pairs.append([edge1[1], edge2[1], [i, j], 1])
            elif edge2[2] == edge1[2]:
                if edge1[1] not in central_ents or freq_dict[edge2[0]] >= 2 or freq_dict[edge1[0]] >= 2:
                    edge_pairs.append([edge1[1], edge2[1], [i, j], 2])
    return edge_pairs

def ent_share_edges(graph):
    edge_pairs = []
    for i in range(len(graph)):
        edge1 = graph[i]
        for j in range(i + 1, len(graph)):
            edge2 = graph[j]
            if set([edge1[0], edge1[2]]) == set([edge2[0], edge2[2]]):
                edge_pairs.append([edge1[1], edge2[1], [i, j], 3])
    return edge_pairs
            
## delete least-centralized node (edge)
## return new grpah and removed entity(s)
def graph_delete(triples, idf_dict, tokenizer):
    # find target node
    print('----Delete least-centralized & least frequent node (edge)----')
    ## convert to networkx graph for adj matrix
    ## keep only entities for degree and other computation
    node_graph = [(t[0], t[2]) for t in triples]
    # print("NODE GRAPH: " ,node_graph)
    degree_dict = graph_degree(node_graph)
    # print("degree_dict:", degree_dict)
    min_degree = min(degree_dict.values())
    candidate_words = [k for k in degree_dict if degree_dict[k] == min_degree]
    # print('----> target words: {}'.format(candidate_words))

    ## pick the one with the lowest frequency word, i.e. highest idf score
    target_words = []
    for phrase in candidate_words:
        ph_idf = phrase_idf(phrase, idf_dict)
        # print((phrase, phrase_list, ph_idf))
        target_words.append((phrase, ph_idf))
    target_words.sort(key=lambda x: x[1], reverse=True)
    deleted_word, deleted_id = target_words[0][0], target_words[0][1]
    print("delete word: {}".format(deleted_word))
    # print("original graph: {}".format(triples))

    new_triples = delete(triples, deleted_word)
    # print("New graph after delete: {}".format(new_triples))

    # ## simple eval: each triple as a sentence
    # references = [' '.join(t) for t in triples]
    # predictions = [' '.join(t) for t in new_triples]
    # results = bertscore.compute(predictions=predictions, references=references, lang="en")

    # sim_score['precision'].append(Average(results['precision']))
    # sim_score['recall'].append(Average(results['recall']))
    # sim_score['f1'].append(Average(results['f1']))

    # print('****')
    return new_triples, [deleted_word]

## replace all possible complex node (edge)
## return new grpah and removed entity(s)
def graph_replace(triples, idf_dict, complex_dict, tokenizer):
    ## find target nodes
    print('----Replace all possible nodes (edges)----')
    graph_phrases = set()
    for edge in triples:
        graph_phrases.add(edge[0])
        graph_phrases.add(edge[1])
        graph_phrases.add(edge[2])
    target_phrases = [p for p in graph_phrases if p in complex_dict]
    if target_phrases:
        ## replace the word with lowest freq, i.e. highest idf score
        target_word_scores = [(phrase, float(sum(idf_dict.get(w, 0) for w in phrase.split(' '))) / float(len(phrase.split(' ')))) for phrase in target_phrases]
        target_word_scores.sort(key=lambda x: x[1], reverse=True)
        print(target_word_scores)
        replace_word = target_word_scores[0][0]
        simple_w = complex_dict[replace_word][0]
        print('----> replace word: {}'.format((replace_word, simple_w)))
        # print("original graph: {}".format(triples))
        new_triples = replace(triples, replace_word, simple_w)
        # print("New graph after replace: {}".format(new_triples))

        # ## check similarity between new graph and original graph with BERT score
        # ## each triple as a sentence
        # references = [' '.join(t) for t in triples]
        # predictions = [' '.join(t) for t in new_triples]
        # results = bertscore.compute(predictions=predictions, references=references, lang="en")

        # sim_score['precision'].append(Average(results['precision']))
        # sim_score['recall'].append(Average(results['recall']))
        # sim_score['f1'].append(Average(results['f1']))
        return new_triples, [replace_word]
    else:
        print('----> no complex phrases to replace')
        return triples, ['']

## merge one pair of edges
## return new grpah and removed (by merge) entity(s)
def graph_merge(triples, idf_dict, tokenizer):
    print('----Merging two least important (idf-based) edges----')
    ## find central node
    ## currently, won't merge if central node will disappear after merge (central node is not the shared entity between two edges)
    central_ents, freq_dict = central_node(triples)
    # central_ents = []

    #### merge_mod indicate which two entities are the new head and tail for the merged triple
    edge_pairs = []
    ## first look for (a, r1, b) + (a, r2, b) --> (a, r3, b)
    edge_pairs = ent_share_edges(triples)
    ## then look for transitive edges: (a, r1, b) + (b, r2, c) --> (a, r3, c)
    if not edge_pairs:
        edge_pairs = transtive_edges(triples, central_ents)
    ## finally look for edges share head/tail entity, (a, r1, b) + (a, r2, c) --> (b, r3, c)
    ## if removing central ent, make sure no separate graph created
    if not edge_pairs:
        edge_pairs = head_or_tail_share_edges(triples, central_ents, freq_dict)
    # print(len(edge_pairs))

    cand_pairs = []
    for r1, r2, idx_list, merge_mod in edge_pairs:
        ## merge if a pair match with a rule
        # for h1, h2, t, conf in rule_list:
        #     match_score_1 = max(list(map(lambda x: phrase_model.get_score(e1[1], x, metric="cosine"), h1)))
        #     match_score_2 = max(list(map(lambda x: phrase_model.get_score(e1[2], x, metric="cosine"), h2)))
        #     # print(match_score_1, match_score_2)
        #     if match_score_1 > 0.7 and match_score_2 > 0.7:
        #         cand_pairs.append([idx_list, t, conf, e1, e2, h1, h2])
        ## merge two connected edges with lowest TF-IDF score
        r1_idf = phrase_idf(r1, idf_dict)
        r2_idf = phrase_idf(r2, idf_dict)
        pair_idf = (r1_idf + r2_idf) / 2
        cand_pairs.append([idx_list, pair_idf, r1, r2, merge_mod])

    ## apply merge to the pair with highest idf score
    # print("original graph: {}".format(triples))
    if cand_pairs:
        target_pair = sorted(cand_pairs, key=lambda x: x[1], reverse=True)[0]
        # print("merging edges: {}, {} -> {}, matching {}, {}".format(target_pair[3], target_pair[4], target_pair[1][0], target_pair[5], target_pair[6]))
        print("merging relations: {} |sep| {} ".format(target_pair[2], target_pair[3]))
        print("merging edges: {} |sep| {} ".format(triples[target_pair[0][0]], triples[target_pair[0][1]]))
        new_triples, del_ent = merge(triples, target_pair)
        # print("New graph after merging: {}".format(new_triples))
        return new_triples, [del_ent]
    else:
        print("Nothing to merge...")
        return triples, ['']

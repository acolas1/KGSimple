import textstat
from rake_nltk import Rake
import spacy
import sys
from eval_semantictree import *
from nltk import Tree
from transformers import pipeline

#return the length of the generated text (Decreasing)
#INPUT: generated_texts --> list of the output texts
#OUTPUT: lengths --> list of the text lengths
def calc_text_length(generated_texts):
    # return {'text_lengths': [len(t.split(' ')) for t in generated_texts]}
    return [len(t.split(' ')) for t in generated_texts]

#return the ratio of the generated text/source text (Decreasing)
#INPUT: generated_texts --> list of the output texts
#INPUT: source --> list of the input texts
#OUTPUT: ratios --> ratio of the text lengths
def calc_compression_ratio(source_texts, generated_texts):
    # return {'text_lengths': [len(t.split(' ')) for t in generated_texts]}
    ratios = []
    for t in zip(source_texts, generated_texts):
        # print(t[0])
        # print(t[1])
        # print(len(t[1].split(' ')))
        # print(len(t[0].split(' ')))
        # print(len(t[1].split(' '))/len(t[0].split(' ')))
        # raise
        ratios.append(len(t[1].split(' '))/len(t[0].split(' ')))
    return ratios

#return the number of syllables in the text
#INPUT: generated_texts --> list of the output texts
#OUTPUT: counts --> list of syllables per instance
def calc_syllables_count(generated_texts):
    # return {'syllables_count': [textstat.syllable_count(text) for t in generated_texts]}
    return[textstat.syllable_count(t) for t in generated_texts]

#return the ratio of the generated text/source syllabols (Decreasing)
#INPUT: generated_texts --> list of the output texts
#INPUT: source --> list of the input texts
#OUTPUT: ratios --> ratio of the syllables lengths
def calc_syllables_ratio(source_texts, generated_texts):
    # return {'text_lengths': [len(t.split(' ')) for t in generated_texts]}
    ratios = []
    for t in zip(source_texts, generated_texts):
        ratios.append(textstat.syllable_count(t[1])/textstat.syllable_count(t[0]))
    return ratios

#graph size and ratio (Decreasing)
#INPUT: original_graphs --> list of the original input triples
#INPUT: generated_graphs --> list of the generated triples
#OUTPUT: lengths --> list of graph lengths
#OUTPUT: ratios --> list of graph ratios (generated/original)
def calc_graph_stats(original_graphs, generated_graphs):
    gen_lens = [len(gen_graph) for gen_graph in generated_graphs]
    graph_ratios = [float(len(ori_graph)) / float(len(gen_graph)) for ori_graph, gen_graph in zip(original_graphs, generated_graphs)]
    # return {'graph_lens': gen_lens, 'graph_ratios': graph_ratios}
    return gen_lens, graph_ratios

#how many entities overlap between graph and new graph (Decreasing)
#how many entities are added (Decreasing)
#how many entities deleted (Increasing)
#INPUT: original_graphs --> list of the original input triples
#INPUT: generated_graphs --> list of the generated triples
#INPUT: generated_texts --> list of the output texts
#OUTPUT: overlap ratio--> list of ratio of overlapping entities (new graph / original graph)
#OUTPUT: added_counts --> list of counts of how many entities added
#OUTPUT: deleted_counts --> list of counts of how many deleted
def comp_graph_eval(original_graphs, generated_graphs):
    ent_overlap_ratios = []
    rel_overlap_ratios = []
    graph_added_counts = []
    graph_deleted_counts = []
    ## get delete entities for each graph
    for ori_graph, gen_graph in zip(original_graphs, generated_graphs):
        ori_graph_ents = set()
        ori_graph_rels = set()
        for edge in ori_graph:
            ori_graph_ents.add(edge[0])
            ori_graph_rels.add(edge[1])
            ori_graph_ents.add(edge[2])
        gen_graph_ents = set()
        gen_graph_rels = set()
        for edge in gen_graph:
            gen_graph_ents.add(edge[0])
            gen_graph_rels.add(edge[1])
            gen_graph_ents.add(edge[2])

        overlap_ents = gen_graph_ents.intersection(ori_graph_ents)
        overlap_rels = gen_graph_rels.intersection(ori_graph_rels)
        add_ents = gen_graph_ents - ori_graph_ents
        del_ents = ori_graph_ents - gen_graph_ents
        
        ent_overlap_ratios.append(float(len(gen_graph_ents)) / float(len(ori_graph_ents)))
        rel_overlap_ratios.append(float(len(gen_graph_rels)) / float(len(ori_graph_rels)))
        graph_added_counts.append(len(add_ents))
        graph_deleted_counts.append(len(del_ents))

    # return {'overlap_ratios': overlap_ratios, 'added_counts': added_counts, 'deleted_counts': deleted_counts}
    return ent_overlap_ratios, rel_overlap_ratios, graph_added_counts, graph_deleted_counts


#how many entities overlap between original sentence and new sentence (Decreasing)
#how many entities are added (Decreasing)
#how many entities deleted (Increasing)
#INPUT: original_graphs --> list of the original input triples
#INPUT: generated_graphs --> list of the generated triples
#INPUT: generated_texts --> list of the output texts
#OUTPUT: overlap ratio--> list of ratio of overlapping entities (new text/sentence)
#OUTPUT: added_counts --> list of counts of how many entities added
#OUTPUT: deleted_counts --> list of counts of how many deleted
def entity_text_eval(original_texts, generated_texts, ner_check):
    text_overlap_ratios = []
    text_added_counts = []
    text_deleted_counts = []
    ## get delete entities for each graph
    for ori_text, gen_text in zip(original_texts, generated_texts):
        ori_text_ents = ner_check.extract_entities(ori_text)
        ori_text_ents = set([ent_dict['text'] for ent_dict in ori_text_ents])
        gen_text_ents = ner_check.extract_entities(gen_text)
        gen_text_ents = set([ent_dict['text'] for ent_dict in gen_text_ents])

        overlap_ents = gen_text_ents.intersection(ori_text_ents)
        add_ents = gen_text_ents - ori_text_ents
        del_ents = ori_text_ents - gen_text_ents
        
        try:
            text_overlap_ratios.append(float(len(gen_text_ents)) / float(len(ori_text_ents)))
            text_added_counts.append(len(add_ents))
            text_deleted_counts.append(len(del_ents))
        except:
            text_overlap_ratios.append(-1)
            text_added_counts.append(-1)
            text_deleted_counts.append(-1)

    # return {'overlap_ratios': overlap_ratios, 'added_counts': added_counts, 'deleted_counts': deleted_counts}
    return  text_overlap_ratios, text_added_counts, text_deleted_counts


#INPUT: original_text
#INPUT: generated_text
#OUTPUT: fluency_score
#Output: saliency_score (Bert score)
def calc_text_metrics(original_texts, generated_texts, fluency_scorer, sim_scorer):
    text_fluency_scores = fluency_scorer.score_batched(generated_texts)['scores']
    text_sim_scores = sim_scorer.score_batched(generated_texts,original_texts)['scores']
    return text_fluency_scores, text_sim_scores

#INPUT: generated_texts
#OUTPUT: constituency parse tree heights
#Output: constituency parse tree diameters
def calc_semantic_tree_metrics(generated_texts, const_parser):
    heights, diameters = [], []
    for gen_text in generated_texts:
        doc = const_parser(gen_text)
        children = []
        for sent in doc.sentences:
            node = sent.constituency
            tree = Tree.fromstring(str(node))
            children.append(tree)

        if len(children) == 1:
            root = children[0]
        else:
            root = Tree("ROOT", [child[0] for child in children])
        
        broot = convertnary2bin(root)
        height, diam = diameter(broot)
        
        heights.append(height)
        diameters.append(diam)
    return heights, diameters

#MERGE Ratio (Increasing)
#REPLACE RATIO (Increasing)
#DELETIONS RATIO (Increasing)
#INPUT: operations --> list of the operations (need to filter only those that are ACCEPTED)
#INPUT: original_graphs --> list of the original input triples
#INPUT: generated_graphs --> list of the generated triples
#OUTPUT: merge_counts --> list of counts of how many times accepted merge
#OUTPUT: replace_ratio --> list of counts of how many times accepted replace
#OUTPUT: deletion_ratio --> list of counts of how many times accepted delete
def calc_operation_stats(operations):
    accept_merge_ratios = []
    accept_replace_ratios = []
    accept_deletion_ratios = []
    rej_merge_ratios = []
    rej_replace_ratios = []
    rej_deletion_ratios = []

    for operation in operations:
        accept_merge_cnt = 0
        accept_replace_cnt = 0
        accept_deletion_cnt = 0
        accept_cnt = 0

        rej_merge_cnt = 0
        rej_replace_cnt = 0
        rej_deletion_cnt = 0
        rej_cnt = 0
  
        for op, stat in operation:
            if stat == 'ACCEPTED':
                accept_cnt += 1
                if op == 'merge':
                    accept_merge_cnt += 1
                elif op == 'replace':
                    accept_replace_cnt += 1
                else:
                    accept_deletion_cnt += 1
            else:
                rej_cnt += 1
                if op == 'merge':
                    rej_merge_cnt += 1
                elif op == 'replace':
                    rej_replace_cnt += 1
                else:
                    rej_deletion_cnt += 1 
        # print(len(operation))
        # print(accept_cnt)
        # print(rej_cnt)
        # print(accept_merge_cnt)
        # print(accept_replace_cnt)
        # print(accept_deletion_cnt)
        # raise
        if accept_cnt:
            accept_merge_ratios.append(float(accept_merge_cnt) / float(accept_cnt))
            accept_replace_ratios.append(float(accept_replace_cnt) / float(accept_cnt))
            accept_deletion_ratios.append(float(accept_deletion_cnt) / float(accept_cnt))
        else:
            # print("NOT FOUND ACCEPTED OPERATION")
            # print(-1)
            accept_merge_ratios.append(-1)
            accept_replace_ratios.append(-1)
            accept_deletion_ratios.append(-1)

        if rej_cnt:
            rej_merge_ratios.append(float(rej_merge_cnt) / float(rej_cnt))
            rej_replace_ratios.append(float(rej_replace_cnt) / float(rej_cnt))
            rej_deletion_ratios.append(float(rej_deletion_cnt) / float(rej_cnt))
        else:
            # print("NOT FOUND REJECTED OPERATION")
            # print(-1)
            rej_merge_ratios.append(-1)
            rej_replace_ratios.append(-1)
            rej_deletion_ratios.append(-1)

    # return {"accept_merge_ratios": accept_merge_ratios, "accept_replace_ratios": accept_replace_ratios, "accept_deletion_ratios": accept_deletion_ratios, "rej_merge_ratios": rej_merge_ratios, "rej_replace_ratios": rej_replace_ratios, "rej_deletion_ratios": rej_deletion_ratios}
    return accept_merge_ratios, accept_replace_ratios, accept_deletion_ratios, rej_merge_ratios, rej_replace_ratios, rej_deletion_ratios


#INPUT: original_text
#OUTPUT: RoBerta-classification score
def roberta_score(classifier, original_texts):
    roberta_scores = classifier(original_texts)
    roberta_scores = list(map(lambda x: x['score'] if x['label'] == 'LABEL_1' else 1 - x['score'], roberta_scores))
    return roberta_scores
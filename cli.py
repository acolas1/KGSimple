import argparse

parser = argparse.ArgumentParser()

# Basic parameters
parser.add_argument("--train_file", default="train.json")
parser.add_argument("--predict_file", default="dev.json")
parser.add_argument("--test_file", default="test.json")
parser.add_argument("--output_dir", default=None, type=str, required=True)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_sample", action='store_true')
parser.add_argument("--do_predict", action='store_true')
parser.add_argument("--dataset", default="webnlg")
parser.add_argument("--golden_file", default="out/golden_generated.txt")
parser.add_argument("--output_csv", default="out/output/output.csv.txt")
parser.add_argument("--prob_dict_path", default="data/wiki/enwiki/enwiki_terms_with_punc.csv")
parser.add_argument("--idf_path", default="data/wiki/enwiki/enwiki_terms.csv")
parser.add_argument("--dict_path", default="data/SimplePPDB/SimplePPDB")
parser.add_argument("--rule_path", default="data/rules/format-wikidata2019-hierarchy-map.txt")



# Model parameters
parser.add_argument("--model_name", type=str, default="bart")
parser.add_argument("--model_path", type=str, default="./bart_model")
parser.add_argument("--tokenizer_path", type=str, default="./bart_model")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--do_lowercase", action='store_true', default=False)
parser.add_argument("--data_path", type=str, default="/home/UPSA/MK-Simple/data/GAP")
parser.add_argument("--format_file", type=str, default="test-large.format")    
parser.add_argument("--entity_entity", action='store_true', default=False)
parser.add_argument("--entity_relation", action='store_true', default=False)
parser.add_argument("--relation_entity", action='store_true', default=False)
parser.add_argument("--relation_relation", action='store_true', default=False)
parser.add_argument("--type_encoding", action='store_true', default=False)

# Preprocessing/decoding-related parameters
parser.add_argument('--max_input_length', type=int, default=32)
parser.add_argument('--max_output_length', type=int, default=20)
parser.add_argument('--num_beams', type=int, default=5)
parser.add_argument('--length_penalty', type=float, default=1.0)
parser.add_argument('--rep_penalty', type=float, default=1.0)
parser.add_argument("--append_another_bos", action='store_true', default=False)
parser.add_argument("--remove_bos", action='store_true', default=False)
parser.add_argument('--max_node_length', type=int, default=50)
parser.add_argument('--max_edge_length', type=int, default=60)
parser.add_argument("--clean_up_spaces", action='store_true', default=True)
parser.add_argument("--predict_batch_size", type=int, default=4)

# Other parameters

parser.add_argument('--prefix', type=str, default='',
                    help="Prefix for saving predictions")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--num_workers', type=int, default=16,
                    help="Number of workers for dataloaders")


#Sampling-related parameters
parser.add_argument("--min_length_of_edited_sent", default=6, type=int)
parser.add_argument("--lexical_simplification", action='store_true', default=False)
parser.add_argument("--delete", default=0.0, type=float)
parser.add_argument("--merge", default=0.0, type=float)
parser.add_argument("--replace", default=0.0, type=float)
parser.add_argument("--num_operations", default=3, type=int)
parser.add_argument("--pass_condition", default='greater_zero', type=str,
                   help="options include [greater_zero, greater_prev]")
parser.add_argument("--check_min_length", action='store_true', default=False)
parser.add_argument("--cos_similarity_threshold", default=0.7, type=float)
parser.add_argument("--cos_value_for_synonym_acceptance", default=0.45, type=float)
parser.add_argument("--min_idf_value_for_ls", default=9, type=int)
parser.add_argument("--sentence_probability_power", default=0.5, type=float)
parser.add_argument("--named_entity_score_power", default=1.0, type=float)
parser.add_argument("--len_power", default=0.25, type=float)
parser.add_argument("--fre_power", default=1.0, type=float)

parser.add_argument("--fluency_scorer", action='store_true', default=False)
parser.add_argument("--simple_scorer", action='store_true', default=False)
parser.add_argument("--saliency_scorer", action='store_true', default=False)
parser.add_argument("--brevity_scorer", action='store_true', default=False)
parser.add_argument("--hallucination_new_scorer", action='store_true', default=False)
parser.add_argument("--hallucination_del_scorer", action='store_true', default=False)

parser.add_argument("--use_temperature", action='store_true')
parser.add_argument('--tinit', type=float, default=3e-2, help='initial temperature')
parser.add_argument('--cr', type=float, default=6e-4, help='cooling rate for temperature')


## eval parameters
parser.add_argument("--eval_mod", default='greater_than_prev', type=str)
parser.add_argument("--eval_batch_size", default=1, type=int)
# parser.add_argument("--output_folder", default='../out/output/', type=str)
parser.add_argument("--dataset_dir", default='WebNLG_Simple', type=str)
parser.add_argument("--graphs_file", default='webnlg_out_condition_greater_than_prev.csv', type=str)


args = parser.parse_args()
python -u ../main.py \
        --do_train \
        --output_dir ../out/ModelB_new_learned_type_enc \
        --model_path ../model/GAP_type_e_r \
        --tokenizer_path ../tokenizer/pretrained_LM/bart-base \
        --data_path ../data/webnlg \
        --format_file test-large.format \
        --prob_dict_path ../data/wiki/enwiki/enwiki_terms_with_punc.csv \
        --idf_path ../data/wiki/enwiki/enwiki_terms.csv \
        --dict_path ../data/SimplePPDB/SimplePPDB \
        --rule_path ../data/rules/format-wikidata2019-hierarchy-map.txt \
        --dataset eventNarrative \
        --max_input_length 256 \
        --max_output_length 512 \
        --entity_entity \
        --entity_relation \
        --type_encoding \
        --max_node_length 60 \
        --length_penalty 5 \
        --rep_penalty 1 \
        --append_another_bos \
        --num_beams 5 \
        --delete 0.33 \
        --merge 0.33 \
        --replace 0.33 \
        --pass_condition greater_prev \
        --use_temperature \
        --fluency_scorer \
        --simple_scorer \
        --saliency_scorer \
        --brevity_scorer \
        --hallucination_new_scorer \
        --hallucination_del_scorer \
        --num_operations 50 \
        --predict_batch_size 64 \
        --golden_file ../data/webnlg/golden_generated.txt \
        --output_csv ../out/output/WebNLG/greater_than_prev/WebNLG_gap_greater_sa.csv

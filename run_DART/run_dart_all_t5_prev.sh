#change model_path path to T5
#change output dir
#change tokenizer_path
#change golden file name
#change output csv name

python -u ../main.py \
        --do_train \
        --model_name t5 \
        --output_dir ../out/T5 \
        --model_path ../model/T5/DART/t5_DART_upper \
        --tokenizer_path ../../pretrained_LM/t5-base \
        --prob_dict_path ../data/wiki/enwiki/enwiki_terms_with_punc.csv \
        --idf_path ../data/wiki/enwiki/enwiki_terms.csv \
        --dict_path ../data/SimplePPDB/SimplePPDB \
        --rule_path ../data/rules/format-wikidata2019-hierarchy-map.txt \
        --data_path ../data/DART \
        --format_file test-large-500-samples.format \
        --dataset eventNarrative \
        --max_input_length 256 \
        --max_output_length 512 \
        --entity_entity \
        --entity_relation \
        --max_node_length 50 \
        --length_penalty 5 \
        --rep_penalty 1 \
        --append_another_bos \
        --num_beams 5 \
        --delete 0.33 \
        --merge 0.33 \
        --replace 0.33 \
        --pass_condition greater_prev \
        --fluency_scorer \
        --simple_scorer \
        --saliency_scorer \
        --brevity_scorer \
        --hallucination_new_scorer \
        --hallucination_del_scorer \
        --num_operations 50 \
        --predict_batch_size 64 \
        --golden_file ../data/DART/golden_generated_t5.txt \
        --output_csv ../out/output/DART_t5_greater_prev.csv
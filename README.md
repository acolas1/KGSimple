# KGSimple
Official repository for [Can Knowledge Graphs Simplify Text?](https://dl.acm.org/doi/10.1145/3583780.3615514) accepted to [CIKM 2023].

Anthony Colas, Haodi Ma, Xuanli He, Yang Bai, and Daisy Zhe Wang. 2023. Can Knowledge Graphs Simplify Text? In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23). Association for Computing Machinery, New York, NY, USA, 379–389. https://doi.org/10.1145/3583780.3615514

  
## Abstract
Knowledge Graph (KG)-to-Text Generation has seen recent improvements in generating fluent and informative sentences which describe a given KG. As KGs are widespread across multiple domains and contain important entity-relation information, and as text simplification aims to reduce the complexity of a text while preserving the meaning of the original text, we propose \textbf{KGSimple}, a novel approach to unsupervised text simplification which infuses KG-established techniques in order to construct a simplified KG path and generate a concise text which preserves the original input's meaning. Through an iterative and sampling KG-first approach, our model is capable of simplifying text when starting from a KG by learning to keep important information while harnessing KG-to-text generation to output fluent and descriptive sentences. We evaluate various settings of the \textbf{KGSimple} model on currently-available KG-to-text datasets, demonstrating its effectiveness compared to unsupervised text simplification models which start with a given complex text.

![Model Overview](paper_resources/Figures/overview.png#pic_center)

## Citation
If you find our work helpful, please consider citing it in your research or project. You can use the following BibTeX entry:
```bibtex
@misc{colas2023knowledge,
      title={Can Knowledge Graphs Simplify Text?}, 
      author={Anthony Colas and Haodi Ma and Xuanli He and Yang Bai and Daisy Zhe Wang},
      year={2023},
      eprint={2308.06975},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Environment Setup
Ensure you have the following dependencies loaded in your environment:

```bash
module load cuda/11.1.0
module load pytorch/1.9.0
```
### Requirements
Download the `requirements.txt` file. Note that this file was auto-generated from our Conda environment.

### NLTK Setup
To use NLTK's punkt, run the following command:
```python -c "import nltk; nltk.download('punkt')"```

## Data and Models

### Data Files

1. **Download Data Files:**
   - [Click here to download the required data files](https://drive.google.com/file/d/1Xh7kXfVfTm9xSj_bfsjNjeYk_8DBLbYG/view?usp=sharing).
   - Unzip the downloaded file.
   - Place the extracted data files in the root directory of your project.

### Pre-trained Models

2. **Download Pre-trained Model Files:**
   - [Access the pre-trained model files here](https://drive.google.com/file/d/1mWmBmjWyxAiLddao-afLTrHYrc1xx6Am/view?usp=sharing).
   - Unzip the downloaded file.
   - Place the extracted pre-trained model files in the root directory of your project.

### Tokenizers (BART-Base and T5)

3. **Download Tokenizers:**
   - For the BART-Base tokenizer, visit [this link](https://huggingface.co/facebook/bart-base).
   - For the T5 tokenizer, visit [this link](https://huggingface.co/t5-base).
   - Download the tokenizer files from the respective links.
   - Create a folder named `tokenizer` in the root directory of your project.
   - Place the downloaded tokenizer files in the `tokenizer` folder.

## Bash Scripts
Several bash scripts are provided corresponding to different configurations:

### DART Dataset 
```bash
cd run_DART
```

Simulated Annealing:
```bash
bash run_dart_all_sa.sh
```

Greater Than Previous Condition:
```bash
bash run_dart_all_t5_prev.sh
```

Greater Than Zero Condition:
```bash
bash run_dart_all_zero.sh
```

### WebNLG Dataset 
```bash
cd run_WebNLG
```

Simulated Annealing:
```bash
bash run_webnlg_all_sa.sh
```

Greater Than Previous Condition:
```bash
bash run_webnlg_all_prev.sh
```

Greater Than Zero Condition:
```bash
bash run_webnlg_all_zero.sh
```

## Next Evaluation Steps

To run the next evaluation steps without folder issues, use the default parameters defined in the bash scripts for the `output_csv` parameter.

### File Storage

The output files should be saved to out/output/

### Explanation of Flags:

- `--eval_mod`: The mode to obtain the graphs. Choices are `greater_than_prev`, `greater_than_zero-last`, `greater_than_zero-best`. For simulated annealing, use `greater_than_prev` on the simulated annealing graph results from the previous scripts.

- `--eval_batch_size`: This flag controls the batch size used during evaluation. The batch size is a hyperparameter that determines the number of samples processed in one iteration. It can affect memory usage and computation time.

- `--output_dir`: The directory containing the output file from the KGSimple previous run scripts.

- `--dataset_dir`: The directory containing the specific output dataset files, e.g., `DART` (inside `output`) for the DART dataset.

- `--graphs_file`: The CSV file containing the simplified results from the previous run scripts.

### Need Help or Encountering Issues?

If you have any questions, encounter issues, or need assistance, please feel free to let us know. We appreciate your interest and feedback and are here to help!

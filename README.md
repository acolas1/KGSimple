# KGSimple
Official repository for [Can Knowledge Graphs Simplify Text?](https://dl.acm.org/doi/10.1145/3583780.3615514) accepted to [CIKM 2023].

Anthony Colas, Haodi Ma, Xuanli He, Yang Bai, and Daisy Zhe Wang. 2023. Can Knowledge Graphs Simplify Text? In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23). Association for Computing Machinery, New York, NY, USA, 379–389. https://doi.org/10.1145/3583780.3615514

  
## Abstract
Knowledge Graph (KG)-to-Text Generation has seen recent improvements in generating fluent and informative sentences which describe a given KG. As KGs are widespread across multiple domains and contain important entity-relation information, and as text simplification aims to reduce the complexity of a text while preserving the meaning of the original text, we propose \textbf{KGSimple}, a novel approach to unsupervised text simplification which infuses KG-established techniques in order to construct a simplified KG path and generate a concise text which preserves the original input's meaning. Through an iterative and sampling KG-first approach, our model is capable of simplifying text when starting from a KG by learning to keep important information while harnessing KG-to-text generation to output fluent and descriptive sentences. We evaluate various settings of the \textbf{KGSimple} model on currently-available KG-to-text datasets, demonstrating its effectiveness compared to unsupervised text simplification models which start with a given complex text.

![Model Overview](paper_resources/Figures/overview.png#pic_center)

## Citation
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

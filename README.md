# LLM4IR-Survey
This is the collection of papers related to large language models for information retrieval. These papers are organized according to our survey paper [Large Language Models for Information Retrieval: A Survey](https://arxiv.org/abs/2308.07107). 

Feel free to contact us if you find a mistake or have any advice. Email: yutaozhu94@gmail.com and dou@ruc.edu.cn.

## 🌟 Citation
Please kindly cite our paper if helps your research:
```BibTex
@article{LLM4IRSurvey,
    author={Yutao Zhu and
            Huaying Yuan and
            Shuting Wang and
            Jiongnan Liu and
            Wenhan Liu and
            Chenlong Deng and
            Haonan Chen and
            Zhicheng Dou and
            Ji-Rong Wen},
    title={Large Language Models for Information Retrieval: A Survey},
    journal={CoRR},
    volume={abs/2308.07107},
    year={2023},
    url={https://arxiv.org/abs/2308.07107},
    eprinttype={arXiv},
    eprint={2308.07107}
}
```
## 🚀 Update Log
- Version 4 \[2025-09-17\]
  - Search Agent: We reformulate the search agent section.
  - Reranker: We add several listwise rerankers and Section 'Reasoning-intensive Rerankers'.
- Version 3 \[2024-09-03\]
  - We refine the background to pay more attention to IR.
  - Rewriter: We add a new section "Formats of Rewritten Queries" to provide a more clear classfication and incorporated up-to-date methods.
  - Retriever: We incorporated up-to-date methods that utilize LLM to enlarge the dataset used for training retrievers or to improve the overall structure and design of retriever systems.
  - Reranker: We have added some unsupervised rerankers, several studies focusing on training data augmentation, and discussions on the limitations of LLM rerankers.
  - Reader: We added the latest studies on readers, particularly enriching the works in the active reader section.
  - Search Agent: We added the latest studies on static and dynamic search agents, particularly enriching the works in benchmarking and self-planning.

- Version 2 \[2024-01-19\]
  - We added a new section to introduce search agents, which represent an innovative approach to integrating LLMs with IR systems.
  - Rewriter:  We added recent works on LLM-based query rewriting, most of which focus on conversational search.
  - Retriever: We added the latest techniques that leverage LLMs to expand the training corpus for retrievers or to enhance retrievers' architectures.
  - Reranker: We added recent LLM-based ranking works to each of the three part: Utilizing LLMs as Supervised Rerankers, Utilizing LLMs as Unsupervised Rerankers, and Utilizing LLMs for Training Data Augmentation.
  - Reader: We added the latest studies in LLM-enhanced reader area, including a section introducing the reference compression technique, a section discussing the applications of LLM-enhanced readers, and a section analyzing the characteristics of LLM-enhanced readers.
  - Future Direction: We added a section about search agents and a section discussing the bias caused by leveraging LLMs into IR systems.

## 📋 Table of Content
- [Query Rewriter](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#query-rewriter)
  - [Prompting Methods](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#prompting-methods)
  - [Fine-tuning Methods](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#fine-tuning-methods)
  - [Knowledge Distillation Methods](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#knowledge-distillation-methods)
- [Retriever](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#retriever)
  - [Leveraging LLMs to Generate Search Data](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#leveraging-llms-to-generate-search-data)
  - [Employing LLMs to Enhance Model Architecture](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#employing-llms-to-enhance-model-architecture)
- [Re-ranker](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#re-ranker)
  - [Utilizing LLMs as Supervised Rerankers](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#utilizing-llms-as-supervised-rerankers)
  - [Utilizing LLMs as Unsupervised Rerankers](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#utilizing-llms-as-unsupervised-rerankers)
  - [Utilizing LLMs for Training Data Augmentation](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#utilizing-llms-for-training-data-augmentation)
  - [Reasoning-intensive Rerankers](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#reasoning-intensive-rerankers)
- [Reader](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#reader)
  - [Passive Reader](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#passive-reader)
  - [Active Reader](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#active-reader)
  - [Compressor](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#compressor)
  - [Analysis](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#analysis)
  - [Applications](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#applications)
- [Search Agent](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#search-agent)
  - [Information Seeking Module](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#information-seeking-module)
- [Other Resources](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#other-resources)

## 📄 Paper List

### Query Rewriter
#### Prompting Methods
1. **Query2doc: Query Expansion with Large Language Models**, _Wang et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2303.07678.pdf)\]
2. **Generative and Pseudo-Relevant Feedback for Sparse, Dense and Learned Sparse Retrieval**, _Mackie et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2305.07477.pdf)\]
3. **Generative Relevance Feedback with Large Language Models**, _Mackie et al._, SIGIR 2023 (short paper). \[[Paper](https://arxiv.org/pdf/2304.13157.pdf)\]
4. **GRM: Generative Relevance Modeling Using Relevance-Aware Sample Estimation for Document Retrieval**, _Mackie et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2306.09938.pdf)\]
5. **Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search**, _Mao et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2303.06573.pdf)\]
6. **Precise Zero-Shot Dense Retrieval without Relevance Labels**, _Gao et al._, ACL 2023.  \[[Paper](https://aclanthology.org/2023.acl-long.99.pdf)\]
7. **Query Expansion by Prompting Large Language Models**, _Jagerman et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2305.03653.pdf)\]
8. **Large Language Models are Strong Zero-Shot Retriever**, _Shen et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2304.14233.pdf)\]
9. **Enhancing Conversational Search: Large Language Model-Aided Informative Query Rewriting**, _Ye et al._, EMNLP 2023 (Findings).  \[[Paper](https://aclanthology.org/2023.findings-emnlp.398.pdf)\]
10. **Can generative llms create query variants for test collections? an exploratory study**, _M. Alaofi et al._, SIGIR 2023 (short paper). \[[Paper](https://arxiv.org/pdf/2501.17981)\]
11. **Corpus-Steered Query Expansion with Large Language Models**, _Lei et al._, EACL 2024 (Short Paper).  \[[Paper](https://aclanthology.org/2024.eacl-short.34.pdf)\]
12. **Large language model based long-tail query rewriting in taobao search**, _Peng et al._, WWW 2024.  \[[Paper](https://arxiv.org/abs/2311.03758)\]
13. **Can Query Expansion Improve Generalization of Strong Cross-Encoder Rankers?**, _Li et al._, SIGIR 2024.  \[[Paper](https://arxiv.org/pdf/2311.09175.pdf)\]
14. **Query Performance Prediction using Relevance Judgments Generated by Large Language Models**, _Meng et al._, arXiv 2024.  \[[Paper](https://arxiv.org/abs/2404.01012)\]
15. **RaFe: Ranking Feedback Improves Query Rewriting for RAG**, _Mao et al._, arXiv 2024.  \[[Paper](https://arxiv.org/abs/2405.14431)\]
16. **Crafting the Path: Robust Query Rewriting for Information Retrieval**, _Baek et al._, arXiv 2024.  \[[Paper](https://arxiv.org/abs/2407.12529)\]
17. **Query Rewriting for Retrieval-Augmented Large Language Models**, _Ma et al._, arXiv 2023.  \[[Paper](https://arxiv.org/abs/2305.14283)\]
    

#### Fine-tuning Methods
1. **QUILL: Query Intent with Large Language Models using Retrieval Augmentation and Multi-stage Distillation**, _Srinivasan et al._, EMNLP 2022 (Industry). \[[Paper](https://aclanthology.org/2022.emnlp-industry.50.pdf)\] (This paper explore fine-tuning methods in baseline experiments.)

#### Knowledge Distillation Methods
1. **QUILL: Query Intent with Large Language Models using Retrieval Augmentation and Multi-stage Distillation**, _Srinivasan et al._, EMNLP 2022 (Industry). \[[Paper](https://aclanthology.org/2022.emnlp-industry.50.pdf)\]
2. **Knowledge Refinement via Interaction Between Search Engines and Large Language Models**, _Feng et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2305.07402.pdf)\]
3. **Query Rewriting for Retrieval-Augmented Large Language Models**, _Ma et al._, arXiv 2023.  \[[Paper](https://arxiv.org/pdf/2305.14283.pdf)\]

### Retriever
#### Leveraging LLMs to Generate Search Data
1. **InPars: Data Augmentation for Information Retrieval using Large Language Models**, _Bonifacio et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2202.05144.pdf)\]
2. **Pre-training with Large Language Model-based Document Expansion for Dense Passage Retrieval**, _Ma et al._, arXiv 2023. \[[Paper](https://doi.org/10.48550/arXiv.2308.08285)\]
3. **InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval**, _Jeronymo et al._, arXiv 2023. \[[Paper](https://arxiv.org/abs/2301.01820)\]
4. **Promptagator: Few-shot Dense Retrieval From 8 Examples**, _Dai et al._, ICLR 2023. \[[Paper](https://arxiv.org/pdf/2209.11755.pdf)\]
5. **AugTriever: Unsupervised Dense Retrieval by Scalable Data Augmentation**, _Meng et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2212.08841.pdf)\]
6. **UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers**, _Saad-Falco et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2303.00807.pdf)\]
7. **Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models**, _Peng et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.08303.pdf)\]
8. **CONVERSER: Few-shot Conversational Dense Retrieval with Synthetic Data Generation**, _Huang et al._, ACL 2023. \[[Paper](https://aclanthology.org/2023.sigdial-1.34)\]
9. **Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval**, _Thakur et al._, arXiv 2023. \[[Paper](https://doi.org/10.48550/arXiv.2311.05800)\]
10. **Questions Are All You Need to Train a Dense Passage Retriever**, _Sachan et al._, ACL 2023. \[[Paper](https://aclanthology.org/2023.tacl-1.35.pdf)\]
11. **Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators**, _Chen et al._, EMNLP 2023. \[[Paper](https://arxiv.org/abs/2310.07289)\]
12. **Gecko: Versatile Text Embeddings Distilled from Large Language Models**, _Lee et al._, arXiv 2024. \[[Paper](https://arxiv.org/abs/2403.20327)\]
13. **Improving Text Embeddings with Large Language Models**, _Wang et al._, ACL 2024. \[[Paper](https://aclanthology.org/2024.acl-long.642/)\]
    
#### Employing LLMs to Enhance Model Architecture
1. **Text and Code Embeddings by Contrastive Pre-Training**, _Neelakantan et al._, arXiv 2022. \[[Paper](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf)\]
2. **Fine-Tuning LLaMA for Multi-Stage Text Retrieval**, _Ma et al._, arXiv 2023. \[[Paper](https://doi.org/10.48550/arXiv.2310.08319)\]
3. **Large Dual Encoders Are Generalizable Retrievers**, _Ni et al._, EMNLP 2022. \[[Paper](https://aclanthology.org/2022.emnlp-main.669.pdf)\]
4. **Task-aware Retrieval with Instructions**, _Asai et al._, ACL 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-acl.225.pdf)\]
5. **Transformer memory as a differentiable search index**, _Tay et al._, NeurIPS 2022. \[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/892840a6123b5ec99ebaab8be1530fba-Paper-Conference.pdf)\]
6. **Large Language Models are Built-in Autoregressive Search Engines**, _Ziems et al._, ACL 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-acl.167.pdf)\]
7. **Chatretriever: Adapting large language models for generalized and robust conversational dense retrieval**, _Mao et al._, arXiv. \[[Paper](https://arxiv.org/pdf/2404.13556)\]
8. **How does generative retrieval scale to millions of passages?**, _Pradeep et al._, ACL 2023. \[[Paper](https://aclanthology.org/2023.emnlp-main.83.pdf)\]
9. **CorpusLM: Towards a Unified Language Model on Corpus for Knowledge-Intensive Tasks**, _Li et al._, SIGIR. \[[Paper](https://dl.acm.org/doi/10.1145/3626772.3657778)]
   
### Reranker

#### Utilizing LLMs as Supervised Rerankers

1. **Multi-Stage Document Ranking with BERT**, *Nogueira et al.*, arXiv 2019. \[[Paper](https://arxiv.org/pdf/1910.14424.pdf)\] 
2. **Document Ranking with a Pretrained Sequence-to-Sequence Model**, _Nogueira et al._, EMNLP 2020 (Findings). \[[Paper](https://aclanthology.org/2020.findings-emnlp.63.pdf)\]
3. **Text-to-Text Multi-view Learning for Passage Re-ranking**, _Ju et al._, SIGIR 2021 (Short Paper). \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463048)\] 
4. **The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models**, _Pradeep et al._, arXiv 2021. \[[Paper](https://arxiv.org/pdf/2101.05667.pdf)\] 
5. **RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses**, _Zhuang et al._, SIGIR 2023 (Short Paper). \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3539618.3592047)\] 
6. **Fine-Tuning LLaMA for Multi-Stage Text Retrieval**, *Ma et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.08319v1.pdf)\]
7. **A Two-Stage Adaptation of Large Language Models for Text Ranking**, *Zhang et al.*, ACL 2024 (Findings). \[[Paper](https://aclanthology.org/2024.findings-acl.706.pdf)\]
8. **Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models**, *Zhang et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2312.02969.pdf)\]
9. **ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval**, *Yoon et al.*, ACL 2024. \[[Paper](https://aclanthology.org/2024.acl-long.125.pdf)\]
10. **Q-PEFT: Query-dependent Parameter Efficient Fine-tuning for Text Reranking with Large Language Models**, *Peng et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2404.04522)\]
11. **Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models**, *Liu et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.14848)\]

#### Utilizing LLMs as Unsupervised Rerankers

1. **Holistic Evaluation of Language Models**, _Liang et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2211.09110.pdf)\] 
2. **Improving Passage Retrieval with Zero-Shot Question Generation**, _Sachan et al._, EMNLP 2022. \[[Paper](https://aclanthology.org/2022.emnlp-main.249.pdf)\] 
3. **Discrete Prompt Optimization via Constrained Generation for Zero-shot Re-ranker**, _Cho et al._, ACL 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-acl.61.pdf)\] 
4. **Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking**, *Zhuang et al.*, EMNLP 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-emnlp.590.pdf)\]
5. **PaRaDe: Passage Ranking using Demonstrations with Large Language Models**, *Drozdov et al.*, EMNLP 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-emnlp.950.pdf)\]
6. **Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels**, *Zhuang et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.14122.pdf)\]
7. **Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent**, _Sun et al._, EMNLP 2023. \[[Paper](https://aclanthology.org/2023.emnlp-main.923.pdf)\] 
8. **Zero-Shot Listwise Document Reranking with a Large Language Model**, _Ma et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2305.02156.pdf)\] 
9. **Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models**, *Tang et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.07712.pdf)\]
10. **Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting**, _Qin et al._, NAACL 2024 (Findings). \[[Paper](https://aclanthology.org/2024.findings-naacl.97.pdf)\] 
11. **A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models**, *Zhuang et al.*, SIGIR 2024. \[[Paper](https://arxiv.org/pdf/2310.09497.pdf)\]
12. **InstUPR: Instruction-based Unsupervised Passage Reranking with Large Language Models**, *Huang and Chen*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2403.16435.pdf)\]
13. **Generating Diverse Criteria On-the-Fly to Improve Point-wise LLM Rankers**, *Guo et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2404.11960)\]
14. **DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task**, *Liu et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.16332)\]
15. **An Investigation of Prompt Variations for Zero-shot LLM-based Rankers**, *Sun et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.14117)\]
16. **TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy**, *Chen et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.11678)\]
17. **Top-Down Partitioning for Efficient List-Wise Ranking**, *Parry et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2405.14589)\]
18. **PRP-Graph: Pairwise Ranking Prompting to LLMs with Graph Aggregation for Effective Text Re-ranking**, *Luo et al.*, ACL 2024. \[[Paper](https://aclanthology.org/2024.acl-long.313.pdf)\]
19. **Consolidating Ranking and Relevance Predictions of Large Language Models through Post-Processing**, *Yan et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2404.11791)\]
20. **Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models**, *Liu et al.*, ACL 2025. \[[Paper](https://aclanthology.org/2025.acl-long.8.pdf)\]
21. **CoRanking: Collaborative Ranking with Small and Large Ranking Agents**, *Liu et al.*, EMNLP 2025 (Findings). \[[Paper](https://arxiv.org/pdf/2503.23427)\]
22. **APEER : Automatic Prompt Engineering Enhances Large Language Model Reranking**, *Jin et al.*, WWW 2025. \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3701716.3717574)\]
23. **Consolidating Ranking and Relevance Predictions of Large Language Models through Post-Processing**, *Yan et al.*, EMNLP 2024. \[[Paper](https://aclanthology.org/2024.emnlp-main.25.pdf)\]

#### Utilizing LLMs for Training Data Augmentation

1. **ExaRanker: Explanation-Augmented Neural Ranker**, _Ferraretto et al._, SIGIR 2023 (Short Paper). \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3539618.3592067)\]
2. **InPars-Light: Cost-Effective Unsupervised Training of Efficient Rankers**, _Boytsov et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2301.02998.pdf)\]
3. **Generating Synthetic Documents for Cross-Encoder Re-Rankers**, _Askari et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2305.02320.pdf)\]
4. **Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers**, *Sun et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2311.01555.pdf)\]
5. **RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models**, *Pradeep et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2309.15088.pdf)\]
6. **RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!**, *Pradeep et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2312.02724.pdf)\]
7. **ExaRanker-Open: Synthetic Explanation for IR using Open-Source LLMs**, *Ferraretto et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2402.06334)\]
8. **Expand, Highlight, Generate: RL-driven Document Generation for Passage Reranking**, *Askari et al.*, EMNLP 2023. \[[Paper](https://aclanthology.org/2023.emnlp-main.623.pdf)\]
9. **FIRST: Faster Improved Listwise Reranking with Single Token Decoding**, *Reddy et al.*, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.15657)\]

#### Reasoning-intensive Rerankers

1. **ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability**, _Liu et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2508.07050)\]
2. **Rank1: Test-Time Compute for Reranking in Information Retrieval**, *Weller et al.*, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2502.18418)\]
3. **Rank-K: Test-Time Reasoning for Listwise Reranking**, *Yang et al.*, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2505.14432)\]
4. **REARANK: Reasoning Re-ranking Agent via Reinforcement Learning**, *Zhang et al.*, arXiv 2025. \[[Paper](https://arxiv.org/abs/2505.20046)\]
5. **Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning**, *Zhuang et al.*, arXiv 2025. \[[Paper](https://arxiv.org/abs/2503.06034)\]
6. **TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking**, *Fan et al.*, arXiv 2025. \[[Paper](https://arxiv.org/abs/2508.09539)\]

### Reader
#### Passive Reader
1. **REALM: Retrieval-Augmented Language Model Pre-Training**, _Guu et al._, ICML 2020. \[[Paper](https://proceedings.mlr.press/v119/guu20a/guu20a.pdf)\]
2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, _Lewis et al._, NeurIPS 2020. \[[Paper](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)\]
3. **REPLUG: Retrieval-Augmented Black-Box Language Models**, _Shi et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2301.12652.pdf)\]
4. **Atlas: Few-shot Learning with Retrieval Augmented Language Models**, _Izacard et al._, JMLR 2023. \[[Paper](https://jmlr.org/papers/volume24/23-0037/23-0037.pdf)\]
5. **Internet-augmented Language Models through Few-shot Prompting for Open-domain Question Answering**, _Lazaridou et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2203.05115.pdf)\]
6. **Rethinking with Retrieval: Faithful Large Language Model Inference**, _He et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2301.00303.pdf)\]
7. **FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation**, _Vu et al._, arxiv 2023. \[[Paper](https://doi.org/10.48550/arXiv.2310.03214)\]
8. **Enabling Large Language Models to Generate Text with Citations**, _Gao et al._, EMNLP 2023. \[[Paper](https://aclanthology.org/2023.emnlp-main.398.pdf)\]
9. **Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models**, _Yu et al._, arxiv 2023. \[[Paper](https://arxiv.org/pdf/2311.09210.pdf)\]
10. **Improving Retrieval-Augmented Large Language Models via Data Importance Learning**, _Lyu et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.03027.pdf)\]
11. **Search Augmented Instruction Learning**, _Luo et al._, EMNLP 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-emnlp.242.pdf)\]
12. **RADIT: Retrieval-Augmented Dual Instruction Tuning**, _Lin et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.01352.pdf)\]
13. **Improving Language Models by Retrieving from Trillions of Tokens**, _Borgeaud et al._, ICML 2022. \[[Paper](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf)\]
14. **In-Context Retrieval-Augmented Language Models**, _Ram et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2302.00083.pdf)\]
15. **Interleaving Retrieval with Chain-of-thought Reasoning for Knowledge-intensive Multi-step Questions**, _Trivedi et al._, ACL 2023. \[[Paper](https://aclanthology.org/2023.acl-long.557.pdf)\]
16. **Improving Language Models via Plug-and-Play Retrieval Feedback**, _Yu et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2305.14002.pdf)\]
17. **Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy**, _Shao et al._, EMNLP 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-emnlp.620.pdf)\]
18. **Retrieval-Generation Synergy Augmented Large Language Models**, _Feng et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.05149.pdf)\]
19. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, _Asai et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.11511.pdf)\]
20. **Active Retrieval Augmented Generation**, _Jiang et al._, EMNLP 2023. \[[Paper](https://aclanthology.org/2023.emnlp-main.495.pdf)\]
#### Active Reader
1. **Measuring and Narrowing the Compositionality Gap in Language Models**, _Press et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2210.03350.pdf)\]
2. **DEMONSTRATE–SEARCH–PREDICT: Composing Retrieval and Language Models for Knowledge-intensive NLP**, _Khattab et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2212.14024.pdf)\]
3. **Answering Questions by Meta-Reasoning over Multiple Chains of Thought**, _Yoran et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2304.13007.pdf)\]
4. **PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers**, _Lee ei al._, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.12430.pdf)]
5. **Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs**, _Wang et al._, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2406.14282.pdf)]
#### Compressor
1. **LeanContext: Cost-Efficient Domain-Specific Question Answering Using LLMs**, _Arefeen et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2309.00841.pdf)\]
2. **RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation**, _Xu et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.04408.pdf)\]
3. **TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction**, _Liu et al._, EMNLP 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-emnlp.655.pdf)\]
4. **Learning to Filter Context for Retrieval-Augmented Generation**, _Wang et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2311.08377.pdf)\]
#### Analysis
1. **Lost in the Middle: How Language Models Use Long Contexts**, _Liu et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.03172.pdf)\]
2. **Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation**, _Ren et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.11019.pdf)\]
3. **Exploring the Integration Strategies of Retriever and Large Language Models**, _Liu et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2308.12574.pdf)\]
4. **Characterizing Attribution and Fluency Tradeoffs for Retrieval-Augmented Large Language Models**, _Aksitov et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2302.05578.pdf)\]
5. **When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories**, _Mallen et al._, ACL 2023. \[[Paper](https://aclanthology.org/2023.acl-long.546.pdf)\]
#### Applications
1. **Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering**, _Wang et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2309.02233.pdf)\]
2. **ATLANTIC: Structure-Aware Retrieval-Augmented Language Model for Interdisciplinary Science**, _Munikoti et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2311.12289.pdf)\]
3. **Crosslingual Retrieval Augmented In-context Learning for Bangla**, _Li et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2311.00587.pdf)\]
4. **Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature**, _Lozano et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.16146.pdf)\]
5. **Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models**, _Zhang et al._, ICAIF 2023. \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3604237.3626866)\]
6. **Interpretable Long-Form Legal Question Answering with Retrieval-Augmented Large Language Models**, _Louis et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2309.17050.pdf)\]
7. **RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit**, _Liu et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2306.05212.pdf)\]
8. **Chameleon: a Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models**, _Jiang et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.09949.pdf)\]
9. **RaLLe: A Framework for Developing and Evaluating Retrieval-Augmented Large Language Models**, _Hoshi et al._, EMNLP 2023. \[[Paper](https://aclanthology.org/2023.emnlp-demo.4.pdf)\]
10. **Don't forget private retrieval: distributed private similarity search for large language models**, _Zyskind et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2311.12955.pdf)\]
   



### Search Agent
#### Information Seeking Module
1. **A cognitive writing perspective for constrained long-form text generation**, _Wan et al._, ACL 2025 (Findings). \[[Paper](https://aclanthology.org/2025.findings-acl.511.pdf)\]
2. **CoSearchAgent: A Lightweight Collaborative Search Agent with Large Language Models**, _Gong et al._, SIGIR 2024. \[[Paper](https://arxiv.org/pdf/2402.06360)\]
3. **Search-o1: Agentic search-enhanced large reasoning models**, _Li et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2501.05366)\]
4. **Agent Laboratory: Using LLM Agents as Research Assistants**, _Schmidgall et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2501.04227)\]
5. **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery**, _Lu et al._, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2408.06292)\]
6. **Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models**, _Yu et al._, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2411.19443)\]
7. **SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis**, _Sun et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2505.16834)\]
8. **Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning**, _Dong et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2505.16410)\]
9. **ZeroSearch: Incentivize the Search Capability of LLMs without Searching**, _Sun et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2505.04588)\]
10. **Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution**, _Qiu et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2505.20286)\]

#### Benchmarks and Resources
1. **TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension**, _Joshi et al._, ACL 2017. \[[Paper](https://aclanthology.org/P17-1147.pdf)\]
2. **Measuring short-form factuality in large language models**, _Wei et al._, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2411.04368v1)\]
3. **When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories**, _Mallen et al._, ACL 2023. \[[Paper](https://arxiv.org/pdf/2212.10511)\]
4. **Natural Questions: a Benchmark for Question Answering Research**, _Kwiatkowski et al._, ACL 2019. \[[Paper](https://aclanthology.org/Q19-1026.pdf)\]
5. **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**, _Yang et al._, EMNLP 2018. \[[Paper](https://aclanthology.org/D18-1259.pdf)\]
6. **Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps**, _Ho et al._, COLING 2020. \[[Paper](https://arxiv.org/pdf/2011.01060)\]
7. **Humanity's Last Exam**, _Phan et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2501.14249)\]
8. **BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents**, _Wei et al._, arXiv 2025. \[[Paper](https://arxiv.org/pdf/2504.12516)\]
9. **GAIA: a benchmark for General AI Assistants**, _Mialon et al._, ICLR 2024. \[[Paper](https://arxiv.org/pdf/2311.12983)\]
10. **AssistantBench: Can Web Agents Solve Realistic and Time-Consuming Tasks?**, _Yoran et al._, EMNLP 2024. \[[Paper](https://arxiv.org/pdf/2407.15711)\]
11. **Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks**, _Fourney et al._, arXiv 2024. \[[Paper](https://arxiv.org/pdf/2411.04468)\]
12. **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**, _Jimenez et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.06770)\]
13. **OctoPack: Instruction Tuning Code Large Language Models**, _Muennighoff et al._, ICLR 2024. \[[Paper](https://arxiv.org/pdf/2308.07124)\]
14. **MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering**, _Chan et al._, ICLR 2025. \[[Paper](https://arxiv.org/pdf/2410.07095)\]
15. **MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation**, _Huang et al._, ICML 2024. \[[Paper](https://arxiv.org/pdf/2310.03302)\]
16. **RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts**, _Wijk et al._, Arxiv 2024. \[[Paper](https://arxiv.org/pdf/2411.15114)\]
17. **ResearchTown: Simulator of Human Research Community**, _Yu et al._, Arxiv 2024. \[[Paper](https://arxiv.org/pdf/2412.17767)\]
18. **WebArena: A Realistic Web Environment for Building Autonomous Agents**, _Zhou et al._, ICLR 2024. \[[Paper](https://arxiv.org/pdf/2307.13854)\]
19. **Spa-Bench: a comprehensive Benchmark for Smartphone Agent Evaluation**, _Chen et al._, ICLR 2025. \[[Paper](https://arxiv.org/pdf/2410.15164)\]
20. **WebWalker: Benchmarking LLMs in Web Traversal**, _Wu et al._, ACL 2025. \[[Paper](https://arxiv.org/pdf/2501.07572)\]
21. **WebDancer: Towards Autonomous Information Seeking Agency**, _Wu et al._, Arxiv 2025. \[[Paper](https://arxiv.org/pdf/2505.22648)\]
22. **WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization**, _Tao et al._, Arxiv 2025. \[[Paper](https://arxiv.org/pdf/2507.15061)\]
23. **WebSailor: Navigating Super-human Reasoning for Web Agent**, _Li et al._, Arxiv 2025. \[[Paper](https://arxiv.org/pdf/2507.02592)\]



### Other Resources
1. **ACL 2023 Tutorial: Retrieval-based Language Models and Applications**, _Asai et al._, ACL 2023. \[[Link](https://acl2023-retrieval-lm.github.io/)\]
2. **A Survey of Large Language Models**, _Zhao et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2303.18223.pdf)\]
3. **Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community**, _Ai et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.09751.pdf)\]

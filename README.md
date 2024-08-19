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
- [Reader](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#reader)
  - [Passive Reader](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#passive-reader)
  - [Active Reader](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#active-reader)
  - [Compressor](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#compressor)
  - [Analysis](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#analysis)
  - [Applications](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#applications)
- [Search Agent](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#search-agent)
  - [Static Agent](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#static-agent)
  - [Dynamic Agent](https://github.com/RUC-NLPIR/LLM4IR-Survey/tree/main#dynamic-agent)
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
10. **Can generative llms create query variants for test collections? an exploratory study**, _M. Alaofi et al._, SIGIR 2023 (short paper). \[[Paper](https://marksanderson.org/publications/my_papers/SIGIR_23_GPT.pdf)\]
11. **Corpus-Steered Query Expansion with Large Language Models**, _Lei et al._, EACL 2024 (Short Paper).  \[[Paper](https://aclanthology.org/2024.eacl-short.34.pdf)\]
12. **PromptReps: Prompting Large Language Models to Generate Dense and Sparse Representations for Zero-Shot Document Retrieval**, _Zhuang et al._, arXiv 2024.  \[[Paper](https://arxiv.org/pdf/2404.18424)\]

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

#### Employing LLMs to Enhance Model Architecture
1. **Text and Code Embeddings by Contrastive Pre-Training**, _Neelakantan et al._, arXiv 2022. \[[Paper](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf)\]
2. **Fine-Tuning LLaMA for Multi-Stage Text Retrieval**, _Ma et al._, arXiv 2023. \[[Paper](https://doi.org/10.48550/arXiv.2310.08319)\]
3. **Large Dual Encoders Are Generalizable Retrievers**, _Ni et al._, EMNLP 2022. \[[Paper](https://aclanthology.org/2022.emnlp-main.669.pdf)\]
4. **Task-aware Retrieval with Instructions**, _Asai et al._, ACL 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-acl.225.pdf)\]
5. **Transformer memory as a differentiable search index**, _Tay et al._, NeurIPS 2022. \[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/892840a6123b5ec99ebaab8be1530fba-Paper-Conference.pdf)\]
6. **Large Language Models are Built-in Autoregressive Search Engines**, _Ziems et al._, ACL 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-acl.167.pdf)\]

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

#### Utilizing LLMs for Training Data Augmentation
1. **ExaRanker: Explanation-Augmented Neural Ranker**, _Ferraretto et al._, SIGIR 2023 (Short Paper). \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3539618.3592067)\]
2. **InPars-Light: Cost-Effective Unsupervised Training of Efficient Rankers**, _Boytsov et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2301.02998.pdf)\]
3. **Generating Synthetic Documents for Cross-Encoder Re-Rankers**, _Askari et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2305.02320.pdf)\]
4. **Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers**, *Sun et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2311.01555.pdf)\]
5. **RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models**, *Pradeep et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2309.15088.pdf)\]
6. **RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!**, *Pradeep et al.*, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2312.02724.pdf)\]

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
#### Static Agent
1. **LaMDA: Language Models for Dialog Applications**, _Thoppilan et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2201.08239.pdf)\]
2. **Language Models that Seek for Knowledge: Modular Search & Generation for Dialogue and Prompt Completion**, _Shuster et al._, EMNLP 2022 (Findings). \[[Paper](https://aclanthology.org/2022.findings-emnlp.27.pdf)\]
3. **Teaching language models to support answers with verified quotes**, _Menick et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2203.11147.pdf)\]
4. **WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences**, _Liu et al._, KDD 2023. \[[Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599931)\]
5. **A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis**, _Gur et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.12856.pdf)\]
6. **Know Where to Go: Make LLM a Relevant, Responsible, and Trustworthy Searcher**, _Shi et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2310.12443.pdf)\]
#### Dynamic Agent
1. **WebGPT: Browser-assisted question-answering with human feedback**, _Nakano et al._, arXiv 2021. \[[Paper](https://arxiv.org/pdf/2112.09332.pdf)\]
2. **WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents**, _Yao et al._, arXiv 2022. \[[Paper](https://arxiv.org/pdf/2207.01206.pdf)\]
3. **WebCPM: Interactive Web Search for Chinese Long-form Question Answering**, _Qin et al._, ACL 2023. \[[Paper](https://aclanthology.org/2023.acl-long.499.pdf)\]
4. **Mind2Web: Towards a Generalist Agent for the Web**, _Deng et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2306.06070.pdf)\]
5. **WebArena: A Realistic Web Environment for Building Autonomous Agents**, _Zhou et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.13854.pdf)\]
6. **Hierarchical Prompting Assists Large Language Model on Web Navigation**, _Sridhar et al._, EMNLP 2023 (Findings). \[[Paper](https://aclanthology.org/2023.findings-emnlp.685.pdf)\]

### Other Resources
1. **ACL 2023 Tutorial: Retrieval-based Language Models and Applications**, _Asai et al._, ACL 2023. \[[Link](https://acl2023-retrieval-lm.github.io/)\]
2. **A Survey of Large Language Models**, _Zhao et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2303.18223.pdf)\]
3. **Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community**, _Ai et al._, arXiv 2023. \[[Paper](https://arxiv.org/pdf/2307.09751.pdf)\]

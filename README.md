# awesome-text-summarization

Text summarization starting from scratch.

This repository will keep updating...



[TOC]



## Basic Concept

### Definition

Summarization is the task of producing a shorter version of one or several documents that preserves most of the input's meaning.



### Types of summarization

**Extractive summaries** (extracts) are produced by concatenating
several sentences taken exactly as they appear in the materials being
summarized.

**Abstractive summaries** (abstracts), are written to convey
the main information in the input and may reuse phrases or clauses
from it, but the summaries are overall expressed in the words of the
summary author.



### Summary Informativeness evaluation

* **ROUGE-N**: measures the N-gram units common between a particular summary and a col-
  lection of reference summaries where N determines the N-gram’s length. E.g., **ROUGE-1**
  for unigrams and **ROUGE-2** for bi-grams.
* **ROUGE-L**:  computes Longest Common Subsequence (LCS) metric.
* **BLUE** : BLEU is basically calculated on the n-gram co-occerance between the generated summary and the gold (You don't need to specify the "n" unlike ROUGE).
* **METEOR** : based on the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of unigram [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall), with recall weighted higher than precision.




## DataSet

- [Annotated English Gigaword](https://catalog.ldc.upenn.edu/LDC2012T21)

  - for sentence summarization

- [CNN/Daily Mail dataset](https://cs.nyu.edu/~kcho/DMQA/)

  - for document summatization

- [DUC 2004](http://www.cis.upenn.edu/~nlp/corpora/sumrepo.html)

- [CORNELL NEWSROOM](https://summari.es/)

  - is a large dataset for training and evaluating summarization systems. It contains 1.3 million articles and summaries written by authors and editors in the newsrooms of 38 major publications. The summaries are obtained from search and social metadata between 1998 and 2017 and use a variety of summarization strategies combining *extraction* and *abstraction*.

- [Google Dataset](https://github.com/google-research-datasets/sentence-compression)

  - Large corpus of uncompressed and compressed sentences from news articles.

    ​

## Papers

### Survey

[Recent automatic text summarization techniques:a survey](./TextSummary/survey/Gambhir-Gupta2017_Article_RecentAutomaticTextSummarizati.pdf)

[Automatic summarization](./TextSummary/1124/Automatic summarization.pdf)



### Abstractive Document summarization



**words-lvt2k-temp-att (Nallapti et al., 2016)** : [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](http://www.aclweb.org/anthology/K16-1028)

**Graph-Based Attn** : [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](http://aclweb.org/anthology/P17-1108)

Pointer-generator + coverage (See et al., 2017) : [Get To The Point: Summarization with Pointer-Generator Networks](http://aclweb.org/anthology/P17-1099)

**KIGN+Prediction-guide** : [Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network](http://aclweb.org/anthology/N18-2009)

**Explicit Info Selection Modeling(Li et al., 2018a)** : [Improving Neural Abstractive Document Summarization with Explicit Information Selection Modeling](http://aclweb.org/anthology/D18-1205)

**Structural Regularization(Li et al., 2018b)** : [Improving Neural Abstractive Document Summarization with Structural Regularization](http://aclweb.org/anthology/D18-1441)

**end2end w/ inconsistency loss (Hsu et al., 2018)**: [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](http://aclweb.org/anthology/P18-1013)

**Pointer + Coverage + EntailmentGen + QuestionGen** (Guo et al., 2018) : [Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation](http://aclweb.org/anthology/P18-1064)



#### Based Reinforcement Learning

**ML+RL ROUGE+Novel, with LM (Kryscinski et al., 2018)**  : [Improving Abstraction in Text Summarization](http://aclweb.org/anthology/D18-1207)

**RL + pg + cbdec (Jiang and Bansal, 2018)**: [Closed-Book Training to Improve Summarization Encoder Memory](http://aclweb.org/anthology/D18-1440)

**rnn-ext + abs + RL + rerank (Chen and Bansal, 2018)**: [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](http://aclweb.org/anthology/P18-1061)

**ML+RL, with intra-attention** : [A Deep Reinforced Model for Abstractive Summarization](https://openreview.net/pdf?id=HkAClQgA-)

**ML+RL ROUGE+Novel, with LM** : [Improving Abstraction in Text Summarization](http://aclweb.org/anthology/D18-1207)

**GAN** : [Generative Adversarial Network for Abstractive Text Summarization](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492)

**DCA (Celikyilmaz et al., 2018)** : [Summarization](http://aclweb.org/anthology/N18-1150)

**ROUGESal+Ent RL** (Pasunuru and Bansal, 2018): [Multi-Reward Reinforced Summarization with Saliency and Entailment](http://aclweb.org/anthology/N18-2102)



### Extractive Document summarization

**TEXTRANK(graph based)**: [TextRank: Bringing Order intoTexts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

**SWAP-NET** : [Extractive Summarization with SWAP-NET: Sentences and Words from Alternating Pointer Networks](http://aclweb.org/anthology/P18-1014)

**NN-SE** : [Neural summarization by extracting sentences and words

**HSASS** : [A Hierarchical Structured Self-Attentive Model for Extractive Document Summarization (HSSAS)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344797)

**NeuSUM (Zhou et al., 2018)** : [Neural Document Summarization by Jointly Learning to Score and Select Sentences](http://aclweb.org/anthology/P18-1061)

**Latent (Zhang et al., 2018**) : [Neural Latent Extractive Document Summarization](http://aclweb.org/anthology/D18-1088)



#### Based Reinforcement Learning

**rnn-ext + RL** (Chen and Bansal, 2018): [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](http://aclweb.org/anthology/P18-1061)

**Bottom-Up Summarization** (Gehrmann et al., 2018): [Bottom-Up Abstractive Summarization](https://arxiv.org/abs/1808.10792)

](http://www.aclweb.org/anthology/P16-1046)

**BANDITSUM** :[BANDITSUM: Extractive Summarization as a Contextual Bandit](https://arxiv.org/abs/1809.09672)

**SummaRuNNer**: [A recurrent neural network based sequence model for extractive summarization of documents](https://arxiv.org/pdf/1611.04230.pdf)

**Refrech**: [Ranking sentences for extractive summarization with reinforcement learning](http://www.aclweb.org/anthology/N18-1158)

**DQN**: [Deep reinforcement learning for extractive document summarization](https://www.sciencedirect.com/science/article/pii/S0925231218300377):

**RNES w/o coherence** :[Learning to Extract Coherent Summary via Deep Reinforcement Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16838/16118)





## Sentence Summarization

**Re^3 Sum (Cao et al., 2018)** : [Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization](http://aclweb.org/anthology/P18-1015)

**FTSum_g (Cao et al., 2018)** : [Faithful to the Original: Fact Aware Neural Abstractive Summarization](https://arxiv.org/pdf/1711.04434.pdf)

**Seq2seq + E2T_cnn (Amplayo et al., 2018)** : [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://www.aclweb.org/anthology/N16-1012)

**EndDec+WFE (Suzuki and Nagata, 2017)** : [Cutting-off Redundant Repeating Generations for Neural Abstractive Summarization](http://aclweb.org/anthology/E17-2047)

**DRGD (Li et al., 2017)** : [Deep Recurrent Generative Decoder for Abstractive Text Summarization](http://aclweb.org/anthology/D17-1222)

**BiRNN + LM Evaluator (Zhao et al. 2018)** : [A Language Model based Evaluator for Sentence Compression](https://aclweb.org/anthology/P18-2028)



## Unsupervised Abstractive Summarization

**MeanSum** : [MeanSum: A Neural Model for Unsupervised Multi-document Abstractive Summarization](https://arxiv.org/abs/1810.05739)

**Semantic Abstractive Sum based AMR(2018 Dohare)**: [Unsupervised Semantic Abstractive Summarization](http://aclweb.org/anthology/P18-3011)

**Paraphrastic Sentence Fusion Model(2018 Nayeem)**: [Abstractive Unsupervised Multi-Document Summarization using Paraphrastic Sentence Fusion](http://aclweb.org/anthology/C18-1102)



## Multi Document Summarization

**(Z Cao 2017)** : [Improving Multi-Document Summarization via Text Classification](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14525/14077)

**Based AMR** : [Abstract Meaning Representation for Multi-Document Summarization.](https://arxiv.org/abs/1806.05655)

[Abstractive Unsupervised Multi-Document Summarization using Paraphrastic Sentence Fusion.](http://aclweb.org/anthology/C18-1102)

[Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization.](https://arxiv.org/abs/1808.06218)

[Salience Estimation via Variational Auto-Encoders for Multi-Document Summarization.](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14613) 

[Supervised Learning of Automatic Pyramid for Optimization-Based Multi-Document Summarization.](https://aclanthology.info/papers/P17-1100/p17-1100)

[Bringing Structure into Summaries: Crowdsourcing a Benchmark Corpus of Concept  Maps](https://aclweb.org/anthology/D17-1320)



## Evaluation Metrics

**ROUGE(2004)** :  [Rouge: A package for automatic evaluation of summaries](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/was2004.pdf)

**BLUE(2002)** : [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)

**BE(2006)** : [Automated Summarization Evaluation with Basic Elements](https://pdfs.semanticscholar.org/45fc/709a2fb8cd3cc71462c65e3d5e1bcb23c444.pdf?_ga=2.109764549.1171702135.1552302856-592061209.1552302856)

**Pyramid Method(2007)** : [Evaluating Content Selection in Summarization: The Pyramid Method](http://www.cs.columbia.edu/~ani/papers/pyramid.pdf)

**(2018 Shaflei)** : [Summarization Evaluation in the Absence of Human Model Summaries Using the Compositionality of Word Embeddings](http://aclweb.org/anthology/C18-1077)

**(2018 Honda)** : [Pruning Basic Elements for Better Automatic Evaluation of Summaries](http://aclweb.org/anthology/N18-2104)



## Other Resources

**awesome-text-summatization** : 

* [The guide to tackle with the Text Summarization](https://github.com/mathsyouth/awesome-text-summarization)
* [A curated list of resources dedicated to text summarization](https://github.com/mathsyouth/awesome-text-summarization)

**SOTA in summarizaiton** : [The current state-of-the-art](https://github.com/sebastianruder/NLP-progress/blob/master/english/summarization.md)


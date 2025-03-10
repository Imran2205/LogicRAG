# Logic-RAG

This is the implementation of paper: _"Logic-RAG: Augmenting Large Multimodal Models with Visual-Spatial 
Knowledge for Road Scene Understanding"_
(accepted to ICRA2025) [[Link](https://github.com/Imran2205/LogicRAG)]

Test Logic-RAG on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Imran2205/LogicRAG/blob/master/inference/inference.ipynb)

- Full documentation will be available soon.

---

## Overview
**Logic-RAG** is a novel Retrieval-Augmented Generation framework that enhances Large 
Multimodal Models' (LMMs) spatial reasoning capabilities in autonomous driving contexts. 
By constructing a dynamic knowledge base of object-object relationships in first-order 
logic (FOL), Logic-RAG significantly improves the accuracy of visual-spatial queries 
in driving scenarios.

This repository contains the complete implementation of the **Logic-RAG** framework, 
including the perception module, query-to-logic embedder, and logical inference engine. 
We provide inference scripts for our proposed framework along with baseline evaluation 
scripts for GPT-4V and Claude 3.5 models to facilitate comparison.

**Logic-RAG framework for visual-spatial reasoning:**
![Logic-RAG Architecture Overview](./figures/logic_rag_pipeline_full.png)

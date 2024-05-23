#  AdaGMLP: AdaBoost GNN-to-MLP Knowledge Distillation (AdaGMLP)

This is a PyTorch implementation of AdaBoost GNN-to-MLP Knowledge Distillation (AdaGMLP) which is built on the source code of KRD (https://github.com/LirongWu/KRD), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv)

* Two evaluation settings: transductive and inductive

* Various teacher GNN architectures (GCN, SAGE, GAT) and student MLPs

* Training paradigm for teacher GNNs and student MLPs



## Main Requirements

* numpy==1.21.6
* scipy==1.7.3
* torch==1.7.1
* dgl == 0.6.1


## Description

* train_and_eval.py  
  * train_teacher() -- Pre-train the teacher GNNs
  * train_student() -- Train the student MLPs with the pre-trained teacher GNNs
  * adagmlp_train_mini_batch() -- Train AdaGMLP student model
  * adagmlp_evaluate_mini_batch() -- Evaluate AdaGMLP student model

* models.py  
  * AdaGMLP() -- AdaGMLP student
  * MLP() -- student MLPs
  * GCN() -- GCN Classifier, working as teacher GNNs
  * GAT() -- GAT Classifier, working as teacher GNNs
  * GraphSAGE() -- GraphSAGE Classifier, working as teacher GNNs
  
* dataloader.py  

  * load_data() -- Load Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv datasets

* utils.py  
  * mask_features() -- Randomly mask a portion of features
  * set_seed() -- Set radom seeds for reproducible results
  * graph_split() -- Split the data for the inductive setting




## Running the code

1. Install the required dependency packages

2. To reproduce the results in paper, please use the command in ./train.sh

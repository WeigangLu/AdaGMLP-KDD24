#!/bin/bash

# Cora
python train.py --dataset cora --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 1024 --dropout_t 0.5 --dropout_s 0.7 --lamb 0.1 --tau 0.4 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 3 --beta 1.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.9 --aug_feat_missing_rate 0.1

python train.py --dataset cora --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 1024 --dropout_t 0.5 --dropout_s 0.5 --lamb 0.5 --tau 0.9 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.0 --aug_feat_missing_rate 0.0
 
python train.py --dataset cora --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 1024 --dropout_t 0.5 --dropout_s 0.1 --lamb 0.7 --tau 0.6 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 2.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.1 --aug_feat_missing_rate 0.1 

# Citeseer
python train.py --dataset citeseer --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 256 --dropout_t 0.4 --dropout_s 0.9 --lamb 0.1 --tau 0.3 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 3 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.9 --aug_feat_missing_rate 0.1

python train.py --dataset citeseer --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 1024 --dropout_t 0.6 --dropout_s 0.9 --lamb 0.1 --tau 0.3 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 3 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.9 --aug_feat_missing_rate 0.1

python train.py --dataset citeseer --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 512 --dropout_t 0.6 --dropout_s 0.9 --lamb 0.7 --tau 0.01 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 3 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.9 --aug_feat_missing_rate 0.0

# Pubmed
python train.py --dataset pubmed --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 256 --dropout_t 0.6 --dropout_s 0.7 --lamb 0.1 --tau 0.6 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 2.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.5 --aug_feat_missing_rate 0.1 

python train.py --dataset pubmed --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 256 --dropout_t 0.4 --dropout_s 0.7 --lamb 0.1 --tau 0.5 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.5 --aug_feat_missing_rate 0.1

python train.py --dataset pubmed --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 256 --dropout_t 0.3 --dropout_s 0.7 --lamb 0.2 --tau 0.05 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 3 --beta 1.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.7 --aug_feat_missing_rate 0.1


# Photo
python train.py --dataset amazon-photo --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.1 --num_layers 2 --hidden_dim 1024 --dropout_t 0.4 --dropout_s 0.8 --lamb 0.5 --tau 0.5 --learning_rate 0.01 --weight_decay 0.0 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 4.0 --selective 1
 
python train.py --dataset amazon-photo --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.1 --num_layers 2 --hidden_dim 512 --dropout_t 0.3 --dropout_s 0.6 --lamb 0.2 --tau 0.1 --learning_rate 0.005 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 0.5 --selective 1
 
python train.py --dataset amazon-photo --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.1 --num_layers 2 --hidden_dim 512 --dropout_t 0.5 --dropout_s 0.6 --lamb 0.1 --tau 0.1 --learning_rate 0.005 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 0.5 --selective 1
 

# CS
python train.py --dataset coauthor-cs --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 256 --dropout_t 0.5 --dropout_s 0.9 --lamb 0.1 --tau 0.9 --learning_rate 0.01 --weight_decay 0.001 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 3 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.0 --aug_feat_missing_rate 0.0
 
python train.py --dataset coauthor-cs --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 128 --dropout_t 0.6 --dropout_s 0.9 --lamb 0.1 --tau 0.9 --learning_rate 0.01 --weight_decay 0.001 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 2.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.9 --aug_feat_missing_rate 0.0
 
python train.py --dataset coauthor-cs --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 128 --dropout_t 0.6 --dropout_s 0.9 --lamb 0.1 --tau 0.5 --learning_rate 0.01 --weight_decay 0.001 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 2.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.0 --aug_feat_missing_rate 0.0
 

# Phy
python train.py --dataset coauthor-phy --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 1024 --dropout_t 0.4 --dropout_s 0.9 --lamb 0.2 --tau 0.05 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 0.5 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.1 --aug_feat_missing_rate 0.0
 
python train.py --dataset coauthor-phy --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 1024 --dropout_t 0.3 --dropout_s 0.3 --lamb 0.2 --tau 0.05 --learning_rate 0.01 --weight_decay 0.001 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 0.5 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.3 --aug_feat_missing_rate 0.0
 
python train.py --dataset coauthor-phy --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 2 --hidden_dim 256 --dropout_t 0.6 --dropout_s 0.9 --lamb 0.1 --tau 0.5 --learning_rate 0.01 --weight_decay 0.0005 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type none --K 2 --beta 4.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.7 --aug_feat_missing_rate 0.0
 

# ogbn-arxiv
python train.py --dataset ogbn-arxiv --teacher GCN --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 2048 --dropout_t 0.3 --dropout_s 0.1 --lamb 0.1 --tau 0.1 --learning_rate 0.005 --weight_decay 0.0 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type batch --K 2 --beta 0.5 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.0 --aug_feat_missing_rate 0.0
 
python train.py --dataset ogbn-arxiv --teacher SAGE --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 2048 --dropout_t 0.5 --dropout_s 0.3 --lamb 0.7 --tau 0.1 --learning_rate 0.005 --weight_decay 0.0 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type batch --K 2 --beta 2.0 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.0 --aug_feat_missing_rate 0.0
 
python train.py --dataset ogbn-arxiv --teacher GAT --student AdaGMLP --exp_setting 0 --split_rate 0.0 --num_layers 3 --hidden_dim 2048 --dropout_t 0.5 --dropout_s 0.1 --lamb 0.2 --tau 0.1 --learning_rate 0.005 --weight_decay 0.0 --max_epoch 500 --seed 52 --save_mode 1 --ablation_mode 2 --norm_type batch --K 2 --beta 0.5 --selective 1 --feat_missing_rate 0.0 --lamb_a 0.0 --aug_feat_missing_rate 0.0
 


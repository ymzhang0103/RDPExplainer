# RDPExplainer
Code for the paper "Towards Explaining Graph Neural Networks via Preserving Prediction Ranking and Structural Dependency"

## Overview
The project contains the following folders and files.
- datasets:  the datasets for the node classification is contained in this fold. For Graph-SST5 and Graph-Twitter used in the paper, they can be download from the Baidu Cloud. Link: https://pan.baidu.com/s/1KCa7VcuBE5A8oyzX34ohug Password: 4c2e.
- checkpoint: trained GNN models to be explained.
- codes
	- load_dataset.py: contains functions for loading datasets.
	- ExplainerNC.py: the explainer for the node classification task.
	- ExplainerGC.py: the explainer for the graph classification task.
	- GNNConfig.py:  parameter configuration of the GNN model to be explained.
	- ExplainConfig.py: parameter configuration of the GNN explainer.
	- metrics.py:  metrics for the evaluation.
- train_gnns.py: the code for the GNN model to be explained.
- main_NCExplain.py: the code for the explainer in the node classification task.
- main_GCExplain.py: the code for the explainer in the graph classification task.

## Prerequisites
python   3.7
torch >= 1.6.0

## To run
Run train_gnns.py to train the GNN model. Change parameter **dataset** per demand.

Run main_NCExplain.py to train and test the explainer in the node classification task. Change **dataset** per demand.

Run main_GCExplain.py to train and test the explainer in the graph classification. task Change **dataset** per demand.

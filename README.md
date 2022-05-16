# RDPExplainer
Code for the paper "Towards Explaining Graph Neural Networks via Preserving Prediction Ranking and Structural Dependency"

## Overview
The package contains the following folders and files:
- datasets: including the datasets used in the paper.
- checkpoint: trained GNN models used in the paper.
- codes
  - load_dataset.py: contains function needed to read datasets.
	- ExplainerNC.py: the explainer of node classification.
	- ExplainerGC.py: the explainer of graph classification.
	- GNNConfig.py: configure parameter of GNN model.
	- ExplainConfig.py: configure parameter of GNN model.
	- metrics.py: contains functions related to evaluation.
- train_gnns.py: main code to train  and test GNN model.
- main_NCExplain.py: main code to train  and test explainer of node classification.
- main_GCExplain.py: main code to train  and test explainer of graph classification.

## Prerequisites
python   3.7
torch >= 1.6.0

## To run
Simply run the train_gnns.py to train GNN model . Change dataset per demand.

train_gnns.py --dataset BA_shapes

Simply run the main_NCExplain.py to train and  test explainer of node classification. Change dataset per demand.

main_NCExplain.py --dataset BA_shapes

Simply run the main_GCExplain.py to train and  test explainer of node classification. Change dataset per demand.

main_GCExplain.py --dataset BA_shapes

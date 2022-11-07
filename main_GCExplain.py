#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
sys.path.append('./codes/forgraph/')

import numpy as np
import torch 
import torch.optim
import time
from codes.ExplainConfig import args
from codes.utils import *
from codes.metrics import XCollector, MaskoutMetric, rho_ndcg
from codes.ExplainerGC import ExplainerGC
import matplotlib
matplotlib.use('Agg')
from codes.GNNConfig import model_args
from codes.GNNmodels import GnnNets
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx



def main(iteration_num=10):
    def test(iteration, indices, model, explainer, topk_arr, plot_flag=False):
        global preds
        global reals
        preds = []
        reals = []
        ndcgs =[]
        plotutils = PlotUtils(dataset_name=args.dataset)
        metric = MaskoutMetric(model, args)
        allnode_related_preds_dict = dict()
        allnode_mask_dict = dict()
        for graphid in indices:
            #data = dataset[graphid]
            sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
            sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
            data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
            logits, prob, sub_embs = model(data)
            label = dataset.data.y[graphid]
            sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
            explainer.eval()
            masked_pred = explainer((sub_features, sub_embs, sub_adj, label, 1.0))
            origin_pred = prob.squeeze()
            ndcg_onegraph, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
            ndcgs.append(ndcg_onegraph)

            mask = explainer.masked_adj
            pred_mask, related_preds_dict = metric.metric_del_edges_GC(topk_arr, sub_features, mask, sub_edge_index, origin_pred, masked_pred, label)

            allnode_related_preds_dict[graphid] = related_preds_dict
            allnode_mask_dict[graphid] = pred_mask
            
            if plot_flag:
                edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]
                edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
                important_nodelist = []
                important_edgelist = []
                for idx in edges_idx_desc:
                    if len(important_edgelist)<12:
                        if (sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()) not in important_edgelist:
                            important_nodelist.append(sub_edge_index[0][idx].item())
                            important_nodelist.append(sub_edge_index[1][idx].item())
                            important_edgelist.append((sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()))
                            important_edgelist.append((sub_edge_index[1][idx].item(), sub_edge_index[0][idx].item()))
                important_nodelist = list(set(important_nodelist))
                ori_graph = to_networkx(data, to_undirected=True)
                if hasattr(dataset, 'supplement'):
                    words = dataset.supplement['sentence_tokens'][str(graphid)]
                    plotutils.plot(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, words=words,
                                figname=os.path.join(save_map + str(iteration), f"example_{graphid}.pdf"))
                else:
                    plotutils.plot(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x,
                                figname=os.path.join(save_map + str(iteration), f"example_{graphid}.pdf"))

        auc = 0
        ndcg = np.mean(ndcgs)
        return auc, ndcg, allnode_related_preds_dict, allnode_mask_dict


    def train(iteration):
        tik = time.time()
        epochs = args.eepochs
        #t0 = args.coff_t0
        #t1 = args.coff_te
        explainer.train()

        best_decline = 0
        f = open(save_map + str(iteration) + "/" + "LOG_" + args.model_filename + "_BEST_train_log.txt", "w")
        for epoch in range(epochs):
            loss = 0
            neuralsort_loss = 0
            value_loss =0
            size_loss=0
            #beta = float(t0 * np.power(t1 / t0, epoch /epochs))
            beta = float(1.0*np.power(0.05,epoch/epochs))
            for graphid in train_instances:
                sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
                sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                logits, prob, sub_embs = model(data)
                label = dataset.data.y[graphid]
                sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
                pred = explainer((sub_features, sub_embs, sub_adj, label, beta), training=True)
                l, sl, vl, ll = explainer.loss(pred, prob.squeeze(), args.loss_flag)
                loss = loss+l
                neuralsort_loss = neuralsort_loss + sl
                value_loss  = value_loss + vl
                size_loss = size_loss + ll
            train_variables = []
            for name, para in explainer.named_parameters():
                if "elayers" in name:
                    train_variables.append(para)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #eval
            auc, ndcg, allnode_related_preds_dict, allnode_mask_dict = test(iteration, eval_indices, model, explainer, [10])

            x_collector = XCollector()
            for graphid in eval_indices:
                related_preds = allnode_related_preds_dict[graphid][10]
                mask = allnode_mask_dict[graphid]
                x_collector.collect_data(mask, related_preds, label=0)
            #print("mask", mask)
            decline = x_collector.fidelity_complete-x_collector.simula_complete
            fidelity = x_collector.fidelity_complete
            simula = x_collector.simula_complete

            if epoch == 0:
                best_decline = decline
            
            if decline >= best_decline:
                print("saving best decline model...")
                f.write("saving best decline model...\n")
                best_decline = decline
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt')
            print("epoch", epoch, "loss", loss, "sort_loss",neuralsort_loss, "value_loss", value_loss, "size_loss", size_loss, "auc", auc, "ndcg", ndcg, "2: ", fidelity, "1: ", simula,"decline",decline)
            f.write("epoch,{}".format(epoch) + ",loss,{}".format(loss)  + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc) + ",2,{}".format(fidelity) + ",1,{}".format(simula)  + ",decline,{}".format(decline) + ",sort_loss,{}".format(neuralsort_loss) + ",value_loss,{}".format(value_loss) + ",size_loss,{}".format(size_loss) + "\n")
            
        tok = time.time()
        f.write("train time,{}".format(tok - tik) + "\n")
        f.close()





    args.elr = 0.01
    args.coff_size = 0.01
    args.batch_size = 128
    args.random_split_flag = True
    args.data_split_ratio =  [0.8, 0.1, 0.1]  #None
    args.seed = 1
    args.eepochs = 100
    args.latent_dim = model_args.latent_dim
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.topk_arr = list(range(10))+list(range(10,101,5))
    #args.dataset = "Graph-Twitter"
    #save_map = "LISA_TEST_LOGS/GRAPH_TWITTER_OUR_001sizeloss_10diffloss_001elr_epoch100_neuralsortandvalue_concrete_valueL1Norm/"
    args.dataset = "Graph-SST5"
    save_map = "LISA_TEST_LOGS/GRAPH_SST5_OUR_001sizeloss_001elr_epoch100_neuralsortandvalue_concrete_valueL1Norm/"
    if not os.path.exists(save_map):
            os.makedirs(save_map)
    args.maskmodel="GAT"     #GAT,  NOGAT
    args.loss_flag = "plandvalue"     #plandvalue,  pl, value

    GNNmodel_ckpt_path = osp.join('checkpoint', args.dataset, 'gcn_best.pth') 

    #train
    args.model_filename = args.dataset
    #test
    test_flag = True
    testmodel_filename = args.dataset + '_BEST'
    args.plot_flag=False

    f_mean = open(save_map + "LOG_" + args.model_filename + "_BEST_multiIter_supplement_selectk_add_minus_abs_removeedges.txt", "w")
    auc_all = []
    ndcg_all = []
    simula_arr = []
    simula_origin_arr = []
    simula_complete_arr = []
    fidelity_arr = []
    fidelity_origin_arr = []
    fidelity_complete_arr = []
    fidelityminus_arr = []
    fidelityminus_origin_arr = []
    fidelityminus_complete_arr = []
    finalfidelity_complete_arr = []
    sparsity_edges_arr = []
    fidelity_nodes_arr = []
    fidelity_origin_nodes_arr = []
    fidelity_complete_nodes_arr = []
    fidelityminus_nodes_arr = []
    fidelityminus_origin_nodes_arr = []
    fidelityminus_complete_nodes_arr = []
    finalfidelity_complete_nodes_arr = []
    sparsity_nodes_arr = []
    for iteration in range(iteration_num):
        print("Starting iteration: {}".format(iteration))
        if not os.path.exists(save_map+str(iteration)):
            os.makedirs(save_map+str(iteration))

        #load data
        dataset = get_dataset(args.dataset_root, args.dataset)
        dataset.data.x = dataset.data.x.float()
        dataset.data.y = dataset.data.y.squeeze().long()
        dataloader_params = {'batch_size': args.batch_size,
                                'random_split_flag': args.random_split_flag,
                                'data_split_ratio': args.data_split_ratio,
                                'seed': args.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices
        train_instances = loader['train'].dataset.indices
        eval_indices = loader['eval'].dataset.indices

        #print("train_instances", train_instances)
        #print("test_indices", test_indices)
        #print("eval_indices", eval_indices)

        model = GnnNets(input_dim=dataset.num_node_features,  output_dim=dataset.num_classes, model_args=model_args)
        ckpt = torch.load(GNNmodel_ckpt_path)
        model.load_state_dict(ckpt['net'])
        model.eval()
        model.to(args.device)

        explainer = ExplainerGC(model=model, args=args)
        explainer.to(args.device)
        optimizer = torch.optim.Adam(explainer.elayers.parameters(), lr=args.elr)
        
        # Training
        if test_flag:
            f = open(save_map + str(iteration) + "/LOG_"+testmodel_filename+"_test_log_supplement_abs_removeedges.txt", "w")
            explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + testmodel_filename+".pt") )
        else:
            f = open(save_map + str(iteration) + "/" + "LOG_" + args.model_filename + "_BEST.txt", "w")
            train(iteration)
            explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt'))

        tik = time.time()
        explainer.eval()
        auc, ndcg, allnode_related_preds_dict, allnode_mask_dict = test(iteration, test_indices, model, explainer, args.topk_arr, plot_flag=args.plot_flag)
        auc_all.append(auc)
        ndcg_all.append(ndcg)

        one_simula_arr = []
        one_simula_origin_arr = []
        one_simula_complete_arr = []
        one_fidelity_arr = []
        one_fidelity_origin_arr = []
        one_fidelity_complete_arr = []
        one_fidelityminus_arr = []
        one_fidelityminus_origin_arr = []
        one_fidelityminus_complete_arr = []
        one_finalfidelity_complete_arr = []
        one_sparsity_edges_arr = []
        one_fidelity_nodes_arr = []
        one_fidelity_origin_nodes_arr = []
        one_fidelity_complete_nodes_arr = []
        one_fidelityminus_nodes_arr = []
        one_fidelityminus_origin_nodes_arr = []
        one_fidelityminus_complete_nodes_arr = []
        one_finalfidelity_complete_nodes_arr = []
        one_sparsity_nodes_arr = []
        for top_k in args.topk_arr:
            print("top_k: ", top_k)
            x_collector = XCollector()
            for graphid in test_indices:
                related_preds = allnode_related_preds_dict[graphid][top_k]
                mask = allnode_mask_dict[graphid]
                x_collector.collect_data(mask, related_preds, label=0)
                f.write("graphid,{}\n".format(graphid))
                f.write("mask,{}\n".format(mask))
                f.write("related_preds,{}\n".format(related_preds))

            one_simula_arr.append(round(x_collector.simula, 4))
            one_simula_origin_arr.append(round(x_collector.simula_origin, 4))
            one_simula_complete_arr.append(round(x_collector.simula_complete, 4))
            one_fidelity_arr.append(round(x_collector.fidelity, 4))
            one_fidelity_origin_arr.append(round(x_collector.fidelity_origin, 4))
            one_fidelity_complete_arr.append(round(x_collector.fidelity_complete, 4))
            one_fidelityminus_arr.append(round(x_collector.fidelityminus, 4))
            one_fidelityminus_origin_arr.append(round(x_collector.fidelityminus_origin, 4))
            one_fidelityminus_complete_arr.append(round(x_collector.fidelityminus_complete, 4))
            one_finalfidelity_complete_arr.append(round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
            one_sparsity_edges_arr.append(round(x_collector.sparsity_edges, 4))
            one_fidelity_nodes_arr.append(round(x_collector.fidelity_nodes, 4))
            one_fidelity_origin_nodes_arr.append(round(x_collector.fidelity_origin_nodes, 4))
            one_fidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes, 4))
            one_fidelityminus_nodes_arr.append(round(x_collector.fidelityminus_nodes, 4))
            one_fidelityminus_origin_nodes_arr.append(round(x_collector.fidelityminus_origin_nodes, 4))
            one_fidelityminus_complete_nodes_arr.append(round(x_collector.fidelityminus_complete_nodes, 4))
            one_finalfidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes - x_collector.fidelityminus_complete_nodes, 4))
            one_sparsity_nodes_arr.append(round(x_collector.sparsity_nodes, 4))

        print("one_auc", auc)
        print("one_ndcg", ndcg)
        print("one_simula_arr =", one_simula_arr)
        print("one_simula_origin_arr =", one_simula_origin_arr)
        print("one_simula_complete_arr =", one_simula_complete_arr)
        print("one_fidelity_arr =", one_fidelity_arr)
        print("one_fidelity_origin_arr =", one_fidelity_origin_arr)
        print("one_fidelity_complete_arr =", one_fidelity_complete_arr)
        print("one_fidelityminus_arr=", one_fidelityminus_arr)
        print("one_fidelityminus_origin_arr=", one_fidelityminus_origin_arr)
        print("one_fidelityminus_complete_arr=", one_fidelityminus_complete_arr)
        print("one_finalfidelity_complete_arr=", one_finalfidelity_complete_arr)
        print("one_sparsity_edges_arr =", one_sparsity_edges_arr)
        print("one_fidelity_nodes_arr =", one_fidelity_nodes_arr)
        print("one_fidelity_origin_nodes_arr =", one_fidelity_origin_nodes_arr)
        print("one_fidelity_complete_nodes_arr =", one_fidelity_complete_nodes_arr)
        print("one_fidelityminus_nodes_arr=", one_fidelityminus_nodes_arr)
        print("one_fidelityminus_origin_nodes_arr=", one_fidelityminus_origin_nodes_arr)
        print("one_fidelityminus_complete_nodes_arr=", one_fidelityminus_complete_nodes_arr)
        print("one_finalfidelity_complete_nodes_arr=", one_finalfidelity_complete_nodes_arr)
        print("one_sparsity_nodes_arr =", one_sparsity_nodes_arr)

        tok = time.time()
        f.write("our_auc={}".format(auc) + "\n")
        f.write("our_ndcg={}".format(ndcg) + "\n")
        f.write("one_simula={}".format(one_simula_arr) + "\n")
        f.write("one_simula_orign={}".format(one_simula_origin_arr) + "\n")
        f.write("one_simula_complete={}".format(one_simula_complete_arr) + "\n")
        f.write("one_fidelity={}".format(one_fidelity_arr) + "\n")
        f.write("one_fidelity_orign={}".format(one_fidelity_origin_arr) + "\n")
        f.write("one_fidelity_complete={}".format(one_fidelity_complete_arr) + "\n")
        f.write("one_fidelityminus={}".format(one_fidelityminus_arr)+"\n")
        f.write("one_fidelityminus_origin={}".format(one_fidelityminus_origin_arr)+"\n")
        f.write("one_fidelityminus_complete={}".format(one_fidelityminus_complete_arr)+"\n")
        f.write("one_finalfidelity_complete={}".format(one_finalfidelity_complete_arr)+"\n")
        f.write("one_sparsity_edges={}".format(one_sparsity_edges_arr) + "\n")
        f.write("one_fidelity_nodes={}".format(one_fidelity_nodes_arr) + "\n")
        f.write("one_fidelity_origin_nodes={}".format(one_fidelity_origin_nodes_arr) + "\n")
        f.write("one_fidelity_complete_nodes={}".format(one_fidelity_complete_nodes_arr) + "\n")
        f.write("one_fidelityminus_nodes={}".format(one_fidelityminus_nodes_arr)+"\n")
        f.write("one_fidelityminus_origin_nodes={}".format(one_fidelityminus_origin_nodes_arr)+"\n")
        f.write("one_finalfidelity_complete_nodes={}".format(one_finalfidelity_complete_nodes_arr)+"\n")
        f.write("one_sparsity_nodes={}".format(one_sparsity_nodes_arr) + "\n")
        f.write("test time,{}".format(tok-tik))
        f.close()
        
        simula_arr.append(one_simula_arr)
        simula_origin_arr.append(one_simula_origin_arr)
        simula_complete_arr.append(one_simula_complete_arr)
        fidelity_arr.append(one_fidelity_arr)
        fidelity_origin_arr.append(one_fidelity_origin_arr)
        fidelity_complete_arr.append(one_fidelity_complete_arr)
        fidelityminus_arr.append(one_fidelityminus_arr)
        fidelityminus_origin_arr.append(one_fidelityminus_origin_arr)
        fidelityminus_complete_arr.append(one_fidelityminus_complete_arr)
        finalfidelity_complete_arr.append(one_finalfidelity_complete_arr)
        sparsity_edges_arr.append(one_sparsity_edges_arr)
        fidelity_nodes_arr.append(one_fidelity_nodes_arr)
        fidelity_origin_nodes_arr.append(one_fidelity_origin_nodes_arr)
        fidelity_complete_nodes_arr.append(one_fidelity_complete_nodes_arr)
        fidelityminus_nodes_arr.append(one_fidelityminus_nodes_arr)
        fidelityminus_origin_nodes_arr.append(one_fidelityminus_origin_nodes_arr)
        fidelityminus_complete_nodes_arr.append(one_fidelityminus_complete_nodes_arr)
        finalfidelity_complete_nodes_arr.append(one_finalfidelity_complete_nodes_arr)
        sparsity_nodes_arr.append(one_sparsity_nodes_arr)

    print("args.dataset", args.dataset)
    print("our_auc_all = ", auc_all)
    print("our_ndcg_all = ", ndcg_all)
    print("our_simula_arr =", simula_arr)
    print("our_simula_origin_arr =", simula_origin_arr)
    print("our_simula_complete_arr =", simula_complete_arr)
    print("our_fidelity_arr =", fidelity_arr)
    print("our_fidelity_origin_arr =", fidelity_origin_arr)
    print("our_fidelity_complete_arr =", fidelity_complete_arr)
    print("our_fidelityminus_arr=", fidelityminus_arr)
    print("our_fidelityminus_origin_arr=", fidelityminus_origin_arr)
    print("our_fidelityminus_complete_arr=", fidelityminus_complete_arr)
    print("our_finalfidelity_complete_arr", finalfidelity_complete_arr)
    print("our_sparsity_edges_arr =", sparsity_edges_arr)
    print("our_fidelity_nodes_arr =", fidelity_nodes_arr)
    print("our_fidelity_origin_nodes_arr =", fidelity_origin_nodes_arr)
    print("our_fidelity_complete_nodes_arr =", fidelity_complete_nodes_arr)
    print("our_fidelityminus_nodes_arr=", fidelityminus_nodes_arr)
    print("our_fidelityminus_origin_nodes_arr=", fidelityminus_origin_nodes_arr)
    print("our_fidelityminus_complete_nodes_arr=", fidelityminus_complete_nodes_arr)
    print("our_finalfidelity_complete_nodes_arr", finalfidelity_complete_nodes_arr)
    print("our_sparsity_nodes_arr =", sparsity_nodes_arr)

    f_mean.write("our_auc_all = {}".format(auc_all)+ "\n")
    f_mean.write("our_ndcg_all = {}".format(ndcg_all)+ "\n")
    f_mean.write("our_simula_arr={}".format(simula_arr) + "\n")
    f_mean.write("our_simula_origin_arr={}".format(simula_origin_arr) + "\n")
    f_mean.write("our_simula_complete_arr={}".format(simula_complete_arr) + "\n")
    f_mean.write("our_fidelity_arr={}".format(fidelity_arr) + "\n")
    f_mean.write("our_fidelity_origin_arr={}".format(fidelity_origin_arr) + "\n")
    f_mean.write("our_fidelity_complete_arr={}".format(fidelity_complete_arr) + "\n")
    f_mean.write("our_fidelityminus_arr = {}".format(fidelityminus_arr)+"\n")
    f_mean.write("our_fidelityminus_origin_arr = {}".format(fidelityminus_origin_arr)+"\n")
    f_mean.write("our_fidelityminus_complete_arr = {}".format(fidelityminus_complete_arr)+"\n")
    f_mean.write("our_finalfidelity_complete_arr = {}".format(finalfidelity_complete_arr)+"\n")
    f_mean.write("our_sparsity_edges_arr={}".format(sparsity_edges_arr) + "\n")
    f_mean.write("our_fidelity_nodes_arr={}".format(fidelity_nodes_arr) + "\n")
    f_mean.write("our_fidelity_origin_nodes_arr={}".format(fidelity_origin_nodes_arr) + "\n")
    f_mean.write("our_fidelity_complete_nodes_arr={}".format(fidelity_complete_nodes_arr) + "\n")
    f_mean.write("our_fidelityminus_nodes_arr = {}".format(fidelityminus_nodes_arr)+"\n")
    f_mean.write("our_fidelityminus_origin_nodes_arr = {}".format(fidelityminus_origin_nodes_arr)+"\n")
    f_mean.write("our_fidelityminus_complete_nodes_arr = {}".format(fidelityminus_complete_nodes_arr)+"\n")
    f_mean.write("our_finalfidelity_complete_nodes_arr = {}".format(finalfidelity_complete_nodes_arr)+"\n")
    f_mean.write("our_sparsity_nodes_arr={}".format(sparsity_nodes_arr) + "\n")

    simula_mean = np.average(np.array(simula_arr), axis=0)
    simula_origin_mean = np.average(np.array(simula_origin_arr), axis=0)
    simula_complete_mean = np.average(np.array(simula_complete_arr),axis=0)
    fidelity_mean = np.average(np.array(fidelity_arr),axis=0)
    fidelity_origin_mean = np.average(np.array(fidelity_origin_arr),axis=0)
    fidelity_complete_mean = np.average(np.array(fidelity_complete_arr),axis=0)
    fidelityminus_mean = np.average(np.array(fidelityminus_arr),axis=0)
    fidelityminus_origin_mean = np.average(np.array(fidelityminus_origin_arr),axis=0)
    fidelityminus_complete_mean = np.average(np.array(fidelityminus_complete_arr),axis=0)
    finalfidelity_complete_mean = np.average(np.array(finalfidelity_complete_arr), axis=0)
    sparsity_edges_mean = np.average(np.array(sparsity_edges_arr),axis=0)
    fidelity_nodes_mean = np.average(np.array(fidelity_nodes_arr),axis=0)
    fidelity_origin_nodes_mean = np.average(np.array(fidelity_origin_nodes_arr),axis=0)
    fidelity_complete_nodes_mean = np.average(np.array(fidelity_complete_nodes_arr),axis=0)
    fidelityminus_nodes_mean = np.average(np.array(fidelityminus_nodes_arr),axis=0)
    fidelityminus_origin_nodes_mean = np.average(np.array(fidelityminus_origin_nodes_arr),axis=0)
    fidelityminus_complete_nodes_mean = np.average(np.array(fidelityminus_complete_nodes_arr),axis=0)
    finalfidelity_complete_nodes_mean = np.average(np.array(finalfidelity_complete_nodes_arr), axis=0)
    sparsity_nodes_mean = np.average(np.array(sparsity_nodes_arr),axis=0)

    print("our_auc_mean =", np.mean(auc_all))
    print("our_ndcg_mean =", np.mean(ndcg_all))
    print("our_simula_mean =", list(simula_mean))
    print("our_simula_origin_mean =", list(simula_origin_mean))
    print("our_simula_complete_mean =", list(simula_complete_mean))
    print("our_fidelity_mean = ", list(fidelity_mean))
    print("our_fidelity_origin_mean =", list(fidelity_origin_mean))
    print("our_fidelity_complete_mean =", list(fidelity_complete_mean))
    print("our_fidelityminus_mean =", list(fidelityminus_mean))
    print("our_fidelityminus_origin_mean =", list(fidelityminus_origin_mean))
    print("our_fidelityminus_complete_mean =", list(fidelityminus_complete_mean))
    print("our_finalfidelity_complete_mean =", list(finalfidelity_complete_mean))
    print("our_sparsity_edges_mean =", list(sparsity_edges_mean))
    print("our_fidelity_nodes_mean =", list(fidelity_nodes_mean))
    print("our_fidelity_origin_nodes_mean =", list(fidelity_origin_nodes_mean))
    print("our_fidelity_complete_nodes_mean =", list(fidelity_complete_nodes_mean))
    print("our_fidelityminus_nodes_mean =", list(fidelityminus_nodes_mean))
    print("our_fidelityminus_origin_nodes_mean =", list(fidelityminus_origin_nodes_mean))
    print("our_fidelityminus_complete_nodes_mean =", list(fidelityminus_complete_nodes_mean))
    print("our_finalfidelity_complete_nodes_mean =", list(finalfidelity_complete_nodes_mean))
    print("our_sparsity_nodes_mean =", list(sparsity_nodes_mean))

    f_mean.write("our_auc_mean = {}".format(np.mean(auc_all))+ "\n")
    f_mean.write("our_ndcg_mean = {}".format(np.mean(ndcg_all))+ "\n")
    f_mean.write("our_simula_mean = {}".format(list(simula_mean))+ "\n")
    f_mean.write("our_simula_origin_mean = {}".format(list(simula_origin_mean))+ "\n")
    f_mean.write("our_simula_complete_mean = {}".format(list(simula_complete_mean))+ "\n")
    f_mean.write("our_fidelity_mean = {}".format(list(fidelity_mean))+ "\n")
    f_mean.write("our_fidelity_origin_mean = {}".format(list(fidelity_origin_mean))+ "\n")
    f_mean.write("our_fidelity_complete_mean = {}".format(list(fidelity_complete_mean))+ "\n")
    f_mean.write("our_fidelityminus_mean = {}".format(list(fidelityminus_mean))+"\n")
    f_mean.write("our_fidelityminus_origin_mean = {}".format(list(fidelityminus_origin_mean))+"\n")
    f_mean.write("our_fidelityminus_complete_mean = {}".format(list(fidelityminus_complete_mean))+"\n")
    f_mean.write("our_finalfidelity_complete_mean = {}".format(list(finalfidelity_complete_mean))+"\n")
    f_mean.write("our_sparsity_edges_mean = {}".format(list(sparsity_edges_mean))+ "\n")
    f_mean.write("our_fidelity_nodes_mean = {}".format(list(fidelity_nodes_mean))+ "\n")
    f_mean.write("our_fidelity_origin_nodes_mean = {}".format(list(fidelity_origin_nodes_mean))+ "\n")
    f_mean.write("our_fidelity_complete_nodes_mean = {}".format(list(fidelity_complete_nodes_mean))+ "\n")
    f_mean.write("our_fidelityminus_nodes_mean = {}".format(list(fidelityminus_nodes_mean))+"\n")
    f_mean.write("our_fidelityminus_origin_nodes_mean = {}".format(list(fidelityminus_origin_nodes_mean))+"\n")
    f_mean.write("our_fidelityminus_complete_nodes_mean = {}".format(list(fidelityminus_complete_nodes_mean))+"\n")
    f_mean.write("our_finalfidelity_complete_nodes_mean = {}".format(list(finalfidelity_complete_nodes_mean))+"\n")
    f_mean.write("our_sparsity_nodes_mean = {}".format(list(sparsity_nodes_mean))+ "\n")
    f_mean.close()



def test_onegraph(explain_graph_arr, explainModel_ckpt_path):
    starttime = time.time()
    top_k=15
    GNNmodel_ckpt_path = osp.join('checkpoint', args.dataset, 'gcn_best.pth') 
    #load data
    dataset = get_dataset(args.dataset_root, args.dataset)
    #load model
    model = GnnNets(input_dim=dataset.num_node_features,  output_dim=dataset.num_classes, model_args=model_args)
    ckpt = torch.load(GNNmodel_ckpt_path)
    model.load_state_dict(ckpt['net'])
    model.eval()
    model.to(args.device)

    explainer = ExplainerGCValueL1Norm(model=model, args=args)
    explainer.to(args.device)
    explainer.load_state_dict(torch.load(explainModel_ckpt_path) )
    explainer.eval()

    for explain_graph in explain_graph_arr:
        sub_features = dataset.data.x[dataset.slices['x'][explain_graph].item():dataset.slices['x'][explain_graph+1].item(), :]
        sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][explain_graph].item():dataset.slices['edge_index'][explain_graph+1].item()]
        sub_data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
        logits, prob, sub_embs = model(sub_data)
        label = dataset.data.y[explain_graph]
        sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
        explainer.eval()
        masked_pred = explainer((sub_features, sub_embs, sub_adj, label, 1.0))
        mask = explainer.masked_adj
    print("get mask time: ",time.time()-starttime)

    '''origin_pred = prob.squeeze()
    ndcg_onegraph, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
    metric = MaskoutMetric(model, args)
    pred_mask, related_preds_dict = metric.metric_del_edges_GC([top_k], sub_features, mask, sub_edge_index, origin_pred, masked_pred, label)

    x_collector = XCollector()
    x_collector.collect_data(pred_mask, related_preds_dict[top_k], label=0)

    print("explain_graph,{}\n".format(explain_graph) + "ndcg_onegraph={}".format(ndcg_onegraph) + "\n")
    print("fidelity_complete=", round(x_collector.fidelity_complete, 4))
    print("fidelityminus_complete=", round(x_collector.fidelityminus_complete, 4))
    print("finalfidelity_complete=", round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
    print("test time: ", time.time()-starttime)'''




if __name__ == '__main__':
    #main()
    #args.dataset="Graph-Twitter"
    #explainModel_ckpt_path ="LISA_TEST_LOGS/GRAPH_TWITTER_OUR_001sizeloss_10diffloss_001elr_epoch100_neuralsortandvalue_concrete_valueL1Norm/0/Graph-Twitter_BEST.pt"
    explain_graph_arr = [random.randint(301, 600) for p in range(0, 100)]
    args.dataset="Graph-SST5"
    explainModel_ckpt_path ="LISA_TEST_LOGS/GRAPH_SST5_OUR_001sizeloss_10diffloss_001elr_epoch100_neuralsortandvalue_concrete_valueL1Norm/0/Graph-SST5_BEST.pt"
    test_onegraph(explain_graph_arr, explainModel_ckpt_path)
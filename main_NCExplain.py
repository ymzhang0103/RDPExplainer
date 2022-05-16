#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append('./codes/fornode/')

import time
import numpy as np
import torch
import torch.optim
import pickle as pkl
from codes.ExplainConfig import args
from codes.utils import *
from codes.metrics import XCollector, MaskoutMetric, rho_ndcg
from codes.Extractor import Extractor
from codes.ExplainerNC import ExplainerNC
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from codes.GNNConfig import model_args
from codes.GNNmodels import GnnNets_NC
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def train(iteration):
    t0 = args.coff_t0
    t1 = args.coff_te
    epochs = args.eepochs
    model.eval()
    explainer.train()
    best_decline = 0
    f = open(save_map + str(iteration) + "/" + "LOG_" + args.model_filename + "_BEST_train_log.txt", "w")
    for epoch in range(epochs):
        loss = 0
        sort_loss = 0
        value_loss = 0
        size_loss = 0
        tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
        for node in trainnodes:
            newid = remap[node]
            nodeid = 0
            sub_output = sub_outputs[newid]
            old_pred_label = torch.argmax(sub_output, 1)

            sub_embed = sub_embeds[newid]
            pred = explainer((sub_features[newid], sub_adjs[newid], nodeid, sub_embed, sub_new_edge_indexs[newid], tmp), training=True)
            l, sl, vl, ll = explainer.loss(pred, sub_output[nodeid], args.loss_flag)
            loss = loss + l
            sort_loss = sort_loss + sl
            value_loss = value_loss + vl
            size_loss = size_loss + ll
            ndcg_onenode, r_mask, _, _ = rho_ndcg(sub_output[nodeid], pred, len(pred))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reals = []
        preds = []
        ndcgs =[]
        x_collector = XCollector()
        metric = MaskoutMetric(model, args)
        for node in valnodes:
            newid = remap[node]
            sub_adj, sub_feature, sub_embed, sub_label, sub_edge_label = sub_adjs[newid], sub_features[newid], sub_embeds[newid], sub_labels[newid], sub_edge_labels[newid]
            nodeid = 0
            origin_pred = sub_outputs[newid][nodeid]
            label = sub_label[nodeid].argmax(-1)
            origin_label = origin_pred.argmax(-1)

            explainer.eval()
            masked_pred = explainer((sub_feature, sub_adj, nodeid, sub_embed,  sub_new_edge_indexs[newid], 1.0))
            mask = explainer.masked_adj
            auc_onenode, real, pred = acc(sub_adj, sub_edge_label, mask)
            reals.extend(real)
            preds.extend(pred)
            ndcg_onenode, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
            ndcgs.append(ndcg_onenode)
            pred_mask, related_preds_dict = metric.metric_del_edges(nodeid, sub_feature, sub_adj, mask, origin_pred, masked_pred, label, [5])
            x_collector.collect_data(pred_mask, related_preds_dict[5], label=0)

        auc = roc_auc_score(reals, preds)
        ndcg = np.mean(ndcgs)
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

        print("epoch", epoch, "loss", loss, "sort_loss",sort_loss, "value_loss",value_loss,"size_loss",size_loss, "ndcg", ndcg, "auc", auc,  "best_decline",best_decline, "2: ",fidelity, "1: ",simula,"decline",decline)
        f.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc)+ ",2,{}".format(fidelity) + ",1,{}".format(simula)  + ",decline,{}".format(decline) + ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",size_loss,{}".format(size_loss)+ "\n")

    f.close()        
    torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_LAST.pt')


def acc(sub_adj, sub_edge_label, mask):
    mask = mask.cpu().detach().numpy()
    real = []
    pred = []
    sub_edge_label = sub_edge_label.todense()
    for r,c in list(zip(sub_adj.row,sub_adj.col)):
        d = sub_edge_label[r,c] + sub_edge_label[c,r]
        if d == 0:
            real.append(0)
        else:
            real.append(1)
        pred.append(mask[r][c]+mask[c][r])

    if len(np.unique(real))==1 or len(np.unique(pred))==1:
        return -1, real, pred
    return roc_auc_score(real, pred), real, pred


def convert_new_directed_edge_index(edge_index):
    edge_num = len(edge_index[0])
    for i in range(edge_num):
        raw_idx = torch.where(edge_index[1] == edge_index[0][i])[0] #head
        col_idx = torch.where(edge_index[0] == edge_index[1][i])[0] #tail
        con_idx = torch.tensor(list(set(col_idx.cpu().numpy())|set(raw_idx.cpu().numpy())))
        temp_edge = torch.cat((torch.tensor([i] * len(con_idx)).to(args.device).unsqueeze(0), con_idx.to(args.device).unsqueeze(0)))
        if i==0:
            new_edge_index = temp_edge
        else:
            new_edge_index = torch.cat((new_edge_index, temp_edge), dim=1)
    return new_edge_index



if args.dataset=="BA_shapes":
    args.coff_size =0.01
    args.elr = 0.03
elif args.dataset=="BA_community":
    args.coff_size =0.01
    args.elr = 0.01
args.eepochs = 100
args.budget = -1
args.coff_ent = 1.0
#args.concat = False
args.latent_dim = model_args.latent_dim
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
args.topk_arr = list(range(10))+list(range(10,101,5))

args.dataset="BA_shapes"
save_map = "LISA_TEST_LOGS/BA_SHAPES/"
#args.dataset="BA_community"
#save_map = "LISA_TEST_LOGS/BA_COMMUNITY/"
if not os.path.exists(save_map):
        os.makedirs(save_map)
args.maskmodel="GAT"     #GAT,  NOGAT
args.loss_flag = "plandvalue"     #plandvalue,  pl, value

filename = 'datasets/'+args.dataset+'/raw/' + args.dataset + '.pkl'
GNNmodel_ckpt_path = osp.join('checkpoint', args.dataset, 'gcn_best.pth')

#train
args.model_filename = args.dataset
#test
test_flag = False
testmodel_filename = args.dataset + '_BEST'
plot_flag = True


f_mean = open(save_map + "LOG_" + args.model_filename + "_BEST_multiIter.txt", "w")
auc_all = []
ndcg_all = []
fidelity_arr = []
fidelity_origin_arr = []
fidelity_complete_arr = []
fidelityminus_complete_arr = []
finalfidelity_complete_arr = []
sparsity_arr = []
fidelity_delnotimp_arr = []
fidelity_origin_delnotimp_arr = []
fidelity_complete_delnotimp_arr = []
sparsity_delnotimp_arr = []
simula_arr = []
simula_origin_arr = []
simula_complete_arr = []
decline_arr = []
fidelity_complete_nodes_arr = []
sparsity_nodes_arr = []
decline_nodes_arr = []
for iteration in range(1):
    print("Starting iteration: {}".format(iteration))
    if not os.path.exists(save_map+str(iteration)):
        os.makedirs(save_map+str(iteration))

    #load data
    with open(filename, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(fin)

    adj = csr_matrix(adj)
    support = preprocess_adj(adj)

    features_tensor = torch.tensor(features).type(torch.float32).to(args.device)
    edge_index = torch.LongTensor([*support[0]]).t().to(args.device)
    edge_data = torch.FloatTensor([*support[1]]).to(args.device)
    support_tensor = torch.sparse.FloatTensor(edge_index, edge_data, torch.Size([*support[2]])).type(torch.float32).to(args.device)

    all_label = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    label = np.where(all_label)[1]
 
    y = torch.from_numpy(label)
    data = Data(x=features_tensor, y=y, edge_index=edge_index)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)

    #load model
    model = GnnNets_NC(input_dim=features.shape[1], output_dim=y_train.shape[1], model_args=model_args)
    model.to_device()
    #ckpt = torch.load(GNNmodel_ckpt_path, map_location=torch.device('cpu'))
    ckpt = torch.load(GNNmodel_ckpt_path)
    model.load_state_dict(ckpt['net'])
    model.eval()
    logits, outputs, embeds = model(data.to(args.device))
    embeds = embeds.detach()

    explainer = ExplainerNC(model=model, args=args)
    explainer.to(args.device)

    hops = len(args.latent_dim)
    extractor = Extractor(adj, edge_index, features, edge_label_matrix, embeds, all_label, hops)

    allnodes = []
    if args.dataset == "BA_shapes":
        trainnodes = torch.where(torch.from_numpy(train_mask) * label != 0)[0].tolist()
        valnodes = torch.where(torch.from_numpy(val_mask) * label != 0)[0].tolist()
        testnodes = torch.where(torch.from_numpy(test_mask) * label != 0)[0].tolist()
    elif args.dataset == "BA_community":
        trainnodes = torch.where(((torch.from_numpy(train_mask) * label != 0) & (torch.from_numpy(train_mask) * label != 4)))[0].tolist()
        valnodes = torch.where(((torch.from_numpy(val_mask) * label != 0) & (torch.from_numpy(val_mask) * label != 4)))[0].tolist()
        testnodes = torch.where(((torch.from_numpy(test_mask) * label != 0) & (torch.from_numpy(test_mask) * label != 4)))[0].tolist()
    allnodes.extend(trainnodes)
    allnodes.extend(valnodes)
    allnodes.extend(testnodes)

    optimizer = torch.optim.Adam(explainer.elayers.parameters(), lr=args.elr)
    clip_value_min = -2.0
    clip_value_max = 2.0

    sub_support_tensors = []
    sub_label_tensors = []
    sub_features = []
    sub_embeds = []
    sub_adjs = []
    sub_edge_labels = []
    sub_labels = []
    sub_nodes = []
    sub_outputs = []
    sub_new_edge_indexs = []
    remap = {}
    for node in allnodes:
        sub_adj, sub_feature, sub_embed, sub_label, sub_edge_label_matrix, sub_node, sub_edge_idx  = extractor.subgraph(node)
        remap[node] = len(sub_adjs)
        sub_edge_index = torch.LongTensor([sub_adj.row, sub_adj.col]).to(args.device)
        sub_edge_data = torch.FloatTensor([sub_adj.data]).to(args.device)
        sub_label_tensor = torch.argmax(torch.Tensor(sub_label).type(torch.int32).to(args.device), -1)
        sub_feature_tensor = torch.Tensor(sub_feature).type(torch.float32).to(args.device)

        #convert edge to node and convert node to edge
        sub_new_edge_index = convert_new_directed_edge_index(sub_edge_index)

        sub_adjs.append(sub_adj)
        sub_features.append(sub_feature_tensor)
        sub_embeds.append(sub_embed)
        sub_labels.append(sub_label)
        sub_edge_labels.append(sub_edge_label_matrix)
        sub_label_tensors.append(sub_label_tensor)
        sub_nodes.append(sub_node)
        sub_new_edge_indexs.append(sub_new_edge_index)

        with torch.no_grad():
            data = Data(x=sub_feature_tensor, edge_index=sub_edge_index)
            _, sub_output, _ = model(data)
        sub_outputs.append(sub_output)

    if test_flag:
        f = open(save_map + str(iteration) + "/LOG_"+testmodel_filename+"_test_log.txt", "w")
        explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + testmodel_filename+".pt") )
    else:
        f = open(save_map + str(iteration) + "/" + "LOG_" + args.model_filename + "_BEST.txt", "w")
        tik = time.time()
        train(iteration)
        tok = time.time()
        f.write("train time,{}".format(tok - tik) + "\n")
        explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt'))

    #metrics
    tik = time.time()
    reals = []
    preds = []
    ndcgs =[]
    plotutils = PlotUtils(dataset_name=args.dataset)
    allnode_related_preds_dict = dict()
    allnode_mask_dict = dict()
    maskoutMetric = MaskoutMetric(model, args)
    for node in testnodes:
        newid = remap[node]
        sub_adj, sub_feature, sub_embed, sub_label, sub_edge_label = sub_adjs[newid], sub_features[newid], sub_embeds[newid], sub_labels[newid], sub_edge_labels[newid]
        nodeid = 0
        #use fullgraph prediction
        #origin_pred = outputs[node]
        #use subgraph prediction
        origin_pred = sub_outputs[newid][nodeid]

        label_onenode = sub_label[nodeid].argmax(-1)
        origin_label_onenode = origin_pred.argmax(-1)

        explainer.eval()
        masked_pred = explainer((sub_feature, sub_adj, nodeid, sub_embed,  sub_new_edge_indexs[newid], 1.0))
        mask = explainer.masked_adj

        auc_onenode, real, pred = acc(sub_adj, sub_edge_label, mask)
        reals.extend(real)
        preds.extend(pred)
        ndcg_onenode, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))

        ndcgs.append(ndcg_onenode)
        pred_mask, related_preds_dict = maskoutMetric.metric_del_edges(nodeid, sub_feature, sub_adj, mask, origin_pred, masked_pred, label_onenode, args.topk_arr)
        
        allnode_related_preds_dict[node] = related_preds_dict
        allnode_mask_dict[node] = pred_mask
        tok = time.time()
        f.write("node,{}".format(node) + ",auc_onenode,{}".format(auc_onenode) + ",ndcg_onenode,{}".format(ndcg_onenode) + ",time,{}".format(tok - tik) + "\n")
        
        # visualization
        if plot_flag:
            edge_mask = mask[sub_adj.row, sub_adj.col]
            edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
            sub_edge_index = torch.tensor([sub_adj.row, sub_adj.col], dtype=torch.int64)
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
            data = Data(x=sub_feature, edge_index=sub_edge_index, y =sub_label.argmax(-1) )
            ori_graph = to_networkx(data, to_undirected=True)
            plotutils.plot(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, y=sub_label.argmax(-1), node_idx=nodeid,
                        figname=os.path.join(save_map + str(iteration), f"node_{node}_new.png"))

    auc = roc_auc_score(reals, preds)
    ndcg = np.mean(ndcgs)
    auc_all.append(auc)
    ndcg_all.append(ndcg)
    tok = time.time()
    f.write("iteration,{}".format(iteration) + ", auc,{}".format(auc) + ", ndcg,{}".format(ndcg) + "\n")

    one_fidelity_arr = []
    one_fidelity_origin_arr = []
    one_fidelity_complete_arr = []
    one_fidelityminus_complete_arr = []
    one_finalfidelity_complete_arr = []
    one_sparsity_arr = []
    one_fidelity_delnotimp_arr = []
    one_fidelity_origin_delnotimp_arr = []
    one_fidelity_complete_delnotimp_arr = []
    one_sparsity_delnotimp_arr = []
    one_simula_arr = []
    one_simula_origin_arr = []
    one_simula_complete_arr = []
    one_decline_arr = []
    one_fidelity_complete_nodes_arr = []
    one_sparsity_nodes_arr = []
    one_decline_nodes_arr = []
    for top_k in args.topk_arr:
        print("top_k: ", top_k)
        x_collector = XCollector()
        for node in testnodes:
            related_preds = allnode_related_preds_dict[node][top_k]
            mask = allnode_mask_dict[node]
            x_collector.collect_data(mask, related_preds, label=0)
            f.write("node,{}\n".format(node))
            f.write("mask,{}\n".format(mask))
            f.write("related_preds,{}\n".format(related_preds))

        one_fidelity_arr.append(round(x_collector.fidelity, 4))
        one_fidelity_origin_arr.append(round(x_collector.fidelity_origin, 4))
        one_fidelity_complete_arr.append(round(x_collector.fidelity_complete, 4))
        one_fidelityminus_complete_arr.append(round(x_collector.fidelityminus_complete, 4))
        one_finalfidelity_complete_arr.append(round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
        one_sparsity_arr.append(round(x_collector.sparsity, 4))
        one_fidelity_delnotimp_arr.append(round(x_collector.fidelity_delnotimp, 4))
        one_fidelity_origin_delnotimp_arr.append(round(x_collector.fidelity_origin_delnotimp, 4))
        one_fidelity_complete_delnotimp_arr.append(round(x_collector.fidelity_complete_delnotimp, 4))
        one_sparsity_delnotimp_arr.append(round(x_collector.sparsity_delnotimp, 4))
        one_simula_arr.append(round(x_collector.simula, 4))
        one_simula_origin_arr.append(round(x_collector.simula_origin, 4))
        one_simula_complete_arr.append(round(x_collector.simula_complete, 4))
        one_decline_arr.append(round(x_collector.fidelity_complete-x_collector.simula_complete, 4))
        one_fidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes, 4))
        one_sparsity_nodes_arr.append(round(x_collector.sparsity_nodes, 4))
        one_decline_nodes_arr.append(round(x_collector.fidelity_complete_nodes-x_collector.simula_complete, 4))

    print("one_auc", auc)
    print("one_ndcg", ndcg)
    print("one_fidelity_arr =", one_fidelity_arr)
    print("one_fidelity_origin_arr =", one_fidelity_origin_arr)
    print("one_fidelity_complete_arr =", one_fidelity_complete_arr)
    print("one_fidelityminus_complete_arr=", one_fidelityminus_complete_arr)
    print("one_finalfidelity_complete_arr=", one_finalfidelity_complete_arr)
    print("one_sparsity_arr =", one_sparsity_arr)
    print("one_fidelity_delnotimp_arr =", one_fidelity_delnotimp_arr)
    print("one_fidelity_origin_delnotimp_arr =", one_fidelity_origin_delnotimp_arr)
    print("one_fidelity_complete_delnotimp_arr =", one_fidelity_complete_delnotimp_arr)
    print("one_sparsity_delnotimp_arr =", one_sparsity_delnotimp_arr)
    print("one_simula_arr =", one_simula_arr)
    print("one_simula_origin_arr =", one_simula_origin_arr)
    print("one_simula_complete_arr =", one_simula_complete_arr)
    print("one_decline_arr =", one_decline_arr)
    print("one_fidelity_complete_nodes_arr =", one_fidelity_complete_nodes_arr)
    print("one_sparsity_nodes_arr =", one_sparsity_nodes_arr)
    print("one_decline_nodes_arr =", one_decline_nodes_arr)

    tok = time.time()
    f.write("our_auc={}".format(auc) + "\n")
    f.write("our_ndcg={}".format(ndcg) + "\n")
    f.write("our_fidelity={}".format(one_fidelity_arr) + "\n")
    f.write("our_fidelity_orign={}".format(one_fidelity_origin_arr) + "\n")
    f.write("our_fidelity_complete={}".format(one_fidelity_complete_arr) + "\n")
    f.write("one_fidelityminus_complete={}".format(one_fidelityminus_complete_arr)+"\n")
    f.write("one_finalfidelity_complete={}".format(one_finalfidelity_complete_arr)+"\n")
    f.write("our_sparsity={}".format(one_sparsity_arr) + "\n")
    f.write("our_fidelity_delnotimp={}".format(one_fidelity_delnotimp_arr) + "\n")
    f.write("our_fidelity_orign_delnotimp={}".format(one_fidelity_origin_delnotimp_arr) + "\n")
    f.write("our_fidelity_complete_delnotimp={}".format(one_fidelity_complete_delnotimp_arr) + "\n")
    f.write("our_sparsity_delnotimp={}".format(one_sparsity_delnotimp_arr) + "\n")
    f.write("our_simula={}".format(one_simula_arr) + "\n")
    f.write("our_simula_orign={}".format(one_simula_origin_arr) + "\n")
    f.write("our_simula_complete={}".format(one_simula_complete_arr) + "\n")
    f.write("our_decline={}".format(one_decline_arr) + "\n")
    f.write("our_fidelity_complete_nodes={}".format(one_fidelity_complete_nodes_arr) + "\n")
    f.write("our_sparsity_nodes={}".format(one_sparsity_nodes_arr) + "\n")
    f.write("our_decline_nodes={}".format(one_decline_nodes_arr) + "\n")
    f.write("test time,{}".format(tok-tik))
    f.close()

    fidelity_arr.append(one_fidelity_arr)
    fidelity_origin_arr.append(one_fidelity_origin_arr)
    fidelity_complete_arr.append(one_fidelity_complete_arr)
    fidelityminus_complete_arr.append(one_fidelityminus_complete_arr)
    finalfidelity_complete_arr.append(one_finalfidelity_complete_arr)
    sparsity_arr.append(one_sparsity_arr)
    fidelity_delnotimp_arr.append(one_fidelity_delnotimp_arr)
    fidelity_origin_delnotimp_arr.append(one_fidelity_origin_delnotimp_arr)
    fidelity_complete_delnotimp_arr.append(one_fidelity_complete_delnotimp_arr)
    sparsity_delnotimp_arr.append(one_sparsity_delnotimp_arr)
    simula_arr.append(one_simula_arr)
    simula_origin_arr.append(one_simula_origin_arr)
    simula_complete_arr.append(one_simula_complete_arr)
    decline_arr.append(one_decline_arr)
    fidelity_complete_nodes_arr.append(one_fidelity_complete_nodes_arr)
    sparsity_nodes_arr.append(one_sparsity_nodes_arr)
    decline_nodes_arr.append(one_decline_nodes_arr)

print("args.dataset", args.dataset)
print("our_auc_all = ", auc_all)
print("our_ndcg_all = ", ndcg_all)
print("our_fidelity_arr =", fidelity_arr)
print("our_fidelity_origin_arr =", fidelity_origin_arr)
print("our_fidelity_complete_arr =", fidelity_complete_arr)
print("our_fidelityminus_complete_arr=", fidelityminus_complete_arr)
print("our_finalfidelity_complete_arr", finalfidelity_complete_arr)
print("our_sparsity_arr =", sparsity_arr)
print("our_fidelity_delnotimp_arr =", fidelity_delnotimp_arr)
print("our_fidelity_origin_delnotimp_arr =", fidelity_origin_delnotimp_arr)
print("our_fidelity_complete_delnotimp_arr =", fidelity_complete_delnotimp_arr)
print("our_sparsity_delnotimp_arr =", sparsity_delnotimp_arr)
print("our_simula_level_arr =", simula_arr)
print("our_simula_level_origin_arr =", simula_origin_arr)
print("our_simula_complete_arr =", simula_complete_arr)
print("our_decline_arr =", decline_arr)
print("our_fidelity_complete_nodes_arr =", fidelity_complete_nodes_arr)
print("our_sparsity_nodes_arr =", sparsity_nodes_arr)
print("our_decline_nodes_arr =", decline_nodes_arr)

f_mean.write("our_auc_all = {}".format(auc_all)+ "\n")
f_mean.write("our_ndcg_all = {}".format(ndcg_all)+ "\n")
f_mean.write("our_fidelity_arr ={}".format(fidelity_arr)+ "\n")
f_mean.write("our_fidelity_origin_arr = {}".format(fidelity_origin_arr)+ "\n")
f_mean.write("our_fidelity_complete_arr = {}".format(fidelity_complete_arr)+ "\n")
f_mean.write("our_fidelityminus_complete_arr = {}".format(fidelityminus_complete_arr)+"\n")
f_mean.write("our_finalfidelity_complete_arr = {}".format(finalfidelity_complete_arr)+"\n")
f_mean.write("our_sparsity_arr = {}".format(sparsity_arr)+ "\n")
f_mean.write("our_fidelity_delnotimp_arr = {}".format(fidelity_delnotimp_arr)+ "\n")
f_mean.write("our_fidelity_origin_delnotimp_arr = {}".format(fidelity_origin_delnotimp_arr)+ "\n")
f_mean.write("our_fidelity_complete_delnotimp_arr = {}".format(fidelity_complete_delnotimp_arr)+ "\n")
f_mean.write("our_sparsity_delnotimp_arr = {}".format(sparsity_delnotimp_arr)+ "\n")
f_mean.write("our_simula_level_arr = {}".format(simula_arr)+ "\n")
f_mean.write("our_simula_level_origin_arr = {}".format(simula_origin_arr)+ "\n")
f_mean.write("our_simula_complete_arr = {}".format(simula_complete_arr)+ "\n")
f_mean.write("our_decline_arr = {}".format(decline_arr)+ "\n")
f_mean.write("our_fidelity_complete_nodes_arr = {}".format(fidelity_complete_nodes_arr)+ "\n")
f_mean.write("our_sparsity_nodes_arr = {}".format(sparsity_nodes_arr)+ "\n")
f_mean.write("our_decline_nodes_arr = {}".format(decline_nodes_arr)+ "\n")

fidelity_mean = np.average(np.array(fidelity_arr),axis=0)
fidelity_origin_mean = np.average(np.array(fidelity_origin_arr),axis=0)
fidelity_complete_mean = np.average(np.array(fidelity_complete_arr),axis=0)
fidelityminus_complete_mean = np.average(np.array(fidelityminus_complete_arr),axis=0)
finalfidelity_complete_mean = np.average(np.array(finalfidelity_complete_arr), axis=0)
sparsity_mean = np.average(np.array(sparsity_arr),axis=0)
fidelity_delnotimp_mean = np.average(np.array(fidelity_delnotimp_arr),axis=0)
fidelity_origin_delnotimp_mean = np.average(np.array(fidelity_origin_delnotimp_arr),axis=0)
fidelity_complete_delnotimp_mean = np.average(np.array(fidelity_complete_delnotimp_arr),axis=0)
sparsity_delnotimp_mean = np.average(np.array(sparsity_delnotimp_arr),axis=0)
simula_mean = np.average(np.array(simula_arr), axis=0)
simula_origin_mean = np.average(np.array(simula_origin_arr), axis=0)
simula_complete_mean = np.average(np.array(simula_complete_arr),axis=0)
maskout_decline = np.average(np.array(decline_arr),axis=0)
fidelity_complete_nodes_mean = np.average(np.array(fidelity_complete_nodes_arr),axis=0)
sparsity_nodes_mean = np.average(np.array(sparsity_nodes_arr),axis=0)
maskout_decline_nodes = np.average(np.array(decline_nodes_arr),axis=0)

print("our_auc_mean =", np.mean(auc_all))
print("our_ndcg_mean =", np.mean(ndcg_all))
print("our_fidelity_mean = ", list(fidelity_mean))
print("our_fidelity_origin_mean =", list(fidelity_origin_mean))
print("our_fidelity_complete_mean =", list(fidelity_complete_mean))
print("our_fidelityminus_complete_mean =", list(fidelityminus_complete_mean))
print("our_finalfidelity_complete_mean =", list(finalfidelity_complete_mean))
print("our_sparsity_mean =", list(sparsity_mean))
print("our_fidelity_delnotimp_mean = ", list(fidelity_delnotimp_mean))
print("our_fidelity_origin_delnotimp_mean =", list(fidelity_origin_delnotimp_mean))
print("our_fidelity_complete_delnotimp_mean =", list(fidelity_complete_delnotimp_mean))
print("our_sparsity_delnotimp_mean =", list(sparsity_delnotimp_mean))
print("our_simula_level_mean =", list(simula_mean))
print("our_simula_level_origin_mean =", list(simula_origin_mean))
print("our_simula_complete_mean =", list(simula_complete_mean))
print("our_maskout_decline =", list(maskout_decline))
print("our_fidelity_complete_nodes_mean =", list(fidelity_complete_nodes_mean))
print("our_sparsity_nodes_mean =", list(sparsity_nodes_mean))
print("our_maskout_decline_nodes =", list(maskout_decline_nodes))

f_mean.write("our_auc_mean = {}".format(np.mean(auc_all))+ "\n")
f_mean.write("our_ndcg_mean = {}".format(np.mean(ndcg_all))+ "\n")
f_mean.write("our_fidelity_mean = {}".format(list(fidelity_mean))+ "\n")
f_mean.write("our_fidelity_origin_mean = {}".format(list(fidelity_origin_mean))+ "\n")
f_mean.write("our_fidelity_complete_mean = {}".format(list(fidelity_complete_mean))+ "\n")
f_mean.write("our_fidelityminus_complete_mean = {}".format(list(fidelityminus_complete_mean))+"\n")
f_mean.write("our_finalfidelity_complete_mean = {}".format(list(finalfidelity_complete_mean))+"\n")
f_mean.write("our_sparsity_mean = {}".format(list(sparsity_mean))+ "\n")
f_mean.write("our_fidelity_delnotimp_mean = {}".format(list(fidelity_delnotimp_mean))+ "\n")
f_mean.write("our_fidelity_origin_delnotimp_mean = {}".format(list(fidelity_origin_delnotimp_mean))+ "\n")
f_mean.write("our_fidelity_complete_delnotimp_mean = {}".format(list(fidelity_complete_delnotimp_mean))+ "\n")
f_mean.write("our_sparsity_delnotimp_mean = {}".format(list(sparsity_delnotimp_mean))+ "\n")
f_mean.write("our_simula_level_mean = {}".format(list(simula_mean))+ "\n")
f_mean.write("our_simula_level_origin_mean = {}".format(list(simula_origin_mean))+ "\n")
f_mean.write("our_simula_complete_mean = {}".format(list(simula_complete_mean))+ "\n")
f_mean.write("our_maskout_decline = {}".format(list(maskout_decline))+ "\n")
f_mean.write("our_fidelity_complete_nodes_mean = {}".format(list(fidelity_complete_nodes_mean))+ "\n")
f_mean.write("our_sparsity_nodes_mean = {}".format(list(sparsity_nodes_mean))+ "\n")
f_mean.write("our_maskout_decline_nodes = {}".format(list(maskout_decline_nodes))+ "\n")
f_mean.close()


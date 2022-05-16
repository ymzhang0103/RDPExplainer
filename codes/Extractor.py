import numpy as np
from scipy.sparse import coo_matrix
import torch

class Extractor(object):
    def __init__(self, adj, edge_index, features, edge_label_matrix, embeds, labels, hops, **kwargs):
        super(Extractor,self).__init__(**kwargs)
        if isinstance(edge_index, torch.Tensor):
            self.edge_index = edge_index.cpu().numpy()
        else:
            self.edge_index = edge_index
        self.features = features
        self.embeds = embeds
        self.labels = labels
        self.hops = hops
        self.ext_adj = self.extend(adj, hops-1)
        self.edge_label_matrix = edge_label_matrix
        if isinstance(adj, np.ndarray):
            adj_coo = coo_matrix(adj)
        else:
            adj_coo = adj.tocoo()
        adj_list = []
        for _ in range(adj_coo.shape[0]):
            adj_list.append(set())

        for r,c in list(zip(adj_coo.row, adj_coo.col)):
            adj_list[r].add(c)
            adj_list[c].add(r)
        self.adj_list= adj_list

    def extend(self, adj, hops):
        ext_adj = adj.copy()
        for hop in range(hops):
            ext_adj = ext_adj @ adj + adj
        return ext_adj

    def subgraph(self, node):
        begin_index = self.ext_adj.indptr[node]
        end_index = self.ext_adj.indptr[node+1]
        subnodes_set = set(self.ext_adj.indices[begin_index:end_index])
        subnodes_set.add(node)
        remap = {}
        remap[node] = 0
        sub_nodes = [node]
        for n in subnodes_set:
            if n not in remap:
                remap[n]=len(remap)
                sub_nodes.append(n)
        row = []
        col = []
        data = []
        edge_label = []
        sub_edge_idx = []
        for n in remap:
            newid = remap[n]
            for nb in self.adj_list[n]:
                if nb in remap:
                    nb_new_id = remap[nb]
                    row.append(newid)
                    col.append(nb_new_id)
                    data.append(1.0)
                    edge_label.append(self.edge_label_matrix[n,nb])
                    
                    edge_idx = np.where((n == self.edge_index[0]) & (nb == self.edge_index[1]))[0]
                    sub_edge_idx.extend(edge_idx)
        sub_adj = coo_matrix((data,(row,col)),shape=(len(remap),len(remap)))
        sub_edge_label_matrix = coo_matrix((edge_label,(row,col)),shape=(len(remap),len(remap)))

        sub_features = self.features[sub_nodes]
        sub_labels = self.labels[sub_nodes]
        sub_embeds = self.embeds[sub_nodes]

        return sub_adj, sub_features, sub_embeds, sub_labels, sub_edge_label_matrix, sub_nodes, sub_edge_idx






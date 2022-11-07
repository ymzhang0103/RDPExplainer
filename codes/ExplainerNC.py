import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GATConv

import numpy as np
from math import sqrt
import scipy.sparse as sp

class ExplainerNC(nn.Module):
    def __init__(self, model, args, **kwargs):
        super(ExplainerNC, self).__init__(**kwargs)   
        
        hiddens = args.latent_dim
        input_dim = hiddens[0] * 3

        self.device = model.device
        
        if args.maskmodel == "GAT":
            self.elayers = nn.Sequential(
                GATConv(input_dim, 64),
                nn.Linear(64, 1)
            )
        elif args.maskmodel == "NOGAT":
            self.elayers = nn.Sequential(
                nn.Linear(input_dim, 1)
            )
        self.model = model
        self.args = args
        self.mask_act = 'sigmoid'
        self.params = []

        self.softmax = nn.Softmax(dim=0)

        self.init_bias = 0.0

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        r""" Set the edge weights before message passing
        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)

        The :attr:`edge_mask` will be randomly initialized when set to :obj:`None`.

        .. note:: When you use the :meth:`~OurExplainer.__set_masks__`,
          the explain flag for all the :class:`torch_geometric.nn.MessagePassing`
          modules in :attr:`model` will be assigned with :obj:`True`. In addition,
          the :attr:`edge_mask` will be assigned to all the modules.
          Please take :meth:`~OurExplainer.__clear_masks__` to reset.
        """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #std = torch.nn.init.calculate_gain('sigmoid') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def _masked_adj(self,mask,adj):
        mask = mask.to(self.device)
        sym_mask = mask
        sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2

        # Create sparse tensor TODO: test and "maybe" a transpose is needed somewhere
        sparseadj = torch.sparse_coo_tensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(torch.Tensor(adj.row),-1), torch.unsqueeze(torch.Tensor(adj.col),-1)], dim=-1), 0, 1).to(torch.int64),
            values=adj.data,
            size=adj.shape
        )

        adj = sparseadj.coalesce().to_dense().to(torch.float32).to(self.device) #FIXME: tf.sparse.reorder was also applied, but probably not necessary. Maybe it needs a .coalesce() too tho?
        self.adj = adj

        masked_adj = torch.mul(adj, sym_mask)

        num_nodes = adj.shape[0]
        ones = torch.ones((num_nodes, num_nodes))
        diag_mask = ones.to(torch.float32) - torch.eye(num_nodes)
        diag_mask = diag_mask.to(self.device)
        return torch.mul(masked_adj,diag_mask)


    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        if training:
            bias = self.args.sample_bias
            random_noise = torch.FloatTensor(log_alpha.shape).uniform_(bias, 1.0-bias)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)
        return gate_inputs

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        # return sparse_to_tuple(adj_normalized)
        return self.sparse_mx_to_torch_sparse_tensor(adj_normalized)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to_dense()

    def forward(self, inputs, training=False):
        x, adj, nodeid, embed, sub_new_edge_index, tmp = inputs
        x = x.to(self.device)
        
        self.tmp = tmp
        if not isinstance(embed[adj.row], torch.Tensor):
            f1 = torch.tensor(embed[adj.row]).to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = torch.tensor(embed[adj.col]).to(self.device)
        else:
            f1 = embed[adj.row].to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = embed[adj.col].to(self.device)
        selfemb = embed[nodeid] if isinstance(embed, torch.Tensor) else torch.tensor(embed[nodeid])
        selfemb = torch.unsqueeze(selfemb, 0).repeat([f1.shape[0], 1]).to(self.device)
        f12self = torch.cat([f1, f2, selfemb], dim=-1)

        h = f12self.to(self.device)
        # GNN
        temp_l = 0
        l = len(self.elayers)
        for elayer in self.elayers:
            if temp_l < l - 1:
                h = elayer(h, sub_new_edge_index) 
            else:
                h = elayer(h)
            temp_l = temp_l + 1

        self.values = torch.reshape(h, [-1,])
        values = self.concrete_sample(self.values, beta=tmp, training=training)

        sparse_edge_mask = torch.sparse.FloatTensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(torch.tensor(adj.row),-1), torch.unsqueeze(torch.tensor(adj.col),-1)], dim=-1), 0, 1).to(torch.int64).to(self.device),
            values=values,
            size=adj.shape
        )

        mask = sparse_edge_mask.coalesce().to_dense().to(torch.float32)  #FIXME: again a reorder() is omitted, maybe coalesce
        masked_adj = self._masked_adj(mask,adj)

        self.mask = mask
        self.masked_adj = masked_adj

        edge_index = dense_to_sparse(masked_adj)[0]
        edge_mask = masked_adj[edge_index[0], edge_index[1]]
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)
        data = Data(x=x, edge_index=edge_index)
        output, probs, embed = self.model(data)

        node_pred = output[nodeid, :]
        res = self.softmax(node_pred)
        self.__clear_masks__()
        return res

    def deterministic_NeuralSort(self, s, tau=0.00001, hard=False):
        """s: input elements to be sorted. 
        Shape: batch_size x n x 1
        tau: temperature for relaxation. Scalar."""
        n = s.size()[1]
        bsize = s.size()[0]
        one = torch.ones((n, 1), dtype = torch.float32).to(self.device)
        A_s = torch.abs(s - s.permute(0, 2, 1))
        B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (n + 1 - 2*(torch.arange(n) + 1)).type(torch.float32).to(self.device)
        C = torch.matmul(s, scaling.unsqueeze(0))
        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)
        if hard==True:
            P = torch.zeros_like(P_hat)
            b_idx = torch.arange(bsize).repeat([1, n]).view(n, bsize).transpose(dim0=1, dim1=0).flatten().type(torch.float32).to(self.device)
            r_idx = torch.arange(n).repeat([bsize, 1]).flatten().type(torch.float32).to(self.device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            #P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P[brc_idx[0].type(torch.int32).cpu().numpy(), brc_idx[1].type(torch.int32).cpu().numpy(), brc_idx[2].type(torch.int32).cpu().numpy()] = 1
            P_hat = (P-P_hat).detach() + P_hat
        return P_hat

    def loss(self, pred, ori_pred, loss_flag):
        """
        Args:
           pred: prediction made by current model
           ori_pred: prediction made by the original model.
        """
        #pl loss
        P = self.deterministic_NeuralSort(pred.unsqueeze(0).unsqueeze(-1), 0.00001) 
        ori_pred_ranked = torch.matmul(P, ori_pred.unsqueeze(0).t())[0].t()[0]
        pl_loss = 0
        for i in range(len(ori_pred_ranked)):
            s = sum(ori_pred_ranked[i:])
            pl_loss = pl_loss - torch.log(ori_pred_ranked[i] / s)

        # value loss
        pre_rp, r = torch.sort(ori_pred, descending=True)
        pred_ranked = pred[r]
        value_loss = self.args.coff_diff* sum(torch.abs(pred_ranked - pre_rp))

        # size
        if self.args.budget<=0:
            size_loss = self.args.coff_size * torch.sum(self.mask)#len(self.mask[self.mask > 0]) 
        else:
            relu = nn.ReLU()
            size_loss = self.args.coff_size * relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)

        if  loss_flag == "plandvalue":
            loss = pl_loss + value_loss +size_loss
        elif loss_flag == "pl":
            loss = pl_loss + size_loss
        elif loss_flag == "value":
            loss = value_loss + size_loss
            
        return loss, pl_loss, value_loss, size_loss

    
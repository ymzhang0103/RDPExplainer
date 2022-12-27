import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import MessagePassing
from torch import Tensor
from math import sqrt
from torch_geometric.nn.conv import GATConv

class ExplainerGC(nn.Module):
    def __init__(self, model, args, **kwargs):
        super(ExplainerGC, self).__init__(**kwargs)

        hiddens = args.latent_dim
        input_dim = hiddens[-1] * 2
        
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
        self.softmax = nn.Softmax(dim=-1)

        self.mask_act = 'sigmoid'
        self.init_bias = 0.0

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        if training:
            #debug_var = 0.0
            #bias = 0.0
            #random_noise = bias + torch.FloatTensor(log_alpha.shape).uniform_(debug_var, 1.0-debug_var)
            bias = self.args.sample_bias
            random_noise = torch.FloatTensor(log_alpha.shape).uniform_(bias, 1.0-bias)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

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
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

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


    def convert_new_directed_edge_index_bak(self, edge_index):
        edge_num = len(edge_index[0])
        for i in range(edge_num):
            raw_idx = torch.where(edge_index[1] == edge_index[0][i])[0] #head
            col_idx = torch.where(edge_index[0] == edge_index[1][i])[0] #tail
            con_idx = torch.tensor(list(set(col_idx.cpu().numpy())|set(raw_idx.cpu().numpy())))
            temp_edge = torch.cat((torch.tensor([i] * len(con_idx)).to(self.device).unsqueeze(0), con_idx.to(self.device).unsqueeze(0)))
            if i==0:
                new_edge_index = temp_edge
            else:
                new_edge_index = torch.cat((new_edge_index, temp_edge), dim=1)
        return new_edge_index
        
    def convert_new_directed_edge_index(self, edge_index):
        edge_num = len(edge_index[0])
        edge_hash = dict()
        for i in range(edge_num):
            start_node = edge_index[0][i].item()
            end_node = edge_index[1][i].item()
            if start_node in edge_hash.keys():
                start_edgeset, end_edgeset = edge_hash[start_node]
                start_edgeset.add(i)
                edge_hash[start_node] = (start_edgeset, end_edgeset)
            else:
                start_edgeset=set()
                start_edgeset.add(i)
                edge_hash[start_node] = (start_edgeset, set())
            if end_node in edge_hash.keys():
                start_edgeset, end_edgeset = edge_hash[end_node]
                end_edgeset.add(i)
                edge_hash[end_node] = (start_edgeset, end_edgeset)
            else:
                end_edgeset=set()
                end_edgeset.add(i)
                edge_hash[end_node] = (set(), end_edgeset)
        
        for i in range(edge_num):
            start_node = edge_index[0][i].item()
            end_node = edge_index[1][i].item()
            _, end_edgeset = edge_hash[start_node]
            start_edgeset, _ = edge_hash[end_node]
            #temp_edge_head = torch.cat((torch.tensor(list(start_edgeset)).unsqueeze(0),torch.tensor([i] * len(start_edgeset)).unsqueeze(0)))
            #temp_edge_tail = torch.cat((torch.tensor([i] * len(end_edgeset)).unsqueeze(0), torch.tensor(list(end_edgeset)).unsqueeze(0)))
            #temp_edge = torch.cat((temp_edge_head, temp_edge_tail), dim=1)
            con_idx = torch.cat((torch.tensor(list(start_edgeset)), torch.tensor(list(end_edgeset)))).unique()
            temp_edge = torch.cat((torch.tensor([i] * len(con_idx)).unsqueeze(0), con_idx.unsqueeze(0)))
            if i==0:
                new_edge_index = temp_edge
            else:
                new_edge_index = torch.cat((new_edge_index, temp_edge), dim=1)
        return new_edge_index.to(self.device)

    '''
    def forward(self, inputs, training=False):
        x, embed, adj, label, tmp = inputs
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        adj = adj.to(self.device)
        self.label = label.to(self.device)
        edge_index = dense_to_sparse(adj)[0]
        row = edge_index[0]
        col = edge_index[1]
        if not isinstance(embed[row], torch.Tensor):
            f1 = torch.Tensor(embed[row]).to(self.device) 
            f2 = torch.Tensor(embed[col]).to(self.device)
        else:
            f1 = embed[row] 
            f2 = embed[col]
        h = torch.cat([f1, f2], dim=-1)
        h = h.to(self.device)

        # GNN
        new_edge_index = self.convert_new_directed_edge_index(edge_index)
        temp_l = 0
        l = len(self.elayers)
        
        for elayer in self.elayers:
            if temp_l < l - 1:
                h = elayer(h, new_edge_index) 
            else:
                h = elayer(h)
            temp_l = temp_l + 1
        self.values = torch.reshape(h, [-1])
        values = self.concrete_sample(self.values, beta=tmp, training=training)

        nodesize = x.shape[0]
        sparsemask = torch.sparse.FloatTensor(
            indices=torch.nonzero(adj).T.to(torch.int64),
            values=values,
            size=[nodesize, nodesize]
        ).to(self.device)
        sym_mask = sparsemask.coalesce().to_dense().to(torch.float32) 
        self.mask = sym_mask

        sym_mask = (sym_mask + sym_mask.T) / 2
        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj
        edge_index = dense_to_sparse(masked_adj)[0]
        edge_mask = masked_adj[edge_index[0], edge_index[1]]
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        output, _, _ = self.model(data)
        res = self.softmax(output.squeeze())
        self.__clear_masks__()
        return res
    '''

    def forward(self, inputs, training=False):
        x, embed, edge_index, new_edge_index, label, tmp = inputs
        adj = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(edge_index.shape[1]).to(self.device), size=(x.shape[0], x.shape[0])).to_dense()
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        self.label = label.to(self.device)
        row = edge_index[0]
        col = edge_index[1]
        if not isinstance(embed[row], torch.Tensor):
            f1 = torch.Tensor(embed[row]).to(self.device) 
            f2 = torch.Tensor(embed[col]).to(self.device)
        else:
            f1 = embed[row] 
            f2 = embed[col]
        h = torch.cat([f1, f2], dim=-1)
        h = h.to(self.device)

        # GNN
        temp_l = 0
        l = len(self.elayers)
        
        for elayer in self.elayers:
            if temp_l < l - 1:
                h = elayer(h, new_edge_index) 
            else:
                h = elayer(h)
            temp_l = temp_l + 1
        self.values = torch.reshape(h, [-1])
        values = self.concrete_sample(self.values, beta=tmp, training=training)

        nodesize = x.shape[0]
        sparsemask = torch.sparse.FloatTensor(
            indices=edge_index,
            values=values,
            size=[nodesize, nodesize]
        ).to(self.device)
        sym_mask = sparsemask.coalesce().to_dense().to(torch.float32) 
        self.mask = sym_mask

        sym_mask = (sym_mask + sym_mask.T) / 2
        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj
        edge_mask = masked_adj[edge_index[0], edge_index[1]]
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        output, _, _ = self.model(data)
        res = self.softmax(output.squeeze())
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
        pred_ranked = torch.matmul(P, ori_pred.unsqueeze(0).t())[0].t()[0]
        pl_loss = 0
        for i in range(len(pred_ranked)):
            s = sum(pred_ranked[i:])
            pl_loss = pl_loss - torch.log(pred_ranked[i] / s)

        #value loss
        pre_rp, r = torch.sort(ori_pred, descending=True)
        pred_ranked = pred[r]
        value_loss = sum(torch.abs(pred_ranked - pre_rp))

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.args.coff_size * torch.sum(mask) 

        if  loss_flag == "plandvalue":
            loss = pl_loss + value_loss +size_loss
        elif loss_flag == "pl":
            loss = pl_loss + size_loss
        elif loss_flag == "value":
            loss = value_loss + size_loss

        return loss, pl_loss, value_loss, size_loss

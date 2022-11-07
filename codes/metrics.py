#from codes.ExplainConfig import dtype
import torch
import torch.nn.functional as F
from typing import List, Union
import numpy as np
from torch import Tensor
from math import sqrt
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from typing import Union
from scipy.sparse import coo_matrix

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    loss = F.cross_entropy(preds, torch.argmax(labels, dim=1), reduction='none')
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    loss = loss * mask
    return torch.mean(loss)

def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)

    loss = F.cross_entropy(preds, torch.argmax(labels, dim=1), reduction='none')
    return torch.mean(loss)

def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    logits = torch.tensor(preds)
    p = torch.tensor(labels)
    loss = p*-torch.log(torch.sigmoid(logits)) + (1-p)*-torch.log(1-torch.sigmoid(logits))
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    loss = loss * mask
    return torch.mean(loss)

def sigmoid_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)

    logits = torch.tensor(preds)
    p = torch.tensor(labels)
    loss = p*-torch.log(torch.sigmoid(logits)) + (1-p)*-torch.log(1-torch.sigmoid(logits))
    return torch.mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32)
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    accuracy_all = accuracy_all * mask
    return torch.mean(accuracy_all)

def accuracy(preds, labels):
    """Accuracy with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)

    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32)
    return torch.mean(accuracy_all)

def rho_ndcg(dists, new_dists, hitsn):
    _, pre_r = torch.sort(dists,descending=True)
    sub_r = pre_r[:hitsn]
    dists = dists[sub_r]
    new_dists = new_dists[sub_r]
    r_dists, r_new = torch.sort(dists,descending=True)
    r_new_dists, r_mask = torch.sort(new_dists,descending=True)
    r_new = len(dists) - r_new
    r_mask = len(new_dists) - torch.IntTensor([list(r_new_dists).index(d) for d in new_dists])

    #print("hits@",hitsn, "ndcg",ndcg, "\ndists",dists, "\nnew_dists",new_dists, "\nr_new",r_new,"\n r_mask",r_mask)
    dcg = torch.sum(torch.divide(r_mask, torch.log2(torch.arange(2, len(r_mask) + 2).float())))
    dcg_pf = torch.sum(torch.divide(r_new.cpu(), torch.log2(torch.arange(2, len(r_new.cpu()) + 2).float())))
    ndcg = torch.divide(dcg, dcg_pf).item()
    return ndcg, r_mask, dists, new_dists

def fidelity_bak(ori_probs: torch.Tensor, maskout_probs: torch.Tensor) -> float:
    drop_probability = ori_probs - maskout_probs
    return drop_probability.mean().item()

def fidelity(ori_probs: torch.Tensor, maskout_probs: torch.Tensor) -> float:
    drop_probability = abs(ori_probs - maskout_probs)
    return drop_probability.mean().item()

def fidelity_complete(ori_probs, maskout_probs):
    drop_prob_complete = [ori_probs[i] - maskout_probs[i] for i in range(len(ori_probs))]
    result = np.mean([sum(abs(i)).item() for i in drop_prob_complete])
    return result


class XCollector(object):
    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'retainimp':[], 'sparsity_edges': [], 'maskimp_nodes':[], 'retainimp_nodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'retainimp_l':[], 'maskimp_nodes_l':[], 'retainimp_nodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'retainimp_ol':[], 'maskimp_nodes_ol':[], 'retainimp_nodes_ol':[], 'label': [], 'origin_label': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity_edges = sparsity
        self.__simula, self.__simula_origin, self.__simula_complete, self.__fidelity, self.__fidelity_origin, self.__fidelity_complete, self.__fidelityminus, self.__fidelityminus_origin, self.__fidelityminus_complete, self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__fidelityminus_nodes, self.__fidelityminus_origin_nodes, self.__fidelityminus_complete_nodes, self.__sparsity_nodes = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'retainimp':[], 'sparsity_edges': [], 'maskimp_nodes':[], 'retainimp_nodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'retainimp_l':[], 'maskimp_nodes_l':[], 'retainimp_nodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'retainimp_ol':[], 'maskimp_nodes_ol':[], 'retainimp_nodes_ol':[], 'label': [], 'origin_label': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__simula, self.__simula_origin, self.__simula_complete, self.__fidelity, self.__fidelity_origin, self.__fidelity_complete, self.__fidelityminus, self.__fidelityminus_origin, self.__fidelityminus_complete, self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__fidelityminus_nodes, self.__fidelityminus_origin_nodes, self.__fidelityminus_complete_nodes, self.__sparsity_nodes = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)
        """
        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)


    @property
    def simula(self):
        if self.__simula:
            return self.__simula
        elif None in self.__related_preds['masked_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__simula = fidelity(origin_preds, masked_preds)
            return self.__simula

    @property
    def simula_origin(self):
        if self.__simula_origin:
            return self.__simula_origin
        elif None in self.__related_preds['masked_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__simula_origin = fidelity(origin_preds, masked_preds)
            return self.__simula_origin

    @property
    def simula_complete(self):
        if self.__simula_complete:
            return self.__simula_complete
        elif None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
            return None
        else:
            masked_preds, origin_preds= self.__related_preds['masked'], self.__related_preds['origin']
            self.__simula_complete = fidelity_complete(origin_preds, masked_preds)
            return self.__simula_complete

    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        elif None in self.__related_preds['maskimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity = fidelity(origin_preds, maskout_preds)
            return self.__fidelity

    @property
    def fidelity_origin(self):
        if self.__fidelity_origin:
            return self.__fidelity_origin
        elif None in self.__related_preds['maskimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_origin

    @property
    def fidelity_complete(self):
        if self.__fidelity_complete:
            return self.__fidelity_complete
        elif None in self.__related_preds['maskimp'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp'], self.__related_preds['origin']
            self.__fidelity_complete = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete

    @property
    def fidelityminus(self):
        if self.__fidelityminus:
            return self.__fidelityminus
        elif None in self.__related_preds['retainimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelityminus = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus

    @property
    def fidelityminus_origin(self):
        if self.__fidelityminus_origin:
            return self.__fidelityminus_origin
        elif None in self.__related_preds['retainimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelityminus_origin = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_origin

    @property
    def fidelityminus_complete(self):
        if self.__fidelityminus_complete:
            return self.__fidelityminus_complete
        elif None in self.__related_preds['retainimp'] or None in self.__related_preds['origin']:
            return None
        else:
            retainimp_preds, origin_preds = self.__related_preds['retainimp'], self.__related_preds['origin']
            self.__fidelityminus_complete = fidelity_complete(origin_preds, retainimp_preds)
            return self.__fidelityminus_complete
            
    @property
    def sparsity_edges(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity_edges:
            return self.__sparsity_edges
        elif None in self.__related_preds['sparsity_edges']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_edges']).mean().item()

    @property
    def fidelity_nodes(self):
        if self.__fidelity_nodes:
            return self.__fidelity_nodes
        elif None in self.__related_preds['maskimp_nodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_nodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity_nodes = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_nodes

    @property
    def fidelity_origin_nodes(self):
        if self.__fidelity_origin_nodes:
            return self.__fidelity_origin_nodes
        elif None in self.__related_preds['maskimp_nodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_nodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin_nodes = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_origin_nodes

    @property
    def fidelity_complete_nodes(self):
        if self.__fidelity_complete_nodes:
            return self.__fidelity_complete_nodes
        elif None in self.__related_preds['maskimp_nodes'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp_nodes'], self.__related_preds['origin']
            self.__fidelity_complete_nodes = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete_nodes
    
    @property
    def fidelityminus_nodes(self):
        if self.__fidelityminus_nodes:
            return self.__fidelityminus_nodes
        elif None in self.__related_preds['retainimp_nodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_nodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelityminus_nodes = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_nodes

    @property
    def fidelityminus_origin_nodes(self):
        if self.__fidelityminus_origin_nodes:
            return self.__fidelityminus_origin_nodes
        elif None in self.__related_preds['retainimp_nodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_nodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelityminus_origin_nodes = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_origin_nodes

    @property
    def fidelityminus_complete_nodes(self):
        if self.__fidelityminus_complete_nodes:
            return self.__fidelityminus_complete_nodes
        elif None in self.__related_preds['retainimp_nodes'] or None in self.__related_preds['origin']:
            return None
        else:
            retainimp_preds, origin_preds = self.__related_preds['retainimp_nodes'], self.__related_preds['origin']
            self.__fidelityminus_complete_nodes = fidelity_complete(origin_preds, retainimp_preds)
            return self.__fidelityminus_complete_nodes
     
    @property
    def sparsity_nodes(self):
        if self.__sparsity_nodes:
            return self.__sparsity_nodes
        elif None in self.__related_preds['sparsity_nodes']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_nodes']).mean().item()




class MaskoutMetric:
    def __init__(
        self,
        model, 
        prog_args
    ):
        self.model = model
        self.model.eval()
        self.prog_args= prog_args

    def GnnNets_NC2value_func_new(self, gnnNets_NC, node_idx):
        def value_func(data):
            with torch.no_grad():
                logits, probs, _ = gnnNets_NC(data=data)
                # select the corresponding node prob through the node idx on all the sampling graphs
                batch_size = data.batch.max() + 1
                probs = probs.reshape(batch_size, -1, probs.shape[-1])
                scores = probs[:, node_idx]
                return scores
        return value_func

    def GnnNets_GC2value_func_new(self, gnnNets):
        def value_func(batch):
            with torch.no_grad():
                logits, probs, _ = gnnNets(data=batch)
                #probs = F.softmax(logits, dim=-1)
                score = probs.squeeze()
            return score
        return value_func

    def gnn_prob(self, coalition: list, data: Data, value_func: str, subgraph_building_method='zero_filling') -> torch.Tensor:
        """ the value of subgraph with selected nodes """
        num_nodes = data.num_nodes
        subgraph_build_func = self.get_graph_build_func(subgraph_building_method)
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
        mask[coalition] = 1.0
        ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
        mask_data = Data(x=ret_x, edge_index=ret_edge_index)
        mask_data = Batch.from_data_list([mask_data])
        score = value_func(mask_data)
        # get the score of predicted class for graph or specific node idx
        return score

    def get_graph_build_func(self, build_method):
        if build_method.lower() == 'zero_filling':
            return self.graph_build_zero_filling
        elif build_method.lower() == 'split':
            return self.graph_build_split
        else:
            raise NotImplementedError

    def graph_build_zero_filling(self, X, edge_index, node_mask: np.array):
        """ subgraph building through masking the unselected nodes with zero features """
        ret_X = X * node_mask.unsqueeze(1)
        return ret_X, edge_index

    def graph_build_split(X, edge_index, node_mask: np.array):
        """ subgraph building through spliting the selected nodes from the original graph """
        row, col = edge_index
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        ret_edge_index = edge_index[:, edge_mask]
        return X, ret_edge_index

    def calculate_selected_nodes(self, edge_index, edge_mask, top_k):
        threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
        hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]
        selected_nodes = []
        #edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx].item(), edge_index[1][edge_idx].item()]
        selected_nodes = list(set(selected_nodes))
        return selected_nodes

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = 0.0
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        
        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

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
        #self.node_feat_masks = None
        self.edge_mask = None

    def evaluate_adj_new(self, node_idx_new, features, adj):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.prog_args.device)
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, new_edge_mask) 
            data = Data(x=features, edge_index=sub_edge_index)
            logit, pred, _ = self.model(data.to(self.prog_args.device))
            _, pred_label = torch.max(pred, 1)
            self.__clear_masks__()
        return pred_label[node_idx_new], pred[node_idx_new]


    def metric_del_edges(self, node_idx_new, feat, sub_adj, masked_adj, origin_pred, masked_pred, label, topk_arr):
        number_of_nodes = feat.shape[0]
        #masked_adj = coo_matrix(masked_adj)
        origin_label = origin_pred.argmax(-1)
        sub_adj = coo_matrix(sub_adj)
        edge_mask = masked_adj[sub_adj.row, sub_adj.col].to(self.prog_args.device)

        sub_edge_index = torch.LongTensor([sub_adj.row, sub_adj.col]).to(self.prog_args.device)
        x =feat.to(self.prog_args.device)

        related_preds_dict = dict()
        for top_k in topk_arr:
            select_k = round(top_k/100 * len(sub_edge_index[0]))

            #mask important edges
            selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]        ##select top_k% important edges
            sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

            delimp_edge_mask = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            #delimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]      #set selected edges' weight to 1-mask
            delimp_edge_mask[selected_impedges_idx] = 0.0  #remove important edges
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, delimp_edge_mask)    
            data = Data(x=x, edge_index=sub_edge_index)
            self.model.eval()
            _, maskimp_preds, embed = self.model(data)
            maskimp_pred = maskimp_preds[node_idx_new]
            self.__clear_masks__()
            
            #retain important edges
            retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #select other edges  exclude top_k% important edges
            #retainimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]    #set other edges' weight to mask
            retainimp_edge_mask[other_notimpedges_idx] = 0.0  #remove not important edges
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, retainimp_edge_mask)    
            data = Data(x=x, edge_index=sub_edge_index)
            self.model.eval()
            _, retainimp_preds, embed = self.model(data)
            retainimp_pred = retainimp_preds[node_idx_new]
            self.__clear_masks__()

            #delete important nodes features
            selected_nodes = self.calculate_selected_nodes(sub_edge_index, edge_mask, select_k)
            if node_idx_new not in selected_nodes:
                selected_nodes.append(node_idx_new)
            maskout_nodes_list = [node for node in range(len(x)) if node not in selected_nodes or node == node_idx_new]
            value_func = self.GnnNets_NC2value_func_new(self.model, node_idx=node_idx_new)
            maskimp_pred_nodes = self.gnn_prob(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
            maskimp_pred_nodes = maskimp_pred_nodes[0]
            retainimp_pred_nodes = self.gnn_prob(selected_nodes, data, value_func, subgraph_building_method='zero_filling')
            retainimp_pred_nodes = retainimp_pred_nodes[0]
            sparsity_nodes = 1 - len(selected_nodes) / number_of_nodes

            related_preds = [{
                'label': label,
                'origin_label': origin_label,
                'origin': origin_pred,
                'origin_l': origin_pred[label],
                'origin_ol': origin_pred[origin_label],
                'masked': masked_pred,
                'masked_l': masked_pred[label],
                'masked_ol': masked_pred[origin_label],
                'maskimp': maskimp_pred,
                'maskimp_l': maskimp_pred[label],
                'maskimp_ol': maskimp_pred[origin_label],
                'retainimp':retainimp_pred,
                'retainimp_l':retainimp_pred[label],
                'retainimp_ol':retainimp_pred[origin_label],
                'sparsity_edges': sparsity_edges,
                'maskimp_nodes': maskimp_pred_nodes,
                'maskimp_nodes_l':maskimp_pred_nodes[label],
                'maskimp_nodes_ol':maskimp_pred_nodes[origin_label],
                'retainimp_nodes': retainimp_pred_nodes,
                'retainimp_nodes_l':retainimp_pred_nodes[label],
                'retainimp_nodes_ol':retainimp_pred_nodes[origin_label],
                'sparsity_nodes': sparsity_nodes
            }]
            related_preds_dict[top_k] = related_preds

        pred_mask = [edge_mask.detach().cpu().numpy()]
        return pred_mask, related_preds_dict


    def evaluate_adj_new_GC(self, features, adj):
        with torch.no_grad():
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            self.__clear_masks__()
            self.__set_masks__(features, sub_edge_index, new_edge_mask) 
            data = Data(x=features, edge_index=sub_edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
            _, pred, _ = self.model(data)
            _, pred_label = torch.max(pred, 1)
            self.__clear_masks__()
        return pred_label, pred[0]

    def metric_del_edges_GC(self, topk_arr, sub_feature, mask, sub_edge_index, origin_pred, masked_pred, label):
        origin_label = torch.argmax(origin_pred)
        edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]
        related_preds_dict = dict()
        for top_k in topk_arr:
            select_k = round(top_k/100 * len(sub_edge_index[0]))

            #mask important edges
            x = sub_feature.to(self.prog_args.device)
            selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]  #select top_k% important edges
            sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

            delimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            delimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]   #set selected edges' weight to 1-mask
            #delimp_edge_mask[selected_impedges_idx] = 0.0
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, delimp_edge_mask)    
            data = Data(x=x, edge_index=sub_edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
            self.model.eval()
            _, maskimp_preds, embed = self.model(data)
            maskimp_pred = maskimp_preds.squeeze()
            self.__clear_masks__()

            #retain important edges
            retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]   #select other edges  exclude top_k% important edges
            retainimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]  #set other edges' weight to mask
            #retainimp_edge_mask[other_notimpedges_idx] = 0.0
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, retainimp_edge_mask)    
            data = Data(x=x, edge_index=sub_edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
            self.model.eval()
            _, retainimp_preds, embed = self.model(data)
            retainimp_pred = retainimp_preds.squeeze()
            self.__clear_masks__()

            #mask nodes
            selected_nodes = self.calculate_selected_nodes(sub_edge_index, edge_mask, top_k)
            maskout_nodes_list = [node for node in range(sub_feature.shape[0]) if node not in selected_nodes]
            value_func = self.GnnNets_GC2value_func_new(self.model)
            maskimp_pred_nodes = self.gnn_prob(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
            retainimp_pred_nodes = self.gnn_prob(selected_nodes, data, value_func, subgraph_building_method='zero_filling')
            sparsity_nodes = 1 - len(selected_nodes) / sub_feature.shape[0]

            related_preds = [{
                'label': label,
                'origin_label': origin_label,
                'origin': origin_pred,
                'origin_l': origin_pred[label],
                'origin_ol': origin_pred[origin_label],
                'masked': masked_pred,
                'masked_l': masked_pred[label],
                'masked_ol': masked_pred[origin_label],
                'maskimp': maskimp_pred,
                'maskimp_l': maskimp_pred[label],
                'maskimp_ol': maskimp_pred[origin_label],
                'retainimp':retainimp_pred,
                'retainimp_l':retainimp_pred[label],
                'retainimp_ol':retainimp_pred[origin_label],
                'sparsity_edges': sparsity_edges,
                'maskimp_nodes': maskimp_pred_nodes,
                'maskimp_nodes_l':maskimp_pred_nodes[label],
                'maskimp_nodes_ol':maskimp_pred_nodes[origin_label],
                'retainimp_nodes': retainimp_pred_nodes,
                'retainimp_nodes_l':retainimp_pred_nodes[label],
                'retainimp_nodes_ol':retainimp_pred_nodes[origin_label],
                'sparsity_nodes': sparsity_nodes
            }]
            related_preds_dict[top_k] = related_preds

        pred_mask = [edge_mask.cpu().detach().numpy()]
        return pred_mask, related_preds_dict


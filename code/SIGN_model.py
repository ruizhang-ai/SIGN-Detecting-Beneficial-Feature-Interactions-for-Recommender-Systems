import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import numpy as np


class L0_SIGN(nn.Module):
    def __init__(self, args, n_feature, device):
        super(L0_SIGN, self).__init__()

        self.pred_edges = args.pred_edges
        self.n_feature = n_feature
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.l0_para = eval(args.l0_para)
        self.device = device


        if self.pred_edges:
            self.linkpred = LinkPred(self.dim, self.hidden_layer, self.n_feature,  self.l0_para)
            self.linkpred = self.linkpred.to(self.device)

        self.sign = SIGN(self.dim, self.hidden_layer)
        self.sign = self.sign.to(self.device)
        self.g = torch.nn.Linear(self.dim, 2)  #2 is the class dimention 
        self.feature_emb = nn.Embedding(self.n_feature, self.dim)


    def forward(self, data, is_training=True):
        # does not conduct link prediction, use all interactions

        x, edge_index, sr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.feature_emb(x)
        x = x.squeeze(1)

        if self.pred_edges:
            sr = torch.transpose(sr, 0, 1)    # [2, num_edges]
            s, l0_penaty = self.linkpred(sr, is_training)
            pred_edge_index, pred_edge_weight = self.construct_pred_edge(edge_index, s, self.device) 
            updated_nodes = self.sign(x, pred_edge_index, edge_weight=pred_edge_weight)
            num_edges = pred_edge_weight.size(0)
        else:
            updated_nodes = self.sign(x, edge_index)
            l0_penaty = 0
            num_edges = edge_index.size(1)
        l2_penaty = (updated_nodes * updated_nodes).sum()
        graph_embedding = global_mean_pool(updated_nodes, batch)
        out = self.g(graph_embedding)
        return out, l0_penaty, l2_penaty, num_edges 

    def construct_pred_edge(self, fe_index, s, device):
        """
        fe_index: full_edge_index, [2, all_edges_batchwise]
        s: predicted edge value, [all_edges_batchwise, 1]

        construct the predicted edge set and corresponding edge weights
        """
        new_edge_index = [[],[]]
        edge_weight = []
        s = torch.squeeze(s)

        sender = torch.unsqueeze(fe_index[0][s>0], 0)
        receiver = torch.unsqueeze(fe_index[1][s>0], 0)
        pred_index = torch.cat((sender, receiver ), 0)
        pred_weight = s[s>0]

        return pred_index, pred_weight 


class SIGN(MessagePassing):
    def __init__(self, dim, hidden_layer):
        super(SIGN, self).__init__(aggr='mean')

        #construct pairwise modeling network
        self.lin1 = torch.nn.Linear(dim, hidden_layer)
        self.lin2 = torch.nn.Linear(hidden_layer, dim)
        self.act = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, dim]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, dim]
        # x_j has shape [E, dim]

        # pairwise analysis
        pairwise_analysis = self.lin1(x_i * x_j)
        pairwise_analysis = self.act(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)

        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis

        return interaction_analysis

    def update(self, aggr_out):
        # aggr_out has shape [N, dim]

        return aggr_out


class LinkPred(nn.Module):
    def __init__(self, D_in, H, n_feature, l0_para):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LinkPred, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        with torch.no_grad():
            #self.linear1.weight.copy_(self.linear1.weight + 0.2 )
            self.linear2.weight.copy_(self.linear2.weight + 0.2 )

        self.temp = l0_para[0]      #temprature
        self.inter_min = l0_para[1] 
        self.inter_max = l0_para[2] 
        self.feature_emb_edge = nn.Embedding(n_feature, D_in)    #D_in is the dimension size
        self.feature_emb_edge.weight.data.normal_(0.2,0.01)

    def forward(self, sender_receiver, is_training):
        #construct permutation input
        sender_emb = self.feature_emb_edge(sender_receiver[0,:])
        receiver_emb = self.feature_emb_edge(sender_receiver[1,:])
        _input =sender_emb * receiver_emb       #element wise product sender and receiver embeddings
        #loc = _input.sum(1)
        h_relu = self.dropout(self.relu(self.linear1(_input)))
        loc = self.linear2(h_relu)
        if is_training:
            u = torch.rand_like(loc)
            logu = torch.log2(u)
            logmu = torch.log2(1-u)
            sum_log = loc + logu - logmu
            s = torch.sigmoid(sum_log/self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min
        else:
            s = torch.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min

        s = torch.clamp(s, min=0, max=1)

        l0_penaty = torch.sigmoid(loc - self.temp * np.log2(-self.inter_min/self.inter_max)).mean()

        return s, l0_penaty 

    def permutate_batch_wise(x, batch):
        """
        x: all feature embeddings all batch
        batch: a list containing feature belongs to which graph
        """
        return


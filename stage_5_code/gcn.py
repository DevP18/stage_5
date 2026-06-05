import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return torch.spmm(adj, torch.mm(x, self.weight)) + self.bias


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1     = GraphConvolution(nfeat, nhid)
        self.gc2     = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.gc2(x, adj), dim=1)
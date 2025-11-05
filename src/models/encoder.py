from layers.gat import GraphAttentionLayer
from layers.gcn import GraphConvolutionLayer
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, layer_type='gcn'):
        super(Encoder, self).__init__()
        self.layer_type = layer_type.lower()
        if self.layer_type == 'gat':
            self.l1 = GraphAttentionLayer(nfeat, nhid)
            self.l2 = GraphAttentionLayer(nhid, nhid)
        else:
            self.l1 = GraphConvolutionLayer(nfeat, nhid)
            self.l2 = GraphConvolutionLayer(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        if self.layer_type == 'gat':
            x = F.elu(self.l1(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.l2(x, adj))
        else:
            x = F.relu(self.l1(x, adj, num_nodes=x.size(0)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.l2(x, adj, num_nodes=x.size(0)))
        return x
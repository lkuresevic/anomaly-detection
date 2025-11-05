from layers.gat import GraphAttentionLayer
from layers.gcn import GraphConvolutionLayer
import torch.nn as nn
import torch.nn.functional as F

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout, layer_type='gcn'):
        super(Structure_Decoder, self).__init__()
        self.layer_type = layer_type.lower()
        if self.layer_type == 'gat':
            self.l1 = GraphAttentionLayer(nhid, nhid)
        else:
            self.l1 = GraphConvolutionLayer(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        if self.layer_type == 'gat':
            x = F.elu(self.l1(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            struct_reconstructed = x @ x.t()
        else:
            x = F.relu(self.l1(x, adj, num_nodes=x.size(0)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            struct_reconstructed = x @ x.t()
        return struct_reconstructed
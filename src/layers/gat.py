import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn.inits import glorot

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, attn_drop=0.0, ffd_drop=0.0, residual=False, activation=F.elu):
        super(GraphAttentionLayer, self).__init__()
        self.activation = activation
        self.negative_slope = negative_slope
        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = Parameter(torch.Tensor(1, 2 * out_features)) # attention vector
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        
        self.residual = residual
        if residual:
            self.res_lin = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else nn.Identity()
        else:
            self.register_parameter('res_lin', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W.weight)
        glorot(self.a)
        if self.residual and not isinstance(self.res_lin, nn.Identity):
            glorot(self.res_lin.weight)

    def forward(self, h, adj):
        if not isinstance(adj, (tuple, list)):
             raise ValueError("GraphAttentionLayer requires (edge_index, edge_weight).")

        N = h.size(0)
        edge_index, _ = adj

        # Linear Transformation & Add Self-Loops
        h_lin = self.W(h)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        row, col = edge_index 
        
        # Calculate Attention Scores (e_ij)
        h_cat = torch.cat([h_lin[row], h_lin[col]], dim=1)
        e = self.leakyrelu(torch.matmul(h_cat, self.a.t())).squeeze(-1)

        # Sparse Softmax (over neighbors j for each i)
        alpha = softmax(e, row, num_nodes=N)
        
        alpha = self.attn_drop(alpha)

        # Message Aggregation (alpha_{i,j} * h_j)
        h_feat = self.ffd_drop(h_lin)
        
        # Aggregate features: sum_j alpha_{i,j} * h_feat[j]
        h_prime = torch.zeros(N, self.W.out_features, device=h.device, dtype=h.dtype)
        h_prime.index_add_(0, row, alpha.unsqueeze(-1) * h_feat[col])

        # Residual and Activation
        if self.residual:
            h_prime = h_prime + self.res_lin(h)

        return self.activation(h_prime)
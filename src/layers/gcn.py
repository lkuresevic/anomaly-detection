import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.inits import glorot, zeros


def build_dense_adj_from_edge_index(edge_index, edge_weight, num_nodes, device=None, dtype=None):
    if device is None:
        device = edge_index.device
    if dtype is None:
        dtype = torch.float32
    row, col = edge_index
    A = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    A[row, col] = 1.0 if edge_weight is None else edge_weight.to(dtype)
    return A


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj, num_nodes=None):
        N = x.size(0)
        
        # Prepare Adjacency Matrix (A) with self-loops
        if isinstance(adj, (tuple, list)):
            edge_index, edge_weight = add_self_loops(adj[0], adj[1], fill_value=1.0, num_nodes=N)
            A = build_dense_adj_from_edge_index(edge_index, edge_weight, N, device=x.device, dtype=x.dtype)
        else:
            A = adj.to(device=x.device, dtype=x.dtype)
            A = A + torch.eye(N, device=x.device, dtype=x.dtype) # Ensure self-loops

        # Symmetrically Normalize A
        deg = A.sum(dim=1).pow(-0.5)
        deg[deg == float('inf')] = 0.0
        D_inv_sqrt = torch.diag(deg)
        A_norm = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)

        # Graph Convolution
        support = torch.matmul(x, self.weight)
        out = torch.matmul(A_norm, support)
        
        if self.bias is not None:
            out = out + self.bias
        return out
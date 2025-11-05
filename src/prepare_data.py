import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[rowsum == 0] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def inject_anomalies(A, X, m=15, n=7, k=50):
    N = A.shape[0]
    A_mod = A.copy()
    X_mod = X.copy()

    anomaly_nodes = set()

    for _ in range(n):
        clique_nodes = np.random.choice(N, m, replace=False)
        anomaly_nodes.update(clique_nodes)
        for i in clique_nodes:
            for j in clique_nodes:
                if i != j:
                    A_mod[i, j] = 1
                    A_mod[j, i] = 1

    attr_anomaly_nodes = np.random.choice(N, m * n, replace=False)
    anomaly_nodes.update(attr_anomaly_nodes)


    for i in attr_anomaly_nodes:
        candidates = np.random.choice(N, k, replace=False)
        dists = euclidean_distances(X_mod[i].reshape(1, -1), X_mod[candidates]).flatten()
        j = candidates[np.argmax(dists)]
        X_mod[i] = X_mod[j]

    labels = np.zeros(N, dtype=np.int64)
    labels[list(anomaly_nodes)] = 1

    return A_mod, X_mod, labels

def prepare_data(datadir = 'raw_data', m  = 15, n = 7, k = 50, save_path='data'):

    for dataset_name in ['CiteSeer', 'Cora']:
        print(f"Loading {dataset_name} and injecting anomalies...")
        
        dataset = Planetoid(root=f'{datadir}/{dataset_name}', name=dataset_name)
        data = dataset[0]

        X = data.x.numpy()
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).toarray()

        A_mod, X_mod, labels_np = inject_anomalies(A, X, m=m, n=n, k=k)

        A_norm_sp = normalize_adj(sp.csr_matrix(A_mod + sp.eye(A_mod.shape[0])))
        edge_index, edge_weight = from_scipy_sparse_matrix(A_norm_sp)

        adj_label_np = (A_mod + np.eye(A_mod.shape[0]))

        anomalous_data = Data(
            x=torch.from_numpy(X_mod).float(),
            edge_index=edge_index,
            edge_weight=edge_weight.float(),
            y=data.y,
            anomaly_labels=torch.from_numpy(labels_np).long(),
            adj_label=torch.from_numpy(adj_label_np).float(),
        )
        
        if save_path:
            torch.save(anomalous_data, save_path+f'/{dataset_name}_anomalous_data.pt')
            print(f"Saved anomalous dataset to {save_path}")

    return

if __name__ == "__main__":
    prepare_data(datadir='raw_data', m=15, n=7, k=50, save_path='data')
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import csv
import os

from models.dominant import Dominant
from utils import load_anomalous_data, set_all_seeds
from utils import loss_func


def train_dominant(dataset="Cora", hidden_dim=64, epoch=101, lr=5e-3, dropout=0.3, alpha=0.8, layer_type='gcn'):
    data = load_anomalous_data(f"data/{dataset}_anomalous_data.pt")

    edge_index = data.edge_index
    edge_weight = data.edge_weight
    feat = data.x
    label = data.anomaly_labels
    adj_label = data.adj_label

    x = torch.FloatTensor(feat)
    adj_label = torch.FloatTensor(adj_label)
    
    x = x.float()
    adj_label = adj_label.float()
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    adj_label = adj_label.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    edge_index = edge_index.to(device)
    label = torch.tensor(label).to(device)

    model = Dominant(feat_size=x.size(1), hidden_size=hidden_dim, dropout=dropout, layer_type=layer_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    output_dir = "analysis/results"
    csv_filename = f"{dataset}_{layer_type}_training_results.csv"
    output_path = os.path.join(output_dir, csv_filename)

    os.makedirs(output_dir, exist_ok=True)

    metrics_data = []
    
    # Define CSV header
    fieldnames = ['Epoch', 'Total_Loss', 'Struct_Loss', 'Feat_Loss', 'AUC']

    # Write header to CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print("Starting training...\n")

    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()

        A_hat, X_hat = model(x, (edge_index, edge_weight))
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, x, X_hat, alpha)

        l = torch.mean(loss)
        l.backward()
        optimizer.step()

        current_metrics = {'Epoch': epoch} 
        
        if epoch % 5 == 0:
            total_loss_item = l.item()
            struct_loss_item = struct_loss.item()
            feat_loss_item = feat_loss.item()

            current_metrics.update({
                'Total_Loss': f"{total_loss_item:.5f}",
                'Struct_Loss': f"{struct_loss_item:.5f}",
                'Feat_Loss': f"{feat_loss_item:.5f}",
                'AUC': 'N/A' # Placeholder for AUC
            })
            
            print(f"Epoch: {epoch:03d} | "
                f"Total Loss: {total_loss_item:.5f} | "
                f"Struct Loss: {struct_loss_item:.5f} | "
                f"Feat Loss: {feat_loss_item:.5f}")

        # Evaluate and save AUC every 10 epochs (and the last epoch)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                A_hat, X_hat = model(x, (edge_index, edge_weight))
                loss_eval, _, _ = loss_func(adj_label, A_hat, x, X_hat, alpha)
                score = loss_eval.detach().cpu().numpy()
                
                try:
                    auc = roc_auc_score(label.cpu().numpy(), score)
                    auc_score_str = f"{auc:.4f}"
                    print(f"→ Epoch {epoch:03d} | AUC: {auc_score_str}")
                except ValueError:
                    auc_score_str = 'N/A'
                    print(f"→ Epoch {epoch:03d} | AUC: N/A (labels not binary)")

            current_metrics['AUC'] = auc_score_str
        
        if epoch % 5 == 0:
            if 'Total_Loss' not in current_metrics:
                current_metrics.update({'Total_Loss': 'N/A', 'Struct_Loss': 'N/A', 'Feat_Loss': 'N/A'})
            if 'AUC' not in current_metrics:
                current_metrics['AUC'] = 'N/A'
            metrics_data.append(current_metrics)

    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows(metrics_data)
    
    print(f"\nTraining completed. Results saved to {output_path}")
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results():
    datasets = ['Cora', 'CiteSeer']
    layer_types = ['gcn', 'gat']
    
    GCN_COLOR = '#1f77b4' 
    GAT_COLOR = '#ff7f0e' 

    data_dir = 'analysis/results'
    output_dir = 'analysis/plots'
    
    file_map = {
        ('Cora', 'gcn'): os.path.join(data_dir, 'Cora_gcn_training_results.csv'),
        ('Cora', 'gat'): os.path.join(data_dir, 'Cora_gat_training_results.csv'),
        ('CiteSeer', 'gcn'): os.path.join(data_dir, 'CiteSeer_gcn_training_results.csv'),
        ('CiteSeer', 'gat'): os.path.join(data_dir, 'CiteSeer_gat_training_results.csv'),
    }

    dataframes = {}
    
    for dataset in datasets:
        for layer_type in layer_types:
            key = (dataset, layer_type)
            try:
                df = pd.read_csv(file_map[key])
                # Convert 'AUC' column to numeric, errors='coerce' turns 'N/A' into NaN
                df['AUC'] = pd.to_numeric(df['AUC'], errors='coerce')
                dataframes[key] = df
            except FileNotFoundError:
                print(f"Warning: File not found for {dataset} {layer_type}. Skipping plots for this combination.")
                return # Exit if essential files are missing

    # Plot AUC Comparison
    print("Generating AUC comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('AUC Performance Comparison (GCN vs GAT)', fontsize=16)

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Plot GCN AUC
        df_gcn = dataframes[(dataset, 'gcn')].dropna(subset=['AUC'])
        ax.plot(df_gcn['Epoch'], df_gcn['AUC'], 
                label='GCN AUC', color=GCN_COLOR, linewidth=2, marker='s', markevery=1)
        
        # Plot GAT AUC
        df_gat = dataframes[(dataset, 'gat')].dropna(subset=['AUC'])
        ax.plot(df_gat['Epoch'], df_gat['AUC'], 
                label='GAT AUC', color=GAT_COLOR, linewidth=2, marker='^', markevery=1)
        
        ax.set_title(f'{dataset} Dataset AUC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ROC-AUC Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'AUC_Comparison.png'))
    print("AUC plots saved.")

    # Plot Loss Component Comparison
    print("Generating Loss component plots...")
    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f'{dataset} Dataset Loss Components (GCN vs GAT)', fontsize=16)

        for i, layer_type in enumerate(layer_types):
            ax = axes[i]
            df = dataframes[(dataset, layer_type)].dropna(subset=['Total_Loss'])
            color = GCN_COLOR if layer_type == 'gcn' else GAT_COLOR
            
            # Total Loss (Full Line)
            ax.plot(df['Epoch'], df['Total_Loss'], 
                    label='Total Loss', color=color, linestyle='-', linewidth=1.5)
            
            # Structural Loss (Triangles, not connected)
            ax.plot(df['Epoch'], df['Struct_Loss'], 
                    label='Structural Loss', color=color, marker='^', markersize=6, 
                    linestyle='None', fillstyle='none', markeredgewidth=1)
            
            # Feature/Attribute Loss (Circle points, not connected)
            ax.plot(df['Epoch'], df['Feat_Loss'], 
                    label='Attribute Loss', color=color, marker='o', markersize=6, 
                    linestyle='None', fillstyle='none', markeredgewidth=1)
            
            ax.set_title(f'{dataset} with {layer_type.upper()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'{dataset}_Loss_Components.png'))
    print("Loss plots saved.")


if __name__ == '__main__':
    plot_training_results()
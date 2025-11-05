# 📉 Anomaly Detection in Graphs

This project focuses on detecting **anomalies in graph-structured data** using Graph Neural Networks (GNNs). It includes implementations of both Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs).

## 🏗️ Project Structure

``
anomaly-detection/
├── analysis/
│   ├── results/        # Training results (e.g., loss curves, metrics in CSV format)
│   ├── plots/          # Generated plots (e.g., ROC curves, feature distributions)
│   └── plot_results.py # Script for generating and saving visualization plots
├── data/               # Processed datasets, ready for model input (including injected anomalies)
├── raw_data/           # Original, raw datasets (e.g., Cora, CiteSeer)
├── src/                # Core Python source code
│   ├── layers/         # GNN Layer Implementations
│   ├── models/         # Model Definitions
│   ├── prepare_data.py # Data preprocessing, graph manipulation, and anomaly injection logic
│   ├── train.py        # Core training and evaluation pipeline/functions
│   ├── main.py         # Primary entry point for executing training and testing
│   └── utils.py        # General utility and helper functions
├── .gitignore          # Files and directories to be ignored by Git
├── LICENSE             # Project license (MIT)
└── README.md           # Project documentation and setup guide
``

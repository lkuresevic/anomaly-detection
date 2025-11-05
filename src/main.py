from train import train_dominant
from utils import set_all_seeds

if __name__ == '__main__':
    set_all_seeds(42)
    datasets = ["Cora", "CiteSeer"]
    layer_types = ["gcn", "gat"]

    for dataset in datasets:
        for layer_type in layer_types:
            train_dominant(dataset=dataset, layer_type=layer_type)
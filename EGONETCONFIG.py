from torch_geometric.datasets import Planetoid, Reddit, KarateClub, SNAPDataset

#current_dataset = { "name": "Cora", "location": "Cora" }
current_dataset = { "name": "Karate Club", "location": "KarateClub" }
#current_dataset = { "name": "Citeseet", "location": "Citeseer" }
#current_dataset = { "name": "Pubmed", "location": "Pubmed" }
#current_dataset = { "name": "Amazon Computers", "location": "AmazonComputers" }
#current_dataset = { "name": "Amazon Photos", "location": "AmazonPhotos" }
#current_dataset = { "name": "Reddit", "location": "Reddit" }

hidden_sizes = ["in", 16, "out"]

layer_design = [
    ["Ego", "GCN"],
    [None, "GCN"]
]

test_nums_in = 1

epochs_in = 60

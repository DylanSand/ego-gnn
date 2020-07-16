from torch_geometric.datasets import Planetoid, Reddit, KarateClub, SNAPDataset

#current_dataset = { "name": "Cora", "location": "Cora" }
#current_dataset = { "name": "Karate Club", "location": "KarateClub" }
#current_dataset = { "name": "Citeseet", "location": "Citeseer" }
#current_dataset = { "name": "Pubmed", "location": "Pubmed" }
#current_dataset = { "name": "Amazon Computers", "location": "AmazonComputers" }
current_dataset = { "name": "Amazon Photos", "location": "AmazonPhotos" }
#current_dataset = { "name": "Reddit", "location": "Reddit" }

hidden_sizes = ["in", "out"]

layer_design = [
    ["Ego", "GCN"],
    [None, None]
]

test_nums_in = 4

train_mask_percent = 0.1

val_mask_percent = 0.4

burnout_num = 200

training_stop_limit = 8

epoch_limit = 600

relus = [True, False, False, False]

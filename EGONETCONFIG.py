from torch_geometric.datasets import Planetoid, Reddit, KarateClub, SNAPDataset

#current_dataset = { "name": "Cora", "location": "Cora" }
#current_dataset = { "name": "Karate Club", "location": "KarateClub" }
#current_dataset = { "name": "Citeseer", "location": "Citeseer" }
current_dataset = { "name": "Pubmed", "location": "Pubmed" }
#current_dataset = { "name": "Amazon Computers", "location": "AmazonComputers" }
#current_dataset = { "name": "Amazon Photos", "location": "AmazonPhotos" }
#current_dataset = { "name": "Reddit", "location": "Reddit" }
#current_dataset = { "name": "Flickr", "location": "Flickr" }

hidden_sizes = ["in", 16, "out"]

layer_design = [
    ["Ego", "GIN"],
    [None, "GIN"]
]

test_nums_in = 4

train_mask_percent = 0.1

val_mask_percent = 0.4

burnout_num = 100

training_stop_limit = 6

epoch_limit = 500

relus = [True, True, False, False]

from torch_geometric.datasets import Planetoid, Reddit, KarateClub, SNAPDataset

current_dataset = { "name": "Cora", "location": "Cora" }
#current_dataset = { "name": "Karate Club", "location": "KarateClub" }
#current_dataset = { "name": "Citeseer", "location": "Citeseer" }
#current_dataset = { "name": "Pubmed", "location": "Pubmed" }
#current_dataset = { "name": "Amazon Computers", "location": "AmazonComputers" }
#current_dataset = { "name": "Amazon Photos", "location": "AmazonPhotos" }
#current_dataset = { "name": "Reddit", "location": "Reddit" }
#current_dataset = { "name": "Flickr", "location": "Flickr" }
#current_dataset = { "name": "OGB Products", "location": "OGBProducts" }

hidden_sizes = ["in", 16, 32, "out"]

layer_design = [
    [None, "GCN", False, True],
    ["Ego", "GCN", True, True],
    [None, "GCN", False, False]
]

#        R        R         R       R    
#   [Ego -> GCN]       [None -> GCN  ]
# in      ->       16         ->        out

test_nums_in = 4

labeled_data = 0.5

val_split = 0.3

burnout_num = 100

training_stop_limit = 9

epoch_limit = 500

numpy_seed = 47

torch_seed = 74


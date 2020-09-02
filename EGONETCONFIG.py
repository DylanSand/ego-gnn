
#current_dataset = { "name": "Cora", "location": "Cora" }
#current_dataset = { "name": "Karate Club", "location": "KarateClub" }
current_dataset = { "name": "Citeseer", "location": "Citeseer" }
#current_dataset = { "name": "Pubmed", "location": "Pubmed" }
#current_dataset = { "name": "Amazon Computers", "location": "AmazonComputers" }
#current_dataset = { "name": "Amazon Photos", "location": "AmazonPhotos" }
#current_dataset = { "name": "Reddit", "location": "Reddit" }
#current_dataset = { "name": "Flickr", "location": "Flickr" }
#current_dataset = { "name": "OGB Products", "location": "OGBProducts" }
#current_dataset = { "name": "GitHub Network", "location": "GitHub" }

hidden_sizes = ["in", "out"]

layer_design = [
    ["Ego", "SAGE", True, False]
    #[None, "GCN", False, False]
]

local_depth = 1
local_power = 3

count_triangles = True
remove_features = False

#        R        R         R       R    
#   [Ego -> GCN]       [None -> GCN  ]
# in      ->       16         ->        out

test_nums_in = 4

labeled_data = 0.5

val_split = 0.3

burnout_num = 80

training_stop_limit = 9

epoch_limit = 300

learning_rate = 0.01

weight_decay = 1e-5

numpy_seed = 44

torch_seed = 74


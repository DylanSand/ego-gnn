print('Begin Python')
import torch
print('CUDA version is: ' + str(torch.version.cuda))
import sys
sys.path.insert(1, './utils')
sys.path.insert(1, './model')
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch.nn import ModuleList
from random import shuffle, randint
import networkx as nx
import matplotlib.pyplot as plt
import random 
from tqdm import tqdm
import pandas as pd
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, erdos_renyi_graph, to_networkx, to_undirected, subgraph, to_dense_adj, remove_self_loops
from torch_geometric.datasets import Amazon, Planetoid, Reddit, KarateClub, SNAPDataset, Flickr
from torch_geometric.data import NeighborSampler, Data, ClusterData, ClusterLoader
import torch.nn as nern
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, GATConv, GINConv, pool, SAGEConv
from helpers import has_num, reindex_edgeindex, get_adj, to_sparse
from ego_gnn import EgoGNN
from EGONETCONFIG import current_dataset, count_triangles, test_nums_in, labeled_data, val_split, burnout_num, training_stop_limit, epoch_limit, numpy_seed, torch_seed
import pickle
import wandb
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import f1_score
np.random.seed(numpy_seed)
torch.manual_seed(torch_seed)

DATASET = current_dataset['name']
print('We are using the dataset: ' + DATASET)
input_path = ''
job_id = ''
import getopt
opts, args = getopt.getopt(sys.argv[1:],"d:",["input_path="])
for opt, arg in opts:
    if opt in ("-d", "--input_path"):
        job_id = arg[11:-2]
        input_path = arg + '/' + current_dataset['location']
    else:
        sys.exit()
print(input_path)
print(job_id)

full_description = ''
with open('./EGONETCONFIG.py', 'r') as f:
    full_description = f.read()

wandb.init(name=current_dataset['name']+" - "+job_id, project="ego-net", notes=full_description)
 
# ---------------------------------------------------------------
print("Done 1")
wandb.log({'action': 'Done 1'})
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
real_data = None
if DATASET == "Karate Club":
    real_data = KarateClub()
elif DATASET == "Cora" or DATASET == "Citeseer" or DATASET == "Pubmed":
    real_data = Planetoid(root=input_path, name=DATASET)
elif DATASET == "Reddit":
    real_data = Reddit(root=input_path)
elif DATASET == "Amazon Computers":
    real_data = Amazon(root=input_path, name="Computers")
elif DATASET == "Amazon Photos":
    real_data = Amazon(root=input_path, name="Photo")
elif DATASET == "Flickr":
    real_data = Flickr(root=input_path)
elif DATASET == "OGB Products":
    real_data = PygNodePropPredDataset(name='ogbn-products')
    split_idx = real_data.get_idx_split()


# ---------------------------------------------------------------
print("Done 2")
wandb.log({'action': 'Done 2'})

graph = real_data[0]
#graph = five_data
graph.edge_index = to_undirected(graph.edge_index, graph.num_nodes)
graph.edge_index = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
graph.coalesce()
temp = NeighborSampler(edge_index=graph.edge_index, sizes=[-1])
batches = temp
egoNets = [0] * graph.num_nodes
adjMats = [0] * graph.num_nodes
plot = 331
curPlot = 0
for batch_size, n_id, adj in batches:
    curData = subgraph(n_id, graph.edge_index)
    updated_e_index = to_undirected(curData[0], n_id.shape[0])
    subgraph_size = torch.numel(n_id)
    cur_n_id = torch.sort(n_id)[0].tolist()
    cur_e_id = adj.e_id.tolist()
    subgraph2 = Data(edge_index=updated_e_index, edge_attr=curData[1], num_nodes=subgraph_size, n_id=cur_n_id, e_id=cur_e_id, adj=get_adj(updated_e_index, graph.edge_index, curPlot, cur_e_id))
    subgraph2.coalesce()
    egoNets[curPlot] = subgraph2
    curPlot = curPlot + 1

# ---------------------------------------------------------------
print("Done 3")
wandb.log({'action': 'Done 3'})

if count_triangles:
    num_triangles = [0] * len(egoNets)
    for i, ego in enumerate(egoNets):
        for edge_idx in range(len(ego.edge_index[0])):
            if not (ego.edge_index[0][edge_idx] == i or ego.edge_index[1][edge_idx] == i or ego.edge_index[0][edge_idx] == ego.edge_index[1][edge_idx]):
                num_triangles[i] = num_triangles[i] + 1
        num_triangles[i] = int(num_triangles[i] / 2)
    num_triangles = torch.tensor(num_triangles)
    graph.y = num_triangles


# ---------------------------------------------------------------
print("Done 4")
wandb.log({'action': 'Done 4'})

#train_loader = ClusterData(graph, num_parts=int(graph.num_nodes / 10), recursive=False)
#train_loader = ClusterLoader(train_loader, batch_size=2, shuffle=True, num_workers=12)

tests_acc = []
tests_f1_macro = []
tests_f1_micro = []
TEST_NUM = test_nums_in
BURNOUT = burnout_num
TRAINING_STOP_LIMIT = training_stop_limit
EPOCH_LIMIT = epoch_limit
for test in range(TEST_NUM):

    # CREATE DATA SPLITS:
    TRAIN_PERCENT = labeled_data - (labeled_data * val_split)
    VAL_PERCENT = labeled_data * val_split
    TEST_PERCENT = 1.0 - (TRAIN_PERCENT + VAL_PERCENT)
    cur_total = int(graph.num_nodes)
    num_train = int(float(int(cur_total)) * TRAIN_PERCENT)
    num_val = int(float(int(cur_total)) * VAL_PERCENT)
    train_mask = torch.tensor(np.array([True] * cur_total))
    val_mask = torch.tensor(np.array([False] * cur_total))
    test_mask = torch.tensor(np.array([False] * cur_total))
    chosen_not_train = np.random.choice(cur_total, cur_total - num_train, replace=False)
    for cur_index in chosen_not_train:
        train_mask[cur_index] = False
        test_mask[cur_index] = True
    chosen_val = np.random.choice(chosen_not_train, num_val, replace=False)
    for cur_index in chosen_val:
        val_mask[cur_index] = True
        test_mask[cur_index] = False
    if DATASET == "OGB Products":
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            if key == "train":
                train_mask = mask
            if key == "valid":
                val_mask = mask
            if key == "test":
                test_mask = mask

    # RUN MODEL:
    if DATASET == "Karate Club":
        model = EgoGNN(egoNets, device, 2, graph.x.shape[1]).to(device)
    else:
        model = EgoGNN(egoNets, device, real_data.num_classes, graph.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    cur_epoch = 1
    training_done = False
    best_score = None
    training_counter = 0
    wandb.watch(model)
    while(not training_done):
        print('Epoch: ' + str(cur_epoch))
        wandb.log({'action': 'Epoch: ' + str(cur_epoch)})
        torch.cuda.empty_cache()
        # USE NO BATCHING:
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = None
        if count_triangles:
            loss = F.mse_loss(out[train_mask], graph.y.to(device)[train_mask])
        else:
            loss = F.nll_loss(out[train_mask], graph.y.to(device)[train_mask])
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        # OR USE BATCHING:
        #for batch in train_loader:
        #    model.train()
        #    optimizer.zero_grad()
        #    out = model(batch.x, batch.edge_index)
        #    loss = F.nll_loss(out[train_mask], graph.y.to(device)[train_mask])
        #    loss.backward()
        #    optimizer.step()
        #    torch.cuda.empty_cache()
        if cur_epoch <= BURNOUT:
            print('     We are in BURNOUT')
            f = open("model.p", "wb")
            pickle.dump(model, f)
            f.close()
            model.eval()
            _, pred = model(graph.x, graph.edge_index).max(dim=1)
            correct = float (pred[val_mask].eq(graph.y.to(device)[val_mask]).sum().item())
            best_score = correct / val_mask.sum().item()
            wandb.log({'epoch': cur_epoch, 'val-accuracy': best_score})
        else:
            model.eval()
            _, pred = model(graph.x, graph.edge_index).max(dim=1)
            correct = float (pred[val_mask].eq(graph.y.to(device)[val_mask]).sum().item())
            cur_acc = correct / val_mask.sum().item()
            wandb.log({'epoch': cur_epoch, 'val-accuracy': cur_acc})
            print('     Current Acc: ' + str(cur_acc))
            print('     Best Acc:    ' + str(best_score))
            if cur_epoch > EPOCH_LIMIT:
                print('          We have hit the epoch limit.')
                training_done = True
            if cur_acc < best_score:
                print('          We did worse.')
                print('          Current counter: ' + str(training_counter))
                if training_counter > TRAINING_STOP_LIMIT:
                    print('          We are done now.')
                    f = open("model.p", "rb")
                    model = pickle.load(f)
                    f.close()
                    training_done = True
                else:
                    training_counter = training_counter + 1
            else:
                print('          We did better!')
                f = open("model.p", "wb")
                pickle.dump(model, f)
                f.close()
                training_counter = 0
                best_score = cur_acc
        cur_epoch = cur_epoch + 1
    model.eval()
    _, pred = model(graph.x, graph.edge_index).max(dim=1)
    correct = float (pred[test_mask].eq(graph.y.to(device)[test_mask]).sum().item())
    acc = correct / test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
    tests_acc.append(acc)
    macro_score = f1_score(graph.y[test_mask], pred[test_mask].cpu(), average='macro')
    print('Macro score is: ' + str(macro_score))
    micro_score = f1_score(graph.y[test_mask], pred[test_mask].cpu(), average='micro')
    print('Micro score is: ' + str(micro_score))
    tests_f1_macro.append(macro_score)
    tests_f1_micro.append(micro_score)
print('Average accuracy of ' + str(len(tests_acc)) + ' tests is: ' + str(sum(tests_acc) / len(tests_acc)))
print('Average F1 macro score of ' + str(len(tests_f1_macro)) + ' tests is: ' + str(sum(tests_f1_macro) / len(tests_f1_macro)))
print('Average F1 micro score of ' + str(len(tests_f1_micro)) + ' tests is: ' + str(sum(tests_f1_micro) / len(tests_f1_micro)))

torch.save(model.state_dict(), osp.join(wandb.run.dir, 'model.p'))

print('Model configuration:')
with open('./EGONETCONFIG.py', 'r') as f:
    print(f.read())

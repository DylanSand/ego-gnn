print('Begin Python')
import torch
print('Version is: ' + str(torch.version.cuda))
DATASET = 'Cora'
input_path = ''
import sys
import getopt
opts, args = getopt.getopt(sys.argv[1:],"d:",["input_path="])
for opt, arg in opts:
    if opt in ("-d", "--input_path"):
       input_path = arg + '/' + DATASET
    else:
       sys.exit()
print(input_path)

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
from torch_geometric.datasets import Planetoid, Reddit, KarateClub, SNAPDataset
from torch_geometric.data import NeighborSampler, Data, ClusterData, ClusterLoader
import torch.nn as nern
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, GATConv, GINConv, pool, SAGEConv
from ./utils/helpers import has_num, reindex_edgeindex, get_adj, to_sparse
from ./model/ego_gnn import EgoGNN
 
# ---------------------------------------------------------------
print("Done 1")
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#real_data = KarateClub()
real_data = Planetoid(root=input_path, name=DATASET) 

# ---------------------------------------------------------------
print("Done 2")
 
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
 
TRAIN_PERCENT = 0.1
cur_total = int(graph.num_nodes)
num_train = int(float(int(cur_total)) * TRAIN_PERCENT)
train_mask = torch.tensor(np.array([False] * cur_total))
test_mask = torch.tensor(np.array([True] * cur_total))
for chosen in np.random.choice(cur_total, num_train):
    train_mask[chosen] = True
    test_mask[chosen] = False
 
# ---------------------------------------------------------------
print("Done 4")

#train_loader = ClusterData(graph, num_parts=int(graph.num_nodes / 10), recursive=False)
#train_loader = ClusterLoader(train_loader, batch_size=2, shuffle=True, num_workers=12)

tests = []
TEST_NUM = 2
EPOCH_NUM = 100
for test in range(TEST_NUM):
    model = EgoGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for epoch in range(EPOCH_NUM):
        print(epoch)
        torch.cuda.empty_cache()

        # USE NO BATCHING:
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
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
    model.eval()
    _, pred = model(graph.x, graph.edge_index).max(dim=1)
    correct = float (pred[test_mask].eq(graph.y.to(device)[test_mask]).sum().item())
    acc = correct / test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
    tests.append(acc)
print('Average is: ' + str(sum(tests) / len(tests)))

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
 
# ---------------------------------------------------------------
print("Done 1")
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#real_data = KarateClub()
real_data = Planetoid(root=input_path, name=DATASET) 
 
# ---------------------------------------------------------------
print("Done 2")
 
def has_num(tensor, num):
    temp = tensor.view(1, -1)
    if(torch.numel((temp[0] == int(num)).nonzero()) == 0):
        return False
    else:
        return True
def reindex_edgeindex(edge_index, num_nodes):
    curOn = 0
    curMax = int(torch.max(edge_index))
    while(not curMax == num_nodes - 1):
        while(has_num(edge_index, curOn)):
            curOn += 1
        edge_index[edge_index == curMax] = curOn
        curMax = int(torch.max(edge_index))
        curOn += 1
    return edge_index
def get_adj(edge_index, global_edge_index, cur_node, e_id):
  n_id = []
  for edge in e_id:
    n_id.append(int(global_edge_index[0][edge]))
  n_id.append(cur_node)
  num_nodes = len(n_id)
  nodeDict = {}
  for i, node in enumerate(n_id):
    nodeDict[node] = i
  adj = torch.eye(n=num_nodes, dtype=torch.float)
  row, col = edge_index
  nRow = row.tolist()
  nCol = col.tolist()
  for i, node in enumerate(nRow):
    adj[nodeDict[node]][nodeDict[nCol[i]]] = 1
  return adj
def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
 
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
 
# ---------------------------------------------------------------
print("Done 3")
 
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
print("Done 4")
 
TRAIN_PERCENT = 0.1
cur_total = int(graph.num_nodes)
num_train = int(float(int(cur_total)) * TRAIN_PERCENT)
train_mask = torch.tensor(np.array([False] * cur_total))
test_mask = torch.tensor(np.array([True] * cur_total))
for chosen in np.random.choice(cur_total, num_train):
    train_mask[chosen] = True
    test_mask[chosen] = False
 
# ---------------------------------------------------------------
print("Done 5")

train_loader = ClusterData(graph, num_parts=int(graph.num_nodes / 10), recursive=False)
train_loader = ClusterLoader(train_loader, batch_size=2, shuffle=True, num_workers=12)
 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1Inter = GINConv(torch.nn.Linear(graph.x.shape[1], graph.x.shape[1]))
        self.conv1Intra = GINConv(torch.nn.Linear(graph.x.shape[1], 16))
        self.conv2Inter = GINConv(torch.nn.Linear(16, 16))
        self.conv2Intra = GINConv(torch.nn.Linear(16, real_data.num_classes))

    def do_conv(self, x, convInter, convIntra, edge_index_in):
        output_temp = convInter(x.data, egoNets[0].edge_index.to(device).data).data
        for i, ego in enumerate(egoNets):
            if i == 0:
                continue
            output_temp = output_temp + convInter(x.data, ego.edge_index.to(device).data).data
            torch.cuda.empty_cache()
        output_temp = output_temp * (1 / len(egoNets))
        output = output_temp.to(device)
        #output = convIntra(output, edge_index_in.to(device).data)
        torch.cuda.empty_cache()
        return output
 
    def forward(self, x_in, edge_index_in):
        x = x_in.to(device)
        x = x + self.do_conv(x, self.conv1Inter, self.conv1Intra, edge_index_in)
        x = self.conv1Intra(x, edge_index_in.to(device))
        torch.cuda.empty_cache()
        x = F.relu(x)
        x = x + self.do_conv(x, self.conv2Inter, self.conv2Intra, edge_index_in)
        x = self.conv2Intra(x, edge_index_in.to(device))
        torch.cuda.empty_cache()
        return F.log_softmax(x, dim=1)
tests = []
TEST_NUM = 2
EPOCH_NUM = 100
for test in range(TEST_NUM):
    model = Net().to(device)
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

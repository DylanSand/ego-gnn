import torch
import json
import pandas as pd
import networkx as nx

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

def load_graph(graph_path):
    """
    Reading a NetworkX graph.
    :param graph_path: Path to the edge list.
    :return graph: NetworkX object.
    """
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def load_targets(target_path):
    data = pd.read_csv(target_path).values.tolist()
    targets = [int(target[2]) for target in data]
    return targets

def load_features(features_path):
    """
    Reading the features from disk.
    :param features_path: Location of feature JSON.
    :return features: Feature hash table.
    """
    features = json.load(open(features_path))
    features = [[float(val) for val in v] for k, v in features.items()]
    max_length = 0
    for feats in features:
        feats.sort(reverse=True)
        if len(feats) > max_length:
            max_length = len(feats)
    for feats in features:
        while len(feats) < max_length:
            feats.append(0)
    return features

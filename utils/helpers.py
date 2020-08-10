import torch

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

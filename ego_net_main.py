print('Begin Python')
import torch
print('CUDA version is: ' + str(torch.version.cuda))
import sys
sys.path.insert(1, './utils')
sys.path.insert(1, './model')
import numpy as np
import math
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
from torch_geometric.utils import add_self_loops, stochastic_blockmodel_graph, degree, erdos_renyi_graph, to_networkx, from_networkx, to_undirected, subgraph, to_dense_adj, remove_self_loops
from torch_geometric.datasets import Amazon, GNNBenchmarkDataset, Planetoid, Reddit, KarateClub, SNAPDataset, Flickr
from torch_geometric.data import NeighborSampler, Data, ClusterData, ClusterLoader
import torch.nn as nern
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, GATConv, GINConv, pool, SAGEConv
from helpers import has_num, reindex_edgeindex, get_adj, to_sparse, load_graph, load_features, load_targets
from ego_gnn import EgoGNN
from EGONETCONFIG import current_dataset, sbm_noise, save_data, load_data, count_triangles, test_nums_in, labeled_data, val_split, learning_rate, weight_decay, burnout_num, training_stop_limit, epoch_limit, numpy_seed, torch_seed, remove_features
import pickle
import wandb
from torch_sparse import spspmm
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
elif DATASET == "CLUSTER":
    real_data = GNNBenchmarkDataset(root=input_path, name="CLUSTER", split="test")
elif DATASET == "PATTERN":
    real_data = GNNBenchmarkDataset(root=input_path, name="PATTERN", split="test")
elif DATASET == "Flickr":
    real_data = Flickr(root=input_path)
elif DATASET == "OGB Products":
    real_data = PygNodePropPredDataset(name='ogbn-products')
    split_idx = real_data.get_idx_split()
elif DATASET == "GitHub Network":
    gitGraph = from_networkx(load_graph(input_path + '/musae_git_edges.csv'))
    gitGraph.x = torch.tensor(load_features(input_path + '/musae_git_features.json'))
    gitGraph.y = torch.tensor(load_targets(input_path + '/musae_git_target.csv'))
elif DATASET == "SBM":

    # Size of blocks
    COMMUNITY_SIZE = 100

    # Number of clusters
    NUM_BLOCKS = 10

    # In-Block prob.
    INTER_PROB = 0.90

    # Between-block prob.
    INTRA_PROB = 0.10

    # Connecting with noise prob
    NOISE_PROB = 0.85

    NUM_NOISE = int(sbm_noise * COMMUNITY_SIZE * NUM_BLOCKS)
    real_data = []
    block_sizes = [COMMUNITY_SIZE] * NUM_BLOCKS
    block_sizes = block_sizes + ([1] * NUM_NOISE)
    edge_probs = []
    for x in range(NUM_BLOCKS + NUM_NOISE):
        cur_probs = []
        for y in range(NUM_BLOCKS + NUM_NOISE):
            cur_prob = 0.0
            if x == y:
                cur_prob = INTER_PROB
            elif x >= NUM_BLOCKS or y >= NUM_BLOCKS:
                cur_prob = NOISE_PROB
            else:
                cur_prob = INTRA_PROB
            cur_probs.append(float(cur_prob))
        edge_probs.append(cur_probs)
    sbm_ei = stochastic_blockmodel_graph(block_sizes, edge_probs, directed=False)
    sbm_x = []
    for x in range((COMMUNITY_SIZE * NUM_BLOCKS) + NUM_NOISE):
        sbm_x.append([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    sbm_x = torch.tensor(sbm_x)
    sbm_y = [int(x / COMMUNITY_SIZE) for x in range(COMMUNITY_SIZE * NUM_BLOCKS)]
    sbm_y = sbm_y + [int(x + NUM_BLOCKS) for x in range(NUM_NOISE)]
    sbm_y = torch.tensor(sbm_y)
    print(sbm_ei)
    print(sbm_x)
    print(sbm_y)
    real_data.append(Data(x=sbm_x , edge_index=sbm_ei, y=sbm_y, num_nodes=int((COMMUNITY_SIZE * NUM_BLOCKS) + NUM_NOISE)))

egoNets = None
graph = None
norm_degrees = None
if load_data:
    extra = ""
    if count_triangles:
        extra = "tri"
    f = open(DATASET+"egoNets"+extra+".p", "rb")
    egoNets = pickle.load(f)
    f.close()
    f = open(DATASET+"graph"+extra+".p", "rb")
    graph = pickle.load(f)
    f.close()
    f = open(DATASET+"norm_degrees"+extra+".p", "rb")
    norm_degrees = pickle.load(f)
    f.close()

# ---------------------------------------------------------------
print("Done 2")
wandb.log({'action': 'Done 2'})
if not load_data:
    if DATASET == "GitHub Network":
        graph = gitGraph
    else:
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
    norm_degrees = []
    for batch_size, n_id, adj in batches:
        curData = subgraph(n_id, graph.edge_index)
        updated_e_index = to_undirected(curData[0], n_id.shape[0])
        subgraph_size = torch.numel(n_id)
        cur_n_id = torch.sort(n_id)[0].tolist()
        cur_e_id = adj.e_id.tolist()
        subgraph2 = Data(edge_index=updated_e_index, edge_attr=curData[1], num_nodes=subgraph_size, n_id=cur_n_id, e_id=cur_e_id, degree=len(cur_n_id)-1, adj=get_adj(updated_e_index, graph.edge_index, curPlot, cur_e_id))
        subgraph2.coalesce()
        ######################
        ego_degrees = {}
        for edge in subgraph2.edge_index[1]:
            cur_edge = int(edge)
            if str(cur_edge) in ego_degrees:
                ego_degrees[str(cur_edge)] = ego_degrees[str(cur_edge)] + 1
            else:
                ego_degrees[str(cur_edge)] = 1
        ######################
        #ego_n_degrees = []
        #for edge in subgraph2.edge_index[1]:
        #    cur_edge = int(edge)
        #    ego_n_degrees.append(float(1 / float(ego_degrees[str(cur_edge)])))
        #ego_n_degrees = torch.tensor(ego_n_degrees)
        #ego_n_degrees = torch.reshape(ego_n_degrees, (subgraph2.edge_index.shape[1],))
        #subgraph2.ego_degrees = ego_n_degrees
        ego_norm_ind = [[],[]]
        ego_norm_val = []
        # UNDO THIS:
        for inner_node in subgraph2.n_id:
            ego_norm_ind[0].append(int(inner_node))
            ego_norm_ind[1].append(int(inner_node))
            ego_norm_val.append(1.0 / math.sqrt(float(ego_degrees[str(inner_node)])))
        ego_norm_ind = torch.tensor(ego_norm_ind)
        ego_norm_val = torch.tensor(ego_norm_val)
        temp1, temp2 = spspmm(ego_norm_ind, ego_norm_val, subgraph2.edge_index, torch.ones((subgraph2.edge_index.shape[1])), graph.num_nodes, graph.num_nodes, graph.num_nodes)
        ego_norm_ind, ego_norm_val = spspmm(temp1, temp2, ego_norm_ind, ego_norm_val, graph.num_nodes, graph.num_nodes, graph.num_nodes)
        subgraph2.ego_norm_ind = ego_norm_ind
        subgraph2.ego_norm_val = ego_norm_val
        temp1 = None
        temp2 = None
        ego_degrees = None
        egoNets[curPlot] = subgraph2
        norm_degrees.append(1.0 / float(subgraph2.degree + 1))
        curPlot = curPlot + 1
    norm_degrees = torch.reshape(torch.tensor(norm_degrees), (len(egoNets), 1))

# ---------------------------------------------------------------
print("Done 3")
wandb.log({'action': 'Done 3'})

if count_triangles and not load_data:
    num_triangles = [0] * len(egoNets)
    clustering_coeff = [float(0)] * len(egoNets)
    for i, ego in enumerate(egoNets):
        for edge_idx in range(len(ego.edge_index[0])):
            if not (ego.edge_index[0][edge_idx] == i or ego.edge_index[1][edge_idx] == i or ego.edge_index[0][edge_idx] == ego.edge_index[1][edge_idx]):
                num_triangles[i] = num_triangles[i] + 1
        num_triangles[i] = float(num_triangles[i] / 2)
        if not (ego.degree == 0 or ego.degree == 1):
            clustering_coeff[i] = float((2.0 * num_triangles[i]) / (ego.degree * (ego.degree - 1)))
    #wandb.log({'triangle-distribution': wandb.Histogram(np_histogram=np.histogram([int(tri) for tri in num_triangles],bins=list(range(int(max(num_triangles)+1)))))})
    num_triangle_classes = int(max(num_triangles)+1)
    plt.hist([int(tri) for tri in num_triangles],bins=int(max(num_triangles)+1),range=(0,max(num_triangles)))
    plt.xlabel('Triangle Number')
    plt.ylabel('Amount of Nodes')
    plt.title('Triangle Distribution')
    wandb.log({'triangle-distribution': wandb.Image(plt)})
    plt.clf()
    plt.close()
    plt.hist([int(tri) for tri in num_triangles],bins=int(max(num_triangles)+1),range=(1,max(num_triangles)))
    plt.xlabel('Triangle Number (Pruned)')
    plt.ylabel('Amount of Nodes')
    plt.title('Triangle Distribution')
    wandb.log({'triangle-distribution-pruned': wandb.Image(plt)})
    plt.clf()
    plt.close()
    plt.hist([int(tri) for tri in num_triangles],bins=int(max(num_triangles)+1),range=(0,max(num_triangles)),log=True)
    plt.xlabel('Triangle Number')
    plt.ylabel('Amount of Nodes')
    plt.title('Triangle Distribution (Log)')
    wandb.log({'triangle-distribution-log': wandb.Image(plt)})
    plt.clf()
    plt.close()
    plt.hist([int(tri) for tri in num_triangles],bins=int(max(num_triangles)+1),range=(1,max(num_triangles)),log=True)
    plt.xlabel('Triangle Number')
    plt.ylabel('Amount of Nodes')
    plt.title('Triangle Distribution (Pruned + Log)')
    wandb.log({'triangle-distribution-pruned-log': wandb.Image(plt)})
    plt.clf()
    plt.close()
    num_divisions = 12
    bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
    bins.reverse()
    num_in_bin = [0] * len(egoNets)
    for i, ego in enumerate(egoNets):
        for ind, bucket in enumerate(bins):
            if clustering_coeff[i] >= bucket:
                num_in_bin[i] = bucket
                break
    bins.reverse()
    plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Amount of Nodes')
    plt.title('CC Distribution (Log) Full Truth')
    wandb.log({'cc-distribution-log-full-truth': wandb.Image(plt)})
    plt.clf()
    plt.close()

    #import plotly.graph_objects as go
    #bins=[str(label) for label in list(range(int(max(num_triangles)+1)))]
    #heights=[0] * int(max(num_triangles)+1)
    #for tri in num_triangles:
    #    heights[int(tri)] = heights[int(tri)] + 1
    #fig = go.Figure([go.Bar(x=bins, y=heights)])
    #wandb.log({'triangle-distribution': fig})

    num_triangles = torch.tensor(num_triangles)
    clustering_coeff = torch.tensor(clustering_coeff)
    graph.y = clustering_coeff
    #graph.y = num_triangles

    print('Average clustering coefficient is: ' + str(float(torch.mean(clustering_coeff))))
    wandb.log({'cluster-avg': float(torch.mean(clustering_coeff))})
elif load_data and count_triangles:
    clustering_coeff = graph.y

if remove_features and not load_data:
    new_features = []
    for node_i in range(len(egoNets)):
        #new_features.append([float(num+1) for num in range(graph.x.shape[1])])
        new_features.append([float(1) for num in range(graph.x.shape[1])])
    #for new_feat in new_features:
    #    random.Random(random.random()).shuffle(new_feat)
    graph.x = torch.tensor(new_features)

# ---------------------------------------------------------------
print("Done 4")
wandb.log({'action': 'Done 4'})

if save_data:
    extra = ""
    if count_triangles:
        extra = "tri"
    f = open(DATASET+"egoNets"+extra+".p", "wb")
    pickle.dump(egoNets, f)
    f.close()
    f = open(DATASET+"graph"+extra+".p", "wb")
    pickle.dump(graph, f)
    f.close()
    f = open(DATASET+"norm_degrees"+extra+".p", "wb")
    pickle.dump(norm_degrees, f)
    f.close()
    #torch.save(model.state_dict(), osp.join(wandb.run.dir, DATASET+"egoNets"+extra+".p"))
    #torch.save(model.state_dict(), osp.join(wandb.run.dir, DATASET+"graph"+extra+".p"))
    #torch.save(model.state_dict(), osp.join(wandb.run.dir, DATASET+"norm_degrees"+extra+".p"))
    wandb.save(DATASET+"egoNets"+extra+".p")
    wandb.save(DATASET+"graph"+extra+".p")
    wandb.save(DATASET+"norm_degrees"+extra+".p")

#train_loader = ClusterData(graph, num_parts=int(graph.num_nodes / 10), recursive=False)
#train_loader = ClusterLoader(train_loader, batch_size=2, shuffle=True, num_workers=12)

tests_acc = []
tests_clus_avg = []
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
    if DATASET == "SBM":
        for noise in range(NUM_NOISE):
            train_mask[(NUM_BLOCKS * COMMUNITY_SIZE) + noise] = False
            val_mask[(NUM_BLOCKS * COMMUNITY_SIZE) + noise] = False
            test_mask[(NUM_BLOCKS * COMMUNITY_SIZE) + noise] = False
    if count_triangles:
        for i, ego in enumerate(egoNets):
            if ego.degree == 0 or ego.degree == 1:
                train_mask[i] = False
                val_mask[i] = False
                test_mask[i] = False
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(clustering_coeff[train_mask])
        for i, ego in enumerate(clustering_coeff[train_mask]):
            for ind, bucket in enumerate(bins):
                if float(clustering_coeff[train_mask][i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Training Set Truth')
        wandb.log({'cc-distribution-log-train-set-truth': wandb.Image(plt)})
        plt.clf()
        plt.close()
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(clustering_coeff[val_mask])
        for i, ego in enumerate(clustering_coeff[val_mask]):
            for ind, bucket in enumerate(bins):
                if float(clustering_coeff[val_mask][i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Validation Set Truth')
        wandb.log({'cc-distribution-log-val-set-truth': wandb.Image(plt)})
        plt.clf()
        plt.close()
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(clustering_coeff[test_mask])
        for i, ego in enumerate(clustering_coeff[test_mask]):
            for ind, bucket in enumerate(bins):
                if float(clustering_coeff[test_mask][i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Test Set Truth')
        wandb.log({'cc-distribution-log-test-set-truth': wandb.Image(plt)})
        plt.clf()
        plt.close()

    # RUN MODEL:
    if count_triangles:
        model = EgoGNN(egoNets, device, 1, graph.x.shape[1], norm_degrees).to(device)
        #model = EgoGNN(egoNets, device, num_triangles_classes, graph.x.shape[1], norm_degrees).to(device)
    elif DATASET == "Karate Club" or DATASET == "GitHub Network":
        model = EgoGNN(egoNets, device, 2, graph.x.shape[1], norm_degrees).to(device)
    elif DATASET == "SBM":
        model = EgoGNN(egoNets, device, NUM_BLOCKS, graph.x.shape[1], norm_degrees).to(device)
    else:
        model = EgoGNN(egoNets, device, real_data.num_classes, graph.x.shape[1], norm_degrees).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            loss = F.mse_loss(out[train_mask], graph.y.to(device).view(-1, 1)[train_mask])
            training_loss = torch.mean(loss)
            #training_loss = torch.mean(F.mse_loss(torch.exp(out[train_mask])-1, clustering_coeff.to(device).view(-1, 1)[train_mask]))
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
            f = open("model"+str(test+1)+".p", "wb")
            pickle.dump(model, f)
            f.close()
            model.eval()
            if count_triangles:
                pred = model(graph.x, graph.edge_index)
                #best_score = torch.mean(F.mse_loss(torch.exp(pred[val_mask])-1, clustering_coeff.to(device).view(-1, 1)[val_mask]))
                best_score = torch.mean(F.mse_loss(pred[val_mask], graph.y.to(device)[val_mask].view(-1, 1)))
                #correct = torch.round(pred)[val_mask].eq(graph.y.to(device).view(-1, 1)[val_mask])
                #best_score = float(correct.sum().item()) / float(val_mask.sum().item())
                wandb.log({'epoch'+str(test+1): cur_epoch, 'val-loss'+str(test+1): best_score, 'train-loss'+str(test+1): training_loss})
            else:
                _, pred = model(graph.x, graph.edge_index).max(dim=1)
                correct = float (pred[val_mask].eq(graph.y.to(device)[val_mask]).sum().item())
                best_score = correct / val_mask.sum().item()
                correct_train = float (pred[train_mask].eq(graph.y.to(device)[train_mask]).sum().item())
                cur_acc_train = correct_train / train_mask.sum().item()
                wandb.log({'epoch'+str(test+1): cur_epoch, 'val-accuracy'+str(test+1): best_score, 'train-accuracy'+str(test+1): cur_acc_train})
        else:
            model.eval()
            cur_acc = 0
            if count_triangles:
                pred = model(graph.x, graph.edge_index)
                #cur_acc = torch.mean(F.mse_loss(torch.exp(pred[val_mask])-1, clustering_coeff.to(device).view(-1, 1)[val_mask]))
                cur_acc = torch.mean(F.mse_loss(pred[val_mask], graph.y.to(device)[val_mask].view(-1, 1)))
                #correct = torch.round(pred)[val_mask].eq(graph.y.to(device).view(-1, 1)[val_mask])
                #cur_acc = float(correct.sum().item()) / float(val_mask.sum().item())
                wandb.log({'epoch'+str(test+1): cur_epoch, 'val-loss'+str(test+1): cur_acc, 'train-loss'+str(test+1): training_loss})
                print('     Current Val Loss: ' + str(float(cur_acc)))
                print('     Best Val Loss:    ' + str(float(best_score)))
            else:
                _, pred = model(graph.x, graph.edge_index).max(dim=1)
                correct = float (pred[val_mask].eq(graph.y.to(device)[val_mask]).sum().item())
                cur_acc = correct / val_mask.sum().item()
                correct_train = float (pred[train_mask].eq(graph.y.to(device)[train_mask]).sum().item())
                cur_acc_train = correct_train / train_mask.sum().item()
                wandb.log({'epoch'+str(test+1): cur_epoch, 'val-accuracy'+str(test+1): cur_acc, 'train-accuracy'+str(test+1): cur_acc_train})
                print('     Current Val Acc: ' + str(cur_acc))
                print('     Best Val Acc:    ' + str(best_score))
            if cur_epoch > EPOCH_LIMIT:
                print('          We have hit the epoch limit.')
                training_done = True
            if (not count_triangles and cur_acc < best_score) or (count_triangles and cur_acc > best_score):
                print('          We did worse.')
                print('          Current counter: ' + str(training_counter))
                if training_counter > TRAINING_STOP_LIMIT:
                    print('          We are done now.')
                    f = open("model"+str(test+1)+".p", "rb")
                    model = pickle.load(f)
                    f.close()
                    training_done = True
                else:
                    training_counter = training_counter + 1
            else:
                print('          We did better!')
                f = open("model"+str(test+1)+".p", "wb")
                pickle.dump(model, f)
                f.close()
                training_counter = 0
                best_score = cur_acc
        cur_epoch = cur_epoch + 1
    f = open("model"+str(test+1)+".p", "rb")
    model = pickle.load(f)
    f.close()
    model.eval()
    acc = 0
    if count_triangles:
        pred = model(graph.x, graph.edge_index)
        #acc = torch.mean(F.mse_loss(torch.exp(pred[test_mask])-1, clustering_coeff.to(device).view(-1, 1)[test_mask]))
        acc = float(torch.mean(F.mse_loss(pred[test_mask], graph.y.to(device)[test_mask].view(-1, 1))))
        avg_pred = float(torch.mean(pred))
        tests_clus_avg.append(avg_pred)
        #correct = torch.round(pred)[test_mask].eq(graph.y.to(device).view(-1, 1)[test_mask])
        #acc = float(correct.sum().item()) / float(test_mask.sum().item())
        print('Test Loss: {:.4f}'.format(acc))
        print('Test Avg Clus.: {:.4f}'.format(avg_pred))
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(pred)
        for i, ego in enumerate(pred):
            for ind, bucket in enumerate(bins):
                if float(pred[i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Full '+str(test+1))
        wandb.log({'cc-distribution-log-full-'+str(test+1): wandb.Image(plt)})
        plt.clf()
        plt.close()
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(pred[train_mask])
        for i, ego in enumerate(pred[train_mask]):
            for ind, bucket in enumerate(bins):
                if float(pred[train_mask][i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Training Set '+str(test+1))
        wandb.log({'cc-distribution-log-train-set-'+str(test+1): wandb.Image(plt)})
        plt.clf()
        plt.close()
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(pred[val_mask])
        for i, ego in enumerate(pred[val_mask]):
            for ind, bucket in enumerate(bins):
                if float(pred[val_mask][i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Validation Set '+str(test+1))
        wandb.log({'cc-distribution-log-val-set-'+str(test+1): wandb.Image(plt)})
        plt.clf()
        plt.close()
        num_divisions = 12
        bins = np.arange(0.0, 1.0, 1.0/num_divisions).tolist()
        bins.reverse()
        num_in_bin = [0] * len(pred[test_mask])
        for i, ego in enumerate(pred[test_mask]):
            for ind, bucket in enumerate(bins):
                if float(pred[test_mask][i]) >= bucket:
                    num_in_bin[i] = bucket
                    break
        bins.reverse()
        plt.hist(num_in_bin,bins=len(bins),range=(0.0,1.0),log=True)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Amount of Nodes')
        plt.title('CC Distribution (Log) Test Set '+str(test+1))
        wandb.log({'cc-distribution-log-test-set-'+str(test+1): wandb.Image(plt)})
        plt.clf()
        plt.close()
    else:
        _, pred = model(graph.x, graph.edge_index).max(dim=1)
        correct = float (pred[test_mask].eq(graph.y.to(device)[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Test Accuracy: {:.4f}'.format(acc))
    tests_acc.append(acc)
    if not count_triangles:
        macro_score = f1_score(graph.y[test_mask], pred[test_mask].cpu(), average='macro')
        print('Macro score is: ' + str(macro_score))
        micro_score = f1_score(graph.y[test_mask], pred[test_mask].cpu(), average='micro')
        print('Micro score is: ' + str(micro_score))
        tests_f1_macro.append(macro_score)
        tests_f1_micro.append(micro_score)
if count_triangles:
    print('Average loss of ' + str(len(tests_acc)) + ' tests is: ' + str(sum(tests_acc) / len(tests_acc)))
    print('Average clustering coefficient of ' + str(len(tests_clus_avg)) + ' tests is: ' + str(sum(tests_clus_avg) / len(tests_clus_avg)))
else:
    print('Average accuracy of ' + str(len(tests_acc)) + ' tests is: ' + str(sum(tests_acc) / len(tests_acc)))
    print('Average F1 macro score of ' + str(len(tests_f1_macro)) + ' tests is: ' + str(sum(tests_f1_macro) / len(tests_f1_macro)))
    print('Average F1 micro score of ' + str(len(tests_f1_micro)) + ' tests is: ' + str(sum(tests_f1_micro) / len(tests_f1_micro)))

best_test_in = 0
best_test = tests_acc[0]
for i, acc in enumerate(tests_acc):
    if count_triangles:
        if acc < best_test:
            best_test = acc
            best_test_in = i
    else:
        if acc > best_test:
            best_test = acc
            best_test_in = i

wandb.save('model'+str(best_test_in+1)+'.p')
#torch.save(model.state_dict(), osp.join(wandb.run.dir, 'model'+str(best_test_in+1)+'.p'))

print('Model configuration:')
with open('./EGONETCONFIG.py', 'r') as f:
    print(f.read())

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, pool, SAGEConv
from ./../EGONETCONFIG import hidden_sizes, layer_design
 
class EgoGNN(torch.nn.Module):
    def __init__(self, egoNets, device, num_out):
        super(Net, self).__init__()

        self.egoNets = egoNets
        self.numNodes = len(self.egoNets)
        self.device = device

        if layer_design[0][0] == "Ego":
            self.conv1Inter = torch.nn.Linear(graph.x.shape[1], graph.x.shape[1])

        if layer_design[0][1] == "GIN":
            if hidden_sizes[1] == "out":
                self.conv1Intra = GINConv(torch.nn.Linear(graph.x.shape[1], num_out))
            else:
                self.conv1Intra = GINConv(torch.nn.Linear(graph.x.shape[1], hidden_sizes[1]))
        if layer_design[0][1] == "GCN":
            if hidden_sizes[1] == "out":
                self.conv1Intra = GCNConv(graph.x.shape[1], num_out)
            else:
                self.conv1Intra = GCNConv(graph.x.shape[1], hidden_sizes[1])

        if layer_design[1][0] == "Ego":
            if hidden_sizes[1] == "out":
                self.conv2Inter = torch.nn.Linear(num_out, num_out)
            else:
                self.conv2Inter = torch.nn.Linear(hidden_sizes[1], hidden_sizes[1])

        if layer_design[1][1] == "GIN":
            if hidden_sizes[2] == "out":
                self.conv1Intra = GINConv(torch.nn.Linear(hidden_sizes[1], num_out))
            else:
                self.conv1Intra = GINConv(torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]))
        if layer_design[1][1] == "GCN":
            if hidden_sizes[2] == "out":
                self.conv1Intra = GCNConv(hidden_sizes[1], num_out)
            else:
                self.conv1Intra = GCNConv(hidden_sizes[1], hidden_sizes[2])

    def do_conv(self, x):
        #orig = x.data.clone()
        #x = convInter[0](x, egoNets[0].edge_index.to(device).data)
        #print(torch.ones((egoNets[0].edge_index.shape[1])))
        output = torch_sparse.spmm(self.egoNets[0].edge_index.to(self.device), torch.ones((self.egoNets[0].edge_index.shape[1],)).to(device), self.numNodes, self.numNodes, x)
        #output = convInter(x, egoNets[0].edge_index.to(device))
        for i, ego in enumerate(self.egoNets):
            if i == 0:
                continue
            #cpu_x = x.to('cpu')
            #del x
            #torch.cuda.empty_cache()
            #cur_edge_index = ego.edge_index.to(device)
            #cur_conv = convInter[i](orig, cur_edge_index)
            #del cur_edge_index
            #torch.cuda.empty_cache()
            #x = cpu_x.to(device) + cur_conv
            #del cpu_x
            #torch.cuda.empty_cache()
            #output = output + convInter(x, ego.edge_index.to(device))
            output = output + torch_sparse.spmm(ego.edge_index.to(self.device), torch.ones((ego.edge_index.shape[1],)).to(device), self.numNodes, self.numNodes, x)
        #x = x * (1 / len(egoNets))
        output = output * (1 / self.numNodes)
        torch.cuda.empty_cache()
        return output
 
    def forward(self, x_in, edge_index_in):
        x = x_in.to(device)
        if layer_design[0][0] == "Ego":
            x = self.do_conv(x)
            x = self.conv1Inter(x)
            torch.cuda.empty_cache()
        #x = F.relu(x)
        if layer_design[0][1] != None:
            x = self.conv1Intra(x, edge_index_in.to(device))
            torch.cuda.empty_cache()
        x = F.relu(x)
        if layer_design[1][0] == "Ego":
            x = self.do_conv(x)
            x = self.conv2Inter(x)
            torch.cuda.empty_cache()
        #x = F.relu(x)
        if layer_design[01][1] != None:
            x = self.conv2Intra(x, edge_index_in.to(device))
            torch.cuda.empty_cache()
        #x = F.relu(x)
        return F.log_softmax(x, dim=1)

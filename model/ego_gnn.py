import sys
sys.path.insert(1, '..')
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, pool, SAGEConv
import torch_sparse
from EGONETCONFIG import hidden_sizes, layer_design
 
class EgoGNN(torch.nn.Module):
    def __init__(self, egoNets, device, num_out, num_feat):
        super(EgoGNN, self).__init__()

        self.egoNets = egoNets
        self.numNodes = len(self.egoNets)
        self.device = device
        modList = []
        for i, layer in enumerate(layer_design):
            input_num = hidden_sizes[i]
            if input_num == "in":
                input_num = num_feat
            if input_num == "out":
                input_num = num_out
            output_num = hidden_sizes[i+1]
            if output_num == "in":
                output_num = num_feat
            if output_num == "out":
                output_num = num_out
            if layer[0] == "Ego":
                modList.append(torch.nn.Linear(input_num, input_num))
            if layer[1] == "GIN":
                modList.append(GINConv(torch.nn.Linear(input_num, output_num)))
            if layer[1] == "GCN":
                modList.append(GCNConv(input_num, output_num))
        self.layers = torch.nn.ModuleList(modList)

    def do_conv(self, x):
        #orig = x.data.clone()
        #x = convInter[0](x, egoNets[0].edge_index.to(device).data)
        #print(torch.ones((egoNets[0].edge_index.shape[1])))
        output = torch_sparse.spmm(self.egoNets[0].edge_index.to(self.device), torch.ones((self.egoNets[0].edge_index.shape[1],)).to(self.device), self.numNodes, self.numNodes, x)
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
            output = output + torch_sparse.spmm(ego.edge_index.to(self.device), torch.ones((ego.edge_index.shape[1],)).to(self.device), self.numNodes, self.numNodes, x)
        #x = x * (1 / len(egoNets))
        output = output * (1 / self.numNodes)
        torch.cuda.empty_cache()
        return output
 
    def forward(self, x_in, edge_index_in):
        curMod = 0
        x = x_in.to(self.device)
        for i, layer in enumerate(layer_design):
            if layer[0] == "Ego":
                x = self.do_conv(x)
                x = self.layers[curMod](x)
                curMod = curMod + 1
            if layer[2]:
                x = F.relu(x)
            if layer[1] == "GIN":
                x = self.layers[curMod](x, edge_index_in.to(self.device))
                curMod = curMod + 1
            if layer[1] == "GCN":
                x = self.layers[curMod](x, edge_index_in.to(self.device))
                curMod = curMod + 1
            if layer[3]:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)

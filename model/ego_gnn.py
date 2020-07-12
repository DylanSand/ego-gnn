import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, pool, SAGEConv
 
class EgoGNN(torch.nn.Module):
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

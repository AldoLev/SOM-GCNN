import torch
import numpy as np
from util import args
from torch_geometric.nn import GCNConv, global_add_pool
from class_LinSum import LinSum
import torch.nn.functional as F
from torch_geometric.utils import scatter_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvN(torch.nn.Module):
    def __init__(self, num_features, num_classes, conv_dim, mlp_dim, load=False, train_batch=False ):
        super(ConvN, self).__init__()
        self.num_features = num_features
        self.hidden_conv = conv_dim
        self.hidden_mlp = mlp_dim
        self.hidden_mlp[0] = conv_dim*3
        self.hidden_mlp[-1] = num_classes
        #self.convolution = GConv(num_features, conv_dim )
        self.conv1 = GCNConv(num_features, conv_dim, improved=False)
        self.conv2 = GCNConv(conv_dim, conv_dim, improved=False)
        self.conv3 = GCNConv(conv_dim, conv_dim, improved=False)
        self.fc1 = torch.nn.Linear(3*conv_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128//2)
        self.fc3 = torch.nn.Linear(128//2, num_classes)
        #self.linear = LinSum(num_classes, conv_dim*3, mlp_dim)
        #self.linfc1 = torch.nn.Linear( 3*conv_dim, 128 )
        #self.linfc2 = torch.nn.Linear( 128, num_classes )
        #self.linfc3 = torch.nn.Linear( self.hidden_mlp[2], self.hidden_mlp[3] )
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if bool(load) == True:
            self.load_state_dict(torch.load('mod_save'+load+'.pt',map_location=device), strict=False)
            #self.linear.load_state_dict(torch.load('lin_save'+load+'.pt',map_location=device), strict=False)
            print('model '+load+' loaded')


    def forward(self, data, train_batch=False):
        if train_batch == True:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            x1 = F.leaky_relu(self.conv1(x, edge_index))
            x2 = F.leaky_relu(self.conv2(x1, edge_index))
            x3 = F.leaky_relu(self.conv3(x2, edge_index))
            x= torch.cat([x1,x2,x3], dim=1)
            x = global_add_pool(x, batch)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x=F.log_softmax(x, dim=-1)
        else:
            x = data.x
            edge_index = data.edge_index
            x1 = F.relu(self.conv1(x, edge_index))
            x2 = F.relu(self.conv2(x1, edge_index))
            x3 = F.relu(self.conv3(x2, edge_index))
            x= torch.cat([x1,x2,x3], dim=1)
            x = torch.sum(x,0)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            #x=F.log_softmax(x, dim=-1)

        return x

    def save_test(self, modelname1, modelname2):
        torch.save(self.state_dict(), modelname1)
        #torch.save(self.linear.state_dict(), modelname2)

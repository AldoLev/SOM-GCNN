# SOM-GCNN
import torch
import numpy as np
from util import args
#from class_GConv import GConv
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from minisom import MiniSom
from sklearn.model_selection import KFold


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvSOM_dense1(torch.nn.Module):
    def __init__(self, num_features, num_classes, conv_dim, lattice_dim, load=False, train_batch=False ):
        super(ConvSOM_dense1, self).__init__()
        self.load = load
        self.p = lattice_dim #lattice sides
        self.num_features = num_features #number of features
        self.hidden_conv = conv_dim # graph convolution dimension
        #self.convolution = GConv(num_features, conv_dim )
        self.conv1 = GCNConv(num_features, conv_dim, improved=False)
        self.conv2 = GCNConv(conv_dim, conv_dim, improved=False)
        self.conv3 = GCNConv(conv_dim, conv_dim, improved=False)
        #self.convolution_batch = GConv(num_features, conv_dim, train_batch=train_batch )
        #self.fc1 = torch.nn.Linear(self.p[0]*self.p[1], 164)
        #self.fc2= torch.nn.Linear(164, 96)
        #self.fc3= torch.nn.Linear(96, 32)
        #self.fc4= torch.nn.Linear(32, 2)
        self.lin1 = torch.nn.Linear( self.p[0]*self.p[1], 1)
        self.S_Tens = torch.zeros(self.p[0]*self.p[1],3*conv_dim, requires_grad=True)
        if bool(load) == True:
            self.load_state_dict(torch.load('./model_to_load/mod64_save'+load+'.pt',map_location=device), strict=False)
            print('model '+load+' loaded')


    def SOM_gradient( self, input, batch=[0]):
        n, q = input.size()
        np_input = input.cpu().detach().numpy()
        if np.array_equal(batch, [0]):
            G = torch.zeros((self.p[0],self.p[1]),dtype=torch.float32).to(device)
            for i in range(n):
                T_Tens = torch.from_numpy(self.som.neighborhood(self.som.winner(np_input[i,:] ), self.sigma)).type(torch.FloatTensor).to(device) # Tensorize the lattice mask
                #S_Tens_scalar_input = torch.matmul( self.S_Tens, input[i,:]) # lattice matrix of the scalar products
                #G = G + torch.mul( S_Tens_scalar_input, T_Tens ) # summing the contribution for each graph vertex
                w_ = self.S_Tens[self.som.winner(np_input[i,:] )]
                x_ = input[i,:]
                HS_norm = torch.exp( -torch.norm( w_- x_ ) ) # SOM with exp
                G = G + T_Tens*HS_norm # summing the contribution for each graph vertex
            return G
        else:
            G = torch.zeros((batch[-1]+1,self.p[0],self.p[1]),dtype=torch.float32).to(device)
            for i in range(len(batch)):
                T_Tens = torch.from_numpy(self.som.neighborhood(self.som.winner(np_input[i,:] ), self.sigma)).type(torch.FloatTensor).to(device) # Tensorize the lattice mask
                #S_Tens_scalar_input = torch.matmul( self.S_Tens, input[i,:]) # lattice matrix of the scalar products
                #G[batch[i]] = G[batch[i]] + torch.mul( S_Tens_scalar_input, T_Tens ) # summing the contribution for each graph vertex
                w_ = self.S_Tens[self.som.winner(np_input[i,:] )]
                x_ = input[i,:]
                HS_norm = torch.exp( -torch.norm( w_- x_ ) ) # SOM with exp
                G[batch[i]] = G[batch[i]] + T_Tens*HS_norm # summing the contribution for each graph vertex
            return G


    def MiniSom( self, sigma_learning, sigma_out, learning_rate, neighborhood_f):
        self.sigma = sigma_out
        # initialize the SOM
        self.som = MiniSom(self.p[0], self.p[1], self.hidden_conv*3, sigma=sigma_learning, learning_rate=learning_rate, neighborhood_function=neighborhood_f)

    def forward(self, data, train_batch=False):
        f = torch.nn.Sigmoid()
        if train_batch == True:
            #x, batch = self.convolution_batch(data)
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            x1 = F.leaky_relu(self.conv1(x, edge_index))
            x2 = F.leaky_relu(self.conv2(x1, edge_index))
            x3 = F.leaky_relu(self.conv3(x2, edge_index))
            # concatenazione
            x= torch.cat([x1,x2,x3], dim=1)
            # SOM con passaggio di gradiente
            x = self.SOM_gradient(x, batch=batch)
            x = x.view((batch[-1]+1,self.p[0]*self.p[1]))
            #x = F.relu(self.fc1(x))
            #x = F.relu(self.fc2(x))
            #x = F.relu(self.fc3(x))
            #x = self.fc4(x)
            #x=F.log_softmax(x, dim=-1)
            # ultimo layer denso
            x =f(self.lin1(x))
            return x
        else:
            #x = self.convolution(data)
            x = data.x
            edge_index = data.edge_index
            x1 = F.leaky_relu(self.conv1(x, edge_index))
            x2 = F.leaky_relu(self.conv2(x1, edge_index))
            x3 = F.leaky_relu(self.conv3(x2, edge_index))
            # concatenazione
            x= torch.cat([x1,x2,x3], dim=1)
            # SOM con passaggio di gradiente
            x = self.SOM_gradient(x)
            x = x.view(self.p[0]*self.p[1])
            # ultimo layer denso
            x =f(self.lin1(x))
            return x

    # Funzione per il pre-training con SVM
    def SVM_pretraining(self, dataset):
        X = []
        Y = []
        Coefficients = []
        Bias = []
        accuracy_tr = []
        accuracy_vl = []
        C = [ 1E3, 1E2, 1, 1E-1, 1E-2, 1E-3, 1E-4]
        for i in range(len(dataset)):
            X.append(np.ndarray.flatten(self.img_out(dataset[i].to(device))))
            Y.append(dataset[i].y.numpy().item()/2)

        X = np.array(X)
        Y = np.array(Y)

        kf = KFold(n_splits=8, shuffle = True)
        kf.split(X)
        for train_ind, val_ind in kf.split(X):
            #print(train_ind, val_ind)

            for c in C:
                modelSVM = LinearSVC(C=c, dual=False, max_iter=2e7 )
                modelSVM.fit(X[train_ind], Y[train_ind])
                accuracy_tr.append(modelSVM.score(X[train_ind],Y[train_ind]))
                accuracy_vl.append(modelSVM.score(X[val_ind],Y[val_ind]))
                Coefficients.append(torch.tensor(modelSVM.coef_, dtype=torch.float32).to(device))
                Bias.append(torch.tensor(modelSVM.intercept_, dtype=torch.float32).to(device))

            break

        print('SVM validation accuracy: ',accuracy_vl, np.argmax(accuracy_vl))
        print('SVM tr accuracy: ',accuracy_tr)
        return Coefficients[np.argmax(accuracy_vl)], Bias[np.argmax(accuracy_vl)]

    # salva il modello
    def save_test(self, modelname1, modelname2):
        torch.save(self.convolution.state_dict(), modelname1)
        torch.save(self.linear.state_dict(), modelname2)

    # Funzione per il training della SOM
    def train_SOM(self, data, num_iterations, first=True, verbose=False):
        #self.convolution.eval()
        #x = self.convolution(data)
        x = data.x
        edge_index = data.edge_index
        with torch.no_grad():
            x1 = F.leaky_relu(self.conv1(x, edge_index))
            x2 = F.leaky_relu(self.conv2(x1, edge_index))
            x3 = F.leaky_relu(self.conv3(x2, edge_index))
            x= torch.cat([x1,x2,x3], dim=1)
            X_in = x.cpu().detach().numpy()
        if first == True:
            #self.som.pca_weights_init(X_in)
            self.som.random_weights_init(X_in)
        self.som.train_batch( X_in, num_iterations, verbose=verbose)
        print()
        SOMweights = torch.from_numpy(self.som.get_weights())
        SOMweights = SOMweights.type(torch.FloatTensor).to(device)
        #self.S_Tens = SOMweights
        self.S_Tens = torch.nn.Parameter( SOMweights)

    # Funzione per ottenere l'errore di quantizzazione e il livello di attivazione dei neuroni della SOM
    def SOM_goodness(self, data, activation = False ):
        #self.convolution.eval()
        #x = self.convolution(data)
        x = data.x
        edge_index = data.edge_index
        with torch.no_grad():
            x1 = F.leaky_relu(self.conv1(x, edge_index))
            x2 = F.leaky_relu(self.conv2(x1, edge_index))
            x3 = F.leaky_relu(self.conv3(x2, edge_index))
            x= torch.cat([x1,x2,x3], dim=1)
            X_in = x.cpu().detach().numpy()
        Q_error = self.som.quantization_error(X_in)
        if activation == True:
            AR = self.som.activation_response(X_in)
            return Q_error, AR
        return Q_error

    # Funzione output della SOM senza passaggio di gradiente
    def img_out(self, graph):
        #self.convolution.eval()
        #x = self.convolution(graph)
        x = graph.x
        edge_index = graph.edge_index
        with torch.no_grad():
            x1 = F.leaky_relu(self.conv1(x, edge_index))
            x2 = F.leaky_relu(self.conv2(x1, edge_index))
            x3 = F.leaky_relu(self.conv3(x2, edge_index))
            x= torch.cat([x1,x2,x3], dim=1)
            X_in = x.cpu().detach().numpy()
        G = np.zeros((self.p[0],self.p[1]))
        S = self.som.get_weights()
        for n in X_in:
            S_scalar_n = np.matmul( S, n )
            #G = G + self.som.neighborhood(self.som.winner(n),self.sigma)*S_scalar_n # activation with the scalar product
            G = G + self.som.neighborhood(self.som.winner(n), self.sigma)*np.exp(-self.som.activate(n)) # activation with the distance
        return G

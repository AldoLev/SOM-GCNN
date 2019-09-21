#Main file per ottenere l'accuratezza
import torch
import pickle
import util
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from class_SOM_batch import ConvSOM_dense1
from sklearn.model_selection import KFold, train_test_split
import time

start = time.time()

dataset = TUDataset(root='./PTC_MR', name='PTC_MR')
print('Dataset information: ')
print('size ', len(dataset), ' graphs')
print( dataset.num_classes , ' classes' )
print( dataset.num_features , ' features' )

# conv_dim è il numero di canali della convoluzione,
# conv_dim deve essere lo stesso del modello che viene caricato
conv_dim = 64
lattice_dim = [util.args.p1, util.args.p2]
sigma_out =util.args.sigma_out
reg= util.args.regularization
wd = util.args.weight_decay
print('lattice: ', lattice_dim)
print('learning rate: ', util.args.learning_rate)
print('repetitions ',util.args.repetitions)
print('batch size: ', util.args.batch_size)
print('epochs: ', util.args.num_epochs)
print('sigma_out: ', sigma_out)
print('regularization: ', reg)
print('weight decay: ', wd )

# cpu o gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ',device)
#loading di tutto il dataset come un solo grafo per inizializzare la SOM
batch_train_SOM = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
all_data = []
for batch in batch_train_SOM:
    all_data.append(batch.to(device))
    break

# liste per salvare l'accuracy di test, validation e training 
accuracy_training = []
accuracy_test = []
accuracy_validation = []
for repetition in range(util.args.repetitions):
    print(time.time()-start)

    for k in range(util.args.fold):

        # indici per la cross validation
        train_indx = np.loadtxt('10fold_idx/train_idx-%d.txt' % (k+1) )
        test_indices = torch.from_numpy(np.loadtxt('10fold_idx/test_idx-%d.txt' % (k+1) )).type(torch.LongTensor)
        train_ind, val_ind, _, _ = train_test_split(train_indx, np.arange(len(train_indx)), test_size=0.1, random_state=42) 
        train_indices = torch.from_numpy(train_ind).type(torch.LongTensor)
        val_indices = torch.from_numpy(val_ind).type(torch.LongTensor)
        #print(train_indices)
        #print(test_indices)

        if util.args.is_test == 1:
            print('test')
            dataset_l = dataset[0:100]
        else:
            dataset_l = dataset[train_indices]
        dataset_t = dataset[test_indices]
        dataset_v = dataset[val_indices]
        # loading del dataset
        batch_learning = DataLoader(dataset_l, batch_size=util.args.batch_size, shuffle=True)
        # inizializza il modello
        model = ConvSOM_dense1(dataset.num_features, 2, conv_dim, lattice_dim, load=str(k), train_batch=t_b).to(device)
        # inizializza la SOM
        model.MiniSom( 2, sigma_out, 0.001, 'bubble')

        #q_error , AC = model.SOM_goodness(all_data[0], activation = True)
        #print('quantizayion error: ', q_error)
        #print('lattice activation: ')
        #print( AC.astype(int))

        # Training della SOM
        print('training the SOM ... ')
        model.train_SOM(all_data[0], 10000)

        # analisi errore di quantizzazione e livello di attivazione
        q_error , AC = model.SOM_goodness(all_data[0], activation = True)
        print('quantization error: ', q_error)
        print('lattice activation: ')
        print( AC)

        # SVM pre-training
        weights, bias = model.SVM_pretraining(dataset_l)
        model.lin1.weight = torch.nn.Parameter(weights)
        model.lin1.bias = torch.nn.Parameter(bias)

        acc_l = []
        acc_t = []
        acc_v = []
        
        model.eval()
        acc = util.accuracy_eval(dataset_v, model, device)
        acc_v.append(acc)
        acc_t.append( util.accuracy_eval(dataset_t, model, device))
        print('test0: ',acc_t[-1])
        acc_l.append( util.accuracy_eval(dataset_l, model, device))
        print('training0: ',acc_l[-1])

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=util.args.learning_rate, weight_decay=wd)

        # Loss Function
        criterion = torch.nn.BCELoss()
        #criterion = torch.nn.L1Loss()
        #criterion  = torch.nn.NLLLoss()
        #print('test ','training ' )
        for epoch in range(util.args.num_epochs):
            
            # TRAINING
            model.train()
            for batch in batch_learning:
                optimizer.zero_grad()
                target = batch.y.to(device).to(torch.float32)
                target=target.view((target.size()[0],1))
                input = batch.to(device)
                out = model(input,train_batch=t_b)
                loss = criterion(out, target)
                # regolarizzazione ultimo layer
                l2_crit = torch.nn.MSELoss()
                reg_loss = 0
                PARAM = list(model.parameters())
                reg_loss += l2_crit(PARAM[-2], torch.from_numpy(np.zeros(PARAM[-2].size())).to(torch.float32).to(device))
                reg_loss += l2_crit(PARAM[-1], torch.from_numpy(np.zeros(PARAM[-1].size())).to(torch.float32).to(device))
                factor = reg
                loss += factor * reg_loss
                # Backpropagation
                loss.backward()
                optimizer.step()
                model.som._weights = model.S_Tens.cpu().detach().numpy()

            model.eval()
            acc = util.accuracy_eval(dataset_v, model, device)
            acc_v.append(acc)
            acc_t.append( util.accuracy_eval(dataset_t, model, device))
            acc_l.append( util.accuracy_eval(dataset_l, model, device))
            #print(acc_t[-1],acc_l[-1])

        accuracy_validation.append(np.amax(acc_v))
        accuracy_training.append(acc_l[np.argmax(acc_v)])
        accuracy_test.append(acc_t[np.argmax(acc_v)])
        print(np.argmax(acc_v))
        if util.args.is_test == 1:
            print('test')
            break

        #print( util.accuracy_eval(dataset_t, model, device))
        #print('test_p: ',acc_t[-1])
        #print( util.accuracy_eval(dataset_l, model, device))
        #print('training_p: ',acc_l[-1])

    if util.args.is_test == 1:
        print('test')
        break

error = np.std(accuracy_test)/2
accuracy_training=np.mean(accuracy_training)
accuracy_validation=np.mean(accuracy_validation)
accuracy_test=np.mean(accuracy_test)
print('test: ', accuracy_test)
print('training: ', accuracy_training)
print('validation: ', accuracy_validation)
print('error: ', error, 'trials: ' , util.args.repetitions*util.args.fold)

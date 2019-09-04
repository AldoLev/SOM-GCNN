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
import time

start = time.time()

dataset = TUDataset(root='./PTC_MR', name='PTC_MR')
print('Dataset information: ')
print('size ', len(dataset), ' graphs')
print( dataset.num_classes , ' classes' )
print( dataset.num_features , ' features' )

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ',device)
batch_train_SOM = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
all_data = []
for batch in batch_train_SOM:
    all_data.append(batch.to(device))
    break

if util.args.batch_size == 1:
    print('online learning')
    t_b = False
else:
    t_b = True
    print('batch learning')

accuracy_training = []
accuracy_test = []
#history_loss = []
for repetition in range(util.args.repetitions):
    print(time.time()-start)

    for k in range(util.args.fold):

        '''
        #   TEST SET: cross_val[k]
        #   VALIDATION SET: cross_val[k+1]
        #   TRAINING SET: the others
        test_T = torch.from_numpy(cross_val[k])
        validation_T = torch.from_numpy(cross_val[k-1])
        train_T = torch.from_numpy(util.make_trainset(cross_val, k))
        test = cross_val[k]
        validation = cross_val[k-1]
        train = util.make_trainset(cross_val, k)

        dataset_l = dataset[train_T]
        dataset_v = dataset[validation_T]
        dataset_t = dataset[test_T]
        '''

        train_indices = torch.from_numpy(np.loadtxt('10fold_idx/train_idx-%d.txt' % (k+1) )).type(torch.LongTensor)
        test_indices = torch.from_numpy(np.loadtxt('10fold_idx/test_idx-%d.txt' % (k+1) )).type(torch.LongTensor)
        #print(train_indices)
        #print(test_indices)

        if util.args.is_test == 1:
            print('test')
            dataset_l = dataset[0:40]
        else:
            dataset_l = dataset[train_indices]
        dataset_t = dataset[test_indices]
        #dataset_v = dataset[train_indices[:val_split]]


        batch_learning = DataLoader(dataset_l, batch_size=util.args.batch_size, shuffle=True)

        model = ConvSOM_dense1(dataset.num_features, 2, conv_dim, lattice_dim, load=str(k), train_batch=t_b).to(device)
        model.MiniSom( 2, sigma_out, 0.3, 'bubble')

        q_error , AC = model.SOM_goodness(all_data[0], activation = True)
        print('quantizayion error: ', q_error)
        print('lattice activation: ')
        print( AC.astype(int))

        print('training the SOM ... ')
        model.train_SOM(all_data[0], 10000)
        #with open('som0.p', 'wb') as outfile:
            #pickle.dump(model.som, outfile)

        q_error , AC = model.SOM_goodness(all_data[0], activation = True)
        print('quantization error: ', q_error)
        print('lattice activation: ')
        print( AC)

        weights, bias = model.SVM_pretraining(dataset_l)
        model.lin1.weight = torch.nn.Parameter(weights)
        model.lin1.bias = torch.nn.Parameter(bias)

        acc_l = []
        acc_t = []
        #acc_v = []
        #Loss_f = []

        model.eval()
        #acc = util.accuracy_eval(dataset_v, model, device)
        #acc_v.append(acc)
        acc_t.append( util.accuracy_eval(dataset_t, model, device))
        print('val0: ',acc_t[-1])
        acc_l.append( util.accuracy_eval(dataset_l, model, device))
        print('training0: ',acc_l[-1])

        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print( name, param.data)

        optimizer = torch.optim.Adam(model.parameters(), lr=util.args.learning_rate, weight_decay=wd)

        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.BCELoss()
        #criterion = torch.nn.L1Loss()
        #criterion  = torch.nn.NLLLoss()
        #print('test ','training ', 'loss ' )
        for epoch in range(util.args.num_epochs):
            #loss_train =0
            #ln = 0
            model.train()
            if util.args.batch_size == 1:
                for i in range(len(dataset_l)):
                    optimizer.zero_grad()
                    target = dataset_l[i].y.to(device).to(torch.float32)
                    input = dataset_l[i].to(device)
                    out = model(input)
                    out = out[None,:]
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    #model.som._weights = model.S_Tens
            else:
                for batch in batch_learning:
                    optimizer.zero_grad()
                    target = batch.y.to(device).to(torch.float32)/2
                    target=target.view((target.size()[0],1))
                    input = batch.to(device)
                    out = model(input,train_batch=t_b)
                    loss = criterion(out, target)
                    l2_crit = torch.nn.MSELoss()
                    reg_loss = 0
                    PARAM = list(model.parameters())
                    reg_loss += l2_crit(PARAM[-2], torch.from_numpy(np.zeros(PARAM[-2].size())).to(torch.float32).to(device))
                    reg_loss += l2_crit(PARAM[-1], torch.from_numpy(np.zeros(PARAM[-1].size())).to(torch.float32).to(device))
                    factor = reg
                    loss += factor * reg_loss
                    #loss_train = loss_train + loss.detach().numpy()
                    #ln = ln +1
                    loss.backward()
                    optimizer.step()
                    model.som._weights = model.S_Tens.cpu().detach().numpy()

            #Loss_f.append( loss_train/ln)
            #print('loss: ',Loss_f[-1])

            #print(model.fc1.weight.grad)
            #print(model.fc1.bias.grad)

            model.eval()
            #acc = util.accuracy_eval(dataset_v, model, device)
            #acc_v.append(acc)
            acc_t.append( util.accuracy_eval(dataset_t, model, device))
            acc_l.append( util.accuracy_eval(dataset_l, model, device))
            #print(acc_v[-1],acc_l[-1])


        #accuracy_validation.append(acc_v)
        accuracy_training.append(acc_l)
        accuracy_test.append(acc_t)
        #history_loss.append(Loss_f)
        if util.args.is_test == 1:
            print('test')
            break

        #weights2, bias2 = model.SVM_pretraining(dataset_l)
        #model.fc1.weight = torch.nn.Parameter(weights2)
        #model.fc1.bias = torch.nn.Parameter(bias2)
        #print( util.accuracy_eval(dataset_t, model, device))
        #print('test_p: ',acc_t[-1])
        #print( util.accuracy_eval(dataset_l, model, device))
        #print('training_p: ',acc_l[-1])

    if util.args.is_test == 1:
        print('test')
        break

error_tr = np.std(accuracy_training,axis=0)
error_ts = np.std(accuracy_test,axis=0)
accuracy_training=np.mean(accuracy_training,axis=0)
accuracy_test=np.mean(accuracy_test,axis=0)
#history_loss = np.mean(history_loss, axis=0)
#print(history_loss)
#np.save('loss.npy', history_loss)
print('test: ', accuracy_test)
np.save('./exp_results/acc_test_100l2.npy', accuracy_test)
print('training: ', accuracy_training)
np.save('./exp_results/acc_training_100l2.npy', accuracy_training)
print('error_tr: ', error_tr)
np.save('./exp_results/error_100tr.npy', error_tr)
print('error_ts: ', error_ts)
np.save('./exp_results/error_100ts.npy', error_ts)

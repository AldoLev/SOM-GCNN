# Main file per il training del modello GCNNsum
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
#from class_sum import ConvN
from class_GCNsum import ConvN
from util_sum import args, make_trainset
from sklearn.model_selection import KFold
import ssl
ssl.match_hostname = lambda cert, hostname: True



dataset = TUDataset(root='./PTC_MR', name='PTC_MR')
print('Dataset information: ')
print('size ', len(dataset), ' graphs')
print( dataset.num_classes , ' classes' )
print( dataset.num_features , ' features' )


conv_dim = 64
mlp_dim = [0,128,2]
print('conv_dim: ', conv_dim)
print('mlp_dim: ', mlp_dim)
#cross_val = np.load('cross_validation.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

if args.batch_size == 1:
    print('online learning')
    t_b = False
else:
    t_b = True
    print('batch learning')

accuracy_training = []
accuracy_test = []
for k in range(args.fold):
    print(k)
    '''
    test_T = torch.from_numpy(cross_val[k])
    train_T = torch.from_numpy(make_trainset(cross_val, k))
    test = cross_val[k]
    train = make_trainset(cross_val, k)

    '''
    train_indices = torch.from_numpy(np.loadtxt('10fold_idx/train_idx-%d.txt' % (k+1) )).type(torch.LongTensor)
    test_indices = torch.from_numpy(np.loadtxt('10fold_idx/test_idx-%d.txt' % (k+1) )).type(torch.LongTensor)
    #print(train_indices)
    #print(test_indices)

    if args.is_test == 1:
        print('test')
        dataset_l = dataset[20:30]
    else:
        dataset_l = dataset[train_indices]
    dataset_t = dataset[test_indices]

    batch_learning = DataLoader(dataset_l, batch_size=args.batch_size, shuffle=True)

    model = ConvN(dataset.num_features, 2, conv_dim, mlp_dim, train_batch=t_b).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion_online = torch.nn.CrossEntropyLoss()
    criterion_batch = torch.nn.NLLLoss()
    model.train()
    for epoch in range (args.num_epochs):
        if args.batch_size == 1:
            for i in range(len(dataset_l)):
                optimizer.zero_grad()
                target = (dataset_l[i].y.to(device))
                input = dataset_l[i].to(device)
                out = model(input)
                out = out[None,:]
                loss = criterion_online(out, target)
                loss.backward()
                optimizer.step()
        else:
            for batch in batch_learning:
                optimizer.zero_grad()
                target = batch.y.to(device).to(torch.int64)
                input = batch.to(device)
                out = model(input,train_batch=t_b)
                loss = criterion_batch(out, torch.div(target,2))
                loss.backward()
                optimizer.step()

    model.eval()
    correct_l = 0
    for i in range(len(dataset_l)):
        pred = torch.argmax(model(dataset_l[i].to(device)))
        if (torch.div(dataset_l[i].y.to(device),2) == pred ):

                correct_l = correct_l + 1

    accuracy_training.append( correct_l / len(dataset_l))

    correct_t = 0
    for i in range(len(dataset_t)):
        pred = torch.argmax(model(dataset_t[i].to(device)))
        if (torch.div(dataset_t[i].y.to(device),2) == pred ):
            correct_t = correct_t + 1

    accuracy_test.append( correct_t / len(dataset_t))
    print(accuracy_test[-1])

    model.save_test('mod64_save'+str(k)+'.pt', 'l64_save'+str(k)+'.pt')

    if args.is_test == 1:
        print('test')
        break

error_test=np.std(accuracy_test)
accuracy_training=np.mean(accuracy_training)
accuracy_test=np.mean(accuracy_test)
print('accuracy test: ',accuracy_test)
print('accuracy training: ',accuracy_training)
print('error_test: ', error_test)

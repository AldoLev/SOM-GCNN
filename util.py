import torch
import numpy as np
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian(x,sigma=1):

    return np.exp(np.linalg.norm(x, axis=0)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def out_saturation(x):
    if x.item() < 0.5:
        return torch.tensor([0]).to(device)
    if x.item() >= 0.5:
        return torch.tensor([2]).to(device)
    else:
        return nan


opt = argparse.ArgumentParser(description='Argparser for graph_classification')
opt.add_argument('-fold', type=int, help='number of fold cross validation')
opt.add_argument('-p1', type=int, help='vertical lattice size')
opt.add_argument('-p2', type=int, help='orizontal lattice size')
opt.add_argument('-is_test', type=int, help='true if it is a test')
opt.add_argument('-repetitions', type=int, help='number repeted runs')
opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
opt.add_argument('-weight_decay', type=float, default=0.000001, help='optimizer weiht dacay')
opt.add_argument('-regularization', type=float, default=0.000005, help='last layer regularization')
opt.add_argument('-sigma_out', type=float, default=1, help='width of the SOM vertex representation')
args, _ = opt.parse_known_args()



def make_trainset(cross_val, num):
    train_1 = np.zeros(1,dtype = int)
    for j in range(args.fold):
        if j == num or j== (num-1)%args.fold:
            continue
        else:
            train_1 = np.append(cross_val[j], train_1)
    train = np.delete(train_1,-1)
    return train


def accuracy_eval(dataset, model, device):
    correct=0
    with torch.no_grad():
        for i in range(len(dataset)):
            pred = out_saturation(model(dataset[i].to(device)))
            if (dataset[i].y.to(device) == pred ):
                correct = correct + 1
    return (correct / len(dataset) )

def accuracy_eval_class(dataset, model, device):
    correct=0
    with torch.no_grad():
        for i in range(len(dataset)):
            pred = torch.argmax(model(dataset[i].to(device)))
            if (dataset[i].y.to(device)/2 == pred ):
                correct = correct + 1

    return (correct / len(dataset))

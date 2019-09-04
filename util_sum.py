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
        return torch.tensor([1]).to(device)
    else:
        return nan


opt = argparse.ArgumentParser(description='Argparser for graph_classification')
opt.add_argument('-fold', type=int, help='number of fold cross validation')
opt.add_argument('-is_test', type=int, help='true if it is a test')
opt.add_argument('-repetitions', type=int, help='number repeted runs')
opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
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
            if ( torch.div(dataset[i].y.to(device),2) == pred ):
                correct = correct + 1
    return (correct / len(dataset) )

def accuracy_eval_class(dataset, model, device):
    correct=0
    with torch.no_grad():
        for i in range(len(dataset)):
            pred = torch.argmax(model(dataset[i].to(device)))
            if ( torch.div(dataset[i].y.to(device),2) == pred ):
                correct = correct + 1

    return (correct / len(dataset))

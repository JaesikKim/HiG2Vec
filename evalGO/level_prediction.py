import torch as th
import torch.nn as nn
import argparse

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import copy

class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_in, int(dim_in/2)),
            nn.BatchNorm1d(int(dim_in/2)),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(int(dim_in/2), dim_out),
        )
    def forward(self, x):
        out = self.main(x).view(-1)
        return out

class Scheduler():
    def __init__(self, optimizer, init_lr, n_warmup, epochs):
        self.optim = optimizer
        self.n_warmup = n_warmup
        self.lr_multiplier = 0.1
        self.init_lr = init_lr
        self.total_epochs = epochs

    def zero_grad(self):
        self.optim.zero_grad()

    def step_and_update_lr(self, curr_eph):
        self.update_lr(curr_eph)
        self.optim.step()
        
    def update_lr(self, curr_eph):
        if curr_eph < self.n_warmup:
            lr = self.init_lr * self.lr_multiplier 
        else:
            lr = self.init_lr * max(0.0,float(self.total_epochs-curr_eph))/float(max(1.0,self.total_epochs-self.n_warmup))
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr


def load_data(samples, objects):
    x_ls = []
    y_ls = []
    for i in range(len(samples)):
        GO = samples.iloc[i,0]
        level = samples.iloc[i,1]
        if GO in objects:
            GOi = objects.index(GO)
            x_ls.append([GOi])
            y_ls.append(level-1) # 0-base
    return np.array(x_ls), np.array(y_ls)

def map_to_vec(samples, embeddings):
    x_ls = []
    for i in range(len(samples)):
        x_ls.append(embeddings[int(samples[i,0].item())])
    return th.FloatTensor(x_ls)

def main():
    parser = argparse.ArgumentParser(description='Predict protein interaction')
    parser.add_argument('-model', help='Embedding model', type=str)
    parser.add_argument('-dset', help='GO level', type=str)
    parser.add_argument('-fout', help='Prediction output', type=str)
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-gpu', help='GPU id', default=0, type=int)
    parser.add_argument('-burnin', help='Epochs of burn in', type=int, default=20)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('-print_each', help='Print loss each n-th epoch', type=int, default=10)
    opt = parser.parse_args()

    # load embeddings
    if opt.model[-3:] == "pth":
        model = th.load(opt.model, map_location="cpu")
        objects, embeddings = model['objects'], model['embeddings'].cpu().numpy()

    else:
        model = np.load(opt.model, allow_pickle=True).item()
        objects, embeddings = model['objects'], model['embeddings']

    # dataset processing
    print("... load data ...")
    if opt.dset[-3:] == "tsv":
        data = pd.read_csv(opt.dset, sep="\t")
    else:
        data = pd.read_csv(opt.dset)
    X, y = load_data(data, objects)
    dim_out = 1
    device = th.device('cuda:'+str(opt.gpu) if th.cuda.is_available() else 'cpu')
    
    ytrue_all = [] 
    yhat_all = []
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        net = Net(opt.dim, dim_out).to(device)
        criterion = nn.MSELoss()
        optimizer = th.optim.Adam(net.parameters(), lr=opt.lr)
        scheduler = Scheduler(optimizer, opt.lr, opt.burnin, opt.epochs)

        # Dataloader
        train_dataset = TensorDataset(th.FloatTensor(X_train), th.FloatTensor(y_train.astype(int)))
        val_dataset = TensorDataset(th.FloatTensor(X_val), th.FloatTensor(y_val.astype(int)))
        test_dataset = TensorDataset(th.FloatTensor(X_test), th.FloatTensor(y_test.astype(int)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batchsize,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
        )

        # Train the model
        print("... Train Network ...")
        opt_eph = 0
        opt_loss = np.inf
        opt_model_state_dict = None
        for epoch in range(opt.epochs):
            epoch_loss = []
            net.train()
            for samples, labels in train_loader:
                samples = map_to_vec(samples, embeddings)
                samples = samples.to(device)
                labels = labels.to(device)
                preds = net(samples)
                loss = criterion(preds, labels)
                scheduler.zero_grad()
                loss.backward()
                scheduler.step_and_update_lr(epoch)
                epoch_loss.append(loss.item())

            with th.no_grad():
                net.eval()
                val_loss = []
                for samples, labels in val_loader:
                    samples = map_to_vec(samples, embeddings)
                    samples = samples.to(device)
                    labels = labels.to(device)
                    preds = net(samples)
                    loss = criterion(preds, labels)
                    val_loss.append(loss.item())
                if np.mean(val_loss) < opt_loss:
                    opt_loss = np.mean(val_loss)
                    opt_eph = epoch
                    opt_model_state_dict = copy.deepcopy(net.state_dict())

            if (epoch+1) % opt.print_each == 0:
                print("Epoch [{}/{}] Train Loss: {:.3f}  Val Loss: {:.3f}".format(epoch+1, opt.epochs, np.mean(epoch_loss), np.mean(val_loss)))

        # Calculate the test result
        net.load_state_dict(opt_model_state_dict)
        print("Optimal tuning: Epoch {}, Val Loss: {:.3f}".format(opt_eph+1, opt_loss))
        ytrue = []
        yhat = []
        with th.no_grad():
            net.eval()
            for samples, labels in test_loader:
                samples = map_to_vec(samples, embeddings)
                samples = samples.to(device)
                preds = net(samples)
                preds = [float(l)+1 for l in preds.cpu().tolist()]
                labels = [float(l)+1 for l in labels.tolist()]
                yhat += preds
                ytrue += labels
        ytrue_all += ytrue 
        yhat_all += yhat 
#     print("Test MSE: {:.3f}".format(mean_squared_error(y, yhat)))
#     print("Test Corr: {:.3f}".format(np.corrcoef(y, yhat)[0,1]))
    pd.DataFrame({'y' : ytrue_all, 'yhat' : yhat_all}).to_csv(opt.fout, index=False)


           
if __name__ == '__main__':
    main()
    


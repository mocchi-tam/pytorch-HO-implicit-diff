import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# classification model for iris
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.fc3 = nn.Linear(1024, 3, bias=False)
        self.dropout1 = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class MyLinear(nn.Linear):
    def forward(self, input):
        weight = self.weight*self.weight
        return F.linear(input, weight, self.bias)
    
# hyperparameters : pseudo-adaptive weight decay
class WDNet(nn.Module):
    def __init__(self):
        super(WDNet, self).__init__()
        self.fc1 = MyLinear(17280, 1, bias=False)
        nn.init.uniform_(self.fc1.weight, 0.0, 1e-2)
        
    def forward(self, w):
        w = w*w
        output = self.fc1(w)
        return output
    
# user-defined dataset for iris
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.label = target
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# main class
class MainClass():
    def __init__(self):
        # settings
        parser = argparse.ArgumentParser(description='PyTorch HO Example')
        parser.add_argument('--gpu', type=int, choices=[-1,0,1,2,3], default=0)
        parser.add_argument('--lr-hyper', type=float, default=0.1)
        parser.add_argument('--lr-neumann', type=float, default=0.1)
        parser.add_argument('--i-neumann', type=int, default=5)
        parser.add_argument('--ho', type=int, choices=[0,1], default=1)
        parser.add_argument('--sgd-wd', type=int, choices=[0,1], default=0)
        
        torch.manual_seed(0)
        
        args = parser.parse_args()
        gpu = args.gpu
        self.device = torch.device('cpu' if gpu < 0 else 'cuda:{}'.format(gpu))
        self.kwargs = {} if gpu < 0 else {'num_workers': 1, 'pin_memory': True}
        
        # experiment setups
        self.lr_hyper = args.lr_hyper
        self.lr_Neumann = args.lr_neumann
        self.i_Neumann = args.i_neumann
        # optimize hyperparameters or not
        self.flag_ho = True if args.ho == 1 else False
        self.flag_sgd_wd = True if args.sgd_wd == 1 else False
        
    def run(self):
        # init state
        num_epochs = 100
        num_epochs_hyper = 100
        sgd_wd = 1e-5
        if self.flag_ho == True or self.flag_sgd_wd == False:            
            sgd_wd = 0
            
        # learning rate
        lr = 0.01
        
        # classification model
        self.model = Net().to(self.device)
        # optimizer for classification model
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=lr, momentum=0.9, nesterov=True, weight_decay=sgd_wd)
        if self.flag_ho:
            # hyperparamter model
            self.model_ho = WDNet().to(self.device)
            # optimizer for hyperparameter model
            self.optimizer_ho = optim.SGD(self.model_ho.parameters(), lr=self.lr_hyper)
        
        # prepare iris
        self.prepare_dataset_iris()
        # hyperparameter tuning
        for epoch_ho in range(1, num_epochs_hyper+1):
            self.model.train()
            # optimize classification model with current hyperparameters
            for epoch in range(1, num_epochs+1):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                self.optimizer.zero_grad()
                loss, _ = self.calc_loss(data, target, flag_wd=self.flag_ho)
                loss.backward()
                self.optimizer.step()
            
            if self.flag_ho:
                self.model.eval()
                # calculate hypergradient
                # for validation
                self.model.zero_grad()
                self.model_ho.zero_grad()
                for batch_idx, (data, target) in enumerate(self.val_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                loss_V, _ = self.calc_loss(data, target, flag_wd=True)
                loss_V.backward()
                v1 = None # dLv/dw
                for params in self.model.parameters():
                    tmp = torch.flatten(params.grad)
                    v1 = tmp if v1 is None else torch.cat((v1, tmp), 0)
                    
                u1 = self.model_ho.fc1.weight.grad.clone() # dLv/dl
                
                # for train
                self.model.zero_grad()
                self.model_ho.zero_grad()
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                loss_T, w_T = self.calc_loss(data, target)
                f = torch.autograd.grad(loss_T, w_T, create_graph=True) # dLt/dw
                lmd = self.model_ho.fc1.weight 
                g = torch.autograd.grad(loss_T, lmd, create_graph=True) # dLt/dl
                
                p = v1.clone()
                for index_j in range(self.i_Neumann):
                    neumann_grad = torch.autograd.grad(
                            f, w_T, retain_graph=True, grad_outputs=v1)[0] # d2Lt/dwdw
                    v1 -= self.lr_Neumann * neumann_grad
                    p -= v1
                    
                v2 = torch.reshape(p, (1, -1))
                v3 = torch.autograd.grad(g, w_T, grad_outputs=v2)[0] # d2Lt/dwdl
                
                # update hyperparameters
                hyper_grad = u1 - v3
                self.optimizer_ho.zero_grad()
                self.model_ho.fc1.weight.grad.data = hyper_grad
                self.optimizer_ho.step()
            else:
                pass
            
            # test
            self.model_test(epoch_ho)
    
    # prepare dataset for iris
    def prepare_dataset_iris(self):
        # train 60, val 60, test 30
        iris_dataset = datasets.load_iris()
        data = iris_dataset.data.astype(np.float32)
        target = iris_dataset.target.astype(np.int64)
        trval_X, test_X, trval_Y, test_Y = train_test_split(data, target, test_size=0.2)
        train_X, val_X, train_Y, val_Y = train_test_split(trval_X, trval_Y, test_size=0.5)
        
        self.train_loader = torch.utils.data.DataLoader(MyDataset(train_X, train_Y),
                                                        batch_size=60, shuffle=True, **self.kwargs)
        self.val_loader = torch.utils.data.DataLoader(MyDataset(val_X, val_Y),
                                                      batch_size=60, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(MyDataset(test_X, test_Y),
                                                       batch_size=30, shuffle=False, **self.kwargs)
    
    # calculate loss
    def calc_loss(self, data, target, flag_wd=True):
        # Xentropy loss
        output = self.model(data)
        loss = F.nll_loss(output, target)
        
        # L2 loss
        w = None
        if flag_wd == True:
            for params in self.model.parameters():
                tmp = torch.flatten(params)
                w = tmp if w is None else torch.cat((w, tmp), 0)
            w_loss = self.model_ho(w)
            loss += F.l1_loss(w_loss, target=torch.zeros_like(w_loss))
            
        return loss, w
    
    # evaludate loss/acc
    def model_test(self, epoch_ho):
        self.model.eval()
        
        train_loss, train_corr = self.model_eval(self.train_loader)
        val_loss, val_corr = self.model_eval(self.val_loader)
        test_loss, test_corr = self.model_eval(self.test_loader)
        
        print('Epoch {} Loss: {:.4f}, {:.4f}, {:.4f}, Accuracy: {:.0f}% {:.0f}% {:.0f}%'.format(
                epoch_ho, train_loss, val_loss, test_loss, train_corr, val_corr, test_corr))
    
    def model_eval(self, loader):
        N_data = len(loader.dataset)
        loss, corr = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True) 
                corr += pred.eq(target.view_as(pred)).sum().item()
                
        loss /= N_data
        corr *= (100.0/N_data)
        return loss, corr
    
if __name__ == '__main__':
    mc = MainClass()
    mc.run()

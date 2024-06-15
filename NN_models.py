# -*- coding: utf-8 -*-
# creat_time: 2021/11/13 18:28


from torch import nn
import torch.nn.functional as F


## NN(1)

class Net1(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_output, dropout):
        super(Net1, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.output = nn.Linear(n_hidden1, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.output(x)
        return x





## NN(2)
class Net2(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, dropout):
        super(Net2, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        #
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        #
        x = self.output(x)
        return x




class Net3(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output, dropout):
        super(Net3, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(n_hidden3, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        #
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        #
        x = self.hidden3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        #
        x = self.output(x)
        return x



## NN(4)

class Net4(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output, dropout):
        super(Net4, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.dropout3 = nn.Dropout(dropout)
        self.hidden4 = nn.Linear(n_hidden3, n_hidden4)
        self.dropout4 = nn.Dropout(dropout)
        #
        self.output = nn.Linear(n_hidden4, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        #
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        #
        x = self.hidden3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        #
        x = self.hidden4(x)
        x = self.dropout4(x)
        x = F.relu(x)
        #
        x = self.output(x)
        return x





## NN(5)

class Net5(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output, dropout):
        super(Net5, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.dropout3 = nn.Dropout(dropout)
        self.hidden4 = nn.Linear(n_hidden3, n_hidden4)
        self.dropout4 = nn.Dropout(dropout)
        self.hidden5 = nn.Linear(n_hidden4, n_hidden5)
        self.dropout5 = nn.Dropout(dropout)
        self.output = nn.Linear(n_hidden5, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        #
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        #
        x = self.hidden3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        #
        x = self.hidden4(x)
        x = self.dropout4(x)
        x = F.relu(x)
        #
        x = self.hidden5(x)
        x = self.dropout5(x)
        x = F.relu(x)
        #
        x = self.output(x)
        return x

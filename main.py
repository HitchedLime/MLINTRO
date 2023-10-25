import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import neural_net
from test_nn import test_nn
from train_nn import train_nn

dataset = pd.read_csv("HeartDisease.csv")

X = dataset.drop(['row.names', 'chd'], axis=1)
y = dataset['chd']
scaler = preprocessing.StandardScaler().fit(X.values)
X = scaler.transform(X.values)
X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.20, random_state=42)

net = neural_net.NeuralNetwork()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

num_epochs = 10

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train).unsqueeze(1).float()
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test).unsqueeze(1).float()

#initializes randomly weights in network to help converge
torch.nn.init.xavier_uniform_(net.linear_relu_stack[0].weight)
torch.nn.init.xavier_uniform_(net.linear_relu_stack[2].weight)


for i in range(num_epochs):
    final_loss, learning_hist, net = train_nn(optimizer, criterion, net, X_train, y_train)
    print(f"This is final  training loss: {final_loss} in epoch: {i} ")
    test_loss = test_nn(net, X_test, y_test, criterion)
    print(f"test loss {test_loss}")

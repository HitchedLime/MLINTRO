import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import neural_net
from sklearn.utils import shuffle
from torch.autograd import Variable
import numpy as np

dataset = pd.read_csv("HeartDisease.csv")

X = dataset.drop(['row.names', 'chd'], axis=1)
y = dataset['chd']
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)


net = neural_net.NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

batch_size = 100
num_epochs = 5
learning_rate = 0.001
batch_no = len(X_train) // batch_size

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch + 1))
    x_train, y_train = shuffle(X_train, y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(X_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end]))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        ypred_var = net(x_var)
        loss = criterion(ypred_var, y_var)
        print(f"Training Loss : {loss}")
        loss.backward()
        optimizer.step()

test_var = Variable(torch.FloatTensor(X_test), requires_grad=True)

with torch.no_grad():
        result = net(test_var)
        values, labels = torch.max(result, 1)
        num_right = np.sum(labels.data.numpy() == y_test)
        print('Accuracy {:.2f}'.format(num_right / len(y_test)))

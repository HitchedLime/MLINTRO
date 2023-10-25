import numpy as np
import torch


def test_nn(net,X_test,y_test,loss_fn):
    with torch.no_grad():
        pred = net(X_test)
        loss = loss_fn(pred,y_test)
        values, labels = torch.max(pred, 1)
        num_right = torch.sum(labels == y_test)
      #  print('Accuracy {:.2f}'.format(num_right / len(y_test)))
        return num_right / len(y_test),loss.data.numpy()
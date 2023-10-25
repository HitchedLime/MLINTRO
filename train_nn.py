import numpy as np


def train_nn(optimizer, loss_fn, net, X, y, threshold=1e-6, minibatches=5):
    learning_hist = []
    old_loss = -np.inf

    # minibatches
    for i in range(minibatches):
        y_pred = net(X)
        loss = loss_fn(y_pred, y)
        loss_val = loss.data.numpy()

        loss_over_time = abs(loss_val - old_loss) / old_loss
        if (loss_over_time < threshold):
            break
        old_loss = loss_val

        learning_hist.append(loss_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_val, learning_hist, net

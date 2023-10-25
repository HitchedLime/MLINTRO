import torch


def test_nn(net, x_test, y_test, loss_fn):
    with torch.no_grad():
        pred = net(x_test)
        loss = loss_fn(pred, y_test)
        return loss.data.numpy()

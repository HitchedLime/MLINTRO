

from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 200),
            nn.Tanh(),

            nn.Linear(200, 1),

            nn.Sigmoid(),

        )



    def forward(self, x):
        x = self.linear_relu_stack(x)



        return x



import random

from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 15),
            nn.ReLU(),
            nn.InstanceNorm1d(15),
            nn.Linear(15, 15),
            nn.Dropout(random.uniform(0,1)),
            nn.ReLU(),
            nn.InstanceNorm1d(15),


        )



        self.linear_relu_stack_2 = nn.Sequential(

            nn.Linear(15, 15),
            nn.InstanceNorm1d(15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.InstanceNorm1d(15),
            nn.ReLU(),
            nn.Linear(15, 2),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)


        logits = self.linear_relu_stack_2(x)
        return logits

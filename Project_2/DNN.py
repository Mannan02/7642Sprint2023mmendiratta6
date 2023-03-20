from torch import nn


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape((-1, self.input_dim))
        out = self.sigmoid(self.layer1(x))
        out = self.sigmoid(self.layer2(out))
        return self.layer3(out)

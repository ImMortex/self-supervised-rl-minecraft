from torch import nn

class GruNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GruNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.in_dim = input_dim
        self.out_dim = output_dim

        self.gru = nn.GRU(self.in_dim, hidden_dim, n_layers, batch_first=False, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, self.out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out)
        out = self.relu(out[:, -1])  # many to one and no negative values
        return out, h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

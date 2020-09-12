import torch
import torch.nn as nn
from torch.autograd import Variable


class StockTrend(nn.Module):
    def __init__(self, **kwargs):
        super(StockTrend, self).__init__()

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 100)
        lstm_num_layers = kwargs.get('lstm_num_layers', 1)

        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=1)

    def forward(self, sequence: torch.Tensor):
        h = Variable(torch.zeros(self.lstm.num_layers, sequence.size(0), self.lstm.hidden_size, device=sequence.device))
        c = Variable(torch.zeros(self.lstm.num_layers, sequence.size(0), self.lstm.hidden_size, device=sequence.device))
        _, (h, _) = self.lstm(sequence, (h, c))
        return self.linear(h.view(-1, self.lstm.hidden_size))

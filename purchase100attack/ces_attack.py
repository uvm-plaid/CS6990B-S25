import torch
import torch.nn as nn

class CesMiaClassifier(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(CesMiaClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.logsoftmax(out)
        return out

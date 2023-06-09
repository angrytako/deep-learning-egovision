import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

#this is the initialization line in the original code:
#models[m] = getattr(model_list, args.models[m].model)()
class Classifier(nn.Module):
    def __init__(self, dim_input, num_classes):
        super().__init__()
        #self.classifier = nn.Linear(dim_input, num_classes)
        self.model = nn.Sequential(
            nn.Linear(dim_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
#should return logits and features
#features is ignored for now
    def forward(self, x):
        return self.model(x), {}
       # return self.classifier(x), {}



class LSTM(nn.Module):
    def __init__(self, dim_input, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> input x needs to have this shape: (batch_size, seq, input_size)
        self.lstm = nn.LSTM(dim_input, hidden_size, num_layers, batch_first=True, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, num_classes, dtype=torch.float32)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(device)

        # x: (128, 5, 1024), h0: (2, n, 128)
        
        # Forward propagate lstm
  
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (128, 1024, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 8)
        return out, {}

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size, num_layers, num_heads=4,  dropout=0.2):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add a batch dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove the batch dimension
        x = x.mean(dim=0)  # Average pooling over the sequence length
        x = self.fc(x)
        return x, {}







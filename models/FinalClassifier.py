import torch
from torch import nn
from utils.logger import logger
from collections import OrderedDict

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
        #x = x.squeeze(0)  # Remove the batch dimension
        x = x.mean(dim=0)  # Average pooling over the sequence length
        x = self.fc(x)
        return x, {}


class CNN(nn.Module):
        #input_shape = nr. of channels; for spectograms is 16
    def __init__(self, input_shape, hidden_units, ouput_shape) -> None:
        super().__init__()
        self.feat_dim = 50
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*48,out_features=self.feat_dim) 
        )
        self.logits =  nn.Linear(self.feat_dim,out_features=ouput_shape)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        features = self.fully_connected(x)
        return self.logits(features), {"features": features}
    
    def get_augmentation(self, modality):
        return None, None
    
    @staticmethod
    def load(path):
        logger.info("Loading Kinetics weights CNN")
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k  # [7:]  # remove `module.`
            check_bn = name.split(".")

            if "logits" in check_bn:
                logger.info(" * Skipping Logits weight for \'{}\'".format(name))
                pass
            else:
                # print(" * Param", name)
                new_state_dict[name] = v

            # load params
        return new_state_dict
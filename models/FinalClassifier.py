import torch
from torch import nn
from utils.logger import logger
from collections import OrderedDict
import math
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -> input x needs to have this shape: (batch_size, seq, input_size)
        self.lstm = nn.LSTM(dim_input, hidden_size, num_layers, batch_first=True, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, num_classes, dtype=torch.float32)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        device = self.device

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

class PositionalEncoding(nn.Module):
# given n vectors( num_clips, because we see out video as a sentence made of 5 words ) each of one made of 1024 entries ( aka dim_input) we create a positional encoding for them
# so our input_dim = 1024 and maxlen = numclips +1

    def __init__(self, input_dim: int, dropout: float = 0.1, max_len: int = 5 +1 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # simply it creates a vector [0,1,2...,max_len]'  (max_len,1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim))  # takes only the even indexes  and creates the div term
        pe = torch.zeros(max_len, 1, input_dim)  # matrix of shape (numclips+1, 1, 1024)
		# it is now creating the matrix for to sum with the embeddings to create the positional encoding embedding vectors
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
		    # because our input has different order we change it before summing 
        
        x = x + self.pe[:x.size(0)]  # it is enoght dimension 1 because the vector is replicated till it matches 32
        return self.dropout(x)

class Transformer(nn.Module):

    def __init__(self, input_dim: int,num_classes, hidden_size: int,
                 num_layers: int,nhead=4, num_clips=5, dropout: float = 0.5):
		
		# hidden_size = dimension of feedforward 
		# nhead = number of heads in the multi attention layer
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)

        


    def forward(self, x, x_mask= None):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``  --> seq_len=num_clips
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        x_perm = torch.permute(x,(1,0,2))
        x_perm = self.pos_encoder(x_perm)
        output = self.transformer_encoder(x_perm, x_mask)
        output = torch.permute(output,(1,0,2))
        output = self.linear(output[:,-1,:])
		    
        return output, {}



    
    



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
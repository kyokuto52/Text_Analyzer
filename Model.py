import torch
import torch.nn as nn
import Util
from torch.optim import Adam

class GRUEncoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, BatchSize, drop_p = 0.5):
        super(GRUEncoder, self).__init__()
        # params: "n_" means dimension
        self.n_vocab = n_vocab     # number of unique words in vocabulary
        self.n_layers = n_layers   # number of LSTM layers 
        self.n_hidden = n_hidden   # number of hidden nodes in LSTM
        self.BatchSize = BatchSize
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.gru = nn.GRU(n_embed, n_hidden, n_layers, batch_first = True)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
                                             # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        gru_out, h = self.gru(embedded_words)         # (batch_size, seq_length, n_hidden)
        gru_out = self.dropout(gru_out)
        gru_out = torch.reshape(gru_out, (-1, self.n_hidden))    # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(gru_out)                      # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)              # (batch_size*seq_length, n_output)
        sigmoid_out = torch.reshape(sigmoid_out, (self.BatchSize, -1))  # (batch_size, seq_length*n_output)
        
        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]               # (batch_size, 1)
        
        return sigmoid_last, h
    
    
    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h

model = GRUEncoder(60, 256, 256, 256, 1, 64)

A = torch.randint(0, 60, (64, 32))
print(model(A)[0].shape)
import torch
import math

from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=512, max_len=5000):
        '''
            embed_dim: embedding dimension of the input -> NOTE: From class notes = 512
            max_len: maximum length of the input sequence
        '''
        super().__init__()
        
        # NOTE: possibly add dropout to prevent overfitting

        # Tensor of 0's 
        pe = torch.zeros(max_len, embed_dim)

        # Position Column 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Divisor -> 10000^(2i/d)
        divisor = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(torch.tensor(10000.0)) / embed_dim)
        )

        # Sin calculation for even indices
        pe[:, 0::2] = torch.sin(position * divisor)

        # Cos calculation for odd indices
        pe[:, 1::2] = torch.cos(position * divisor)

        # Unsqueeze to add batch dimension
        # NOTE: If strange behavior occurs, I have seen this used as pe = pe.unsqueeze(0).transpose(0,1) - which swaps the dimensions after unsqueezing
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
            x: input tensor of shape (seq_len, batch_size, embed_dim)
        '''
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1)]
        return x
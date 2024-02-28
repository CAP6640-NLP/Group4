from torch import nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class Encoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, feedforward_dim=2048, dropout=0.1):
        '''
            embed_dim: embedding dimension of the input (a.k.a model dimensions)-> NOTE: From class notes = 512
            num_heads: number of heads in the multihead attention
            feedforward_dim: dimension of the feedforward layer
            dropout: dropout probability
        '''
        super().__init__()
    
        # Multihead attention
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward
        self.ff = FeedForward(embed_dim=embed_dim, feedforward_dim=feedforward_dim, dropout=dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        '''
            x: input tensor of shape -> the positionally encoded sequence (gone through input embeddings + positional encoding)
                (batch_size, sequence length, embed_dim)
            mask: mask tensor of shape (batch_size, 1, 1, sequence length)
        '''
        # Multihead attention
        attention = self.attention(x, x, x, mask)

        # Add and norm
        x = self.norm1(x + self.dropout(attention))

        # Feedforward
        x2 = self.ff(x)

        # Add and norm
        x = self.norm2(x + self.dropout(x2))

        return x
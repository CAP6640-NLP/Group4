import torch
import math

from torch import nn 

class MultiHeadAttention(nn.Module): 
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        '''
            embed_dim: embedding dimension of the input (a.k.a model dimensions)-> NOTE: From class notes = 512
            num_heads: number of heads in the multi-head attention
            dropout: dropout probability
        '''
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) should be divisible by the number of heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Intialize the weight matrices for the query, key, and value
        self.q_weights = nn.Linear(embed_dim, embed_dim)
        self.k_weights = nn.Linear(embed_dim, embed_dim)
        self.v_weights = nn.Linear(embed_dim, embed_dim)

        # Intialize the weight matrix for the output
        self.output_weights = nn.Linear(embed_dim, embed_dim)

        # Intialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        ''' 
            query: query tensor of shape (batch_size, query length, embed_dim)
            key: key tensor of shape (batch_size, key length, embed_dim)
            value: value tensor of shape (batch_size, value length, embed_dim)
            mask: mask for decoder use
        '''
        batch_size = key.size(0)

        # Calculate the Q K and V tensors by multiplying the input tensors by the weight matrices
        Q = self.q_weights(query)
        K = self.k_weights(key)
        V = self.v_weights(value)

        # Reshape the Q, K, and V tensors to have the shape (batch_size, sequence length, num_heads, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # Permute switches the second and third axis
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute the scaled dot product -> Q * K^T / sqrt(head_dim)
        scaled_dot_product = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # If mask exists - fill the 0 positions with -inf (largest negative float value)
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, float('-inf'))

        # Apply the softmax function to the scaled dot product
        attention_softmax = torch.softmax(scaled_dot_product, dim=-1)

        # Multiply by value to get attention
        attention = torch.matmul(self.dropout(attention_softmax), V)

        # Reshape the attention tensor to have the shape (batch_size, sequence length, embed_dim)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads*self.head_dim)

        # Apply the output weight matrix to the attention tensor
        output = self.output_weights(attention)

        return output, attention_softmax
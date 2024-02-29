from torch import nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, feedforward_dim=2048, dropout=0.1):
        '''
        Arguments:
        
            embed_dim:          embedding dimension of the input (a.k.a model dimensions)-> NOTE: From class notes = 512
            num_heads:          number of heads in the multihead attention
            feedforward_dim:    dimension of the feedforward layer
            dropout:            dropout probability
        '''
        super().__init__()
    
        # Multihead attention
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.attention_masked = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward
        self.ff = FeedForward(embed_dim=embed_dim, feedforward_dim=feedforward_dim, dropout=dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, output, input_mask, output_mask):
        '''
        Arguments:
        
            input:              input tensor of shape (batch_size, sequence length, embed_dim)
            output:             desired output tensor of shape (batch_size, sequence length, embed_dim)
            input_mask:         mask tensor of shape (batch_size, 1, 1, sequence length)
            output_mask:        mask tensor of shape (batch_size, 1, 1, sequence length)
        
        Returns:
            output:             Word sequences after encoding and attention
            scores:             Attention scores
        '''
        # Masked multihead attention of output
        attention_masked, masked_scores = self.attention_masked(output, output, output, output_mask)

        # norm of masked multihead attention
        output = self.norm1(output + self.dropout(attention_masked))
        
        # input + output through multihead attention
        attention, scores = self.attention(output, input, input, input_mask)
        
        # norm of multihead attention
        output = self.norm2(output + self.dropout(attention))

        # Feedforward
        output2 = self.ff(output)

        # Norm for feedworard
        output = self.norm2(output + self.dropout(output2))

        return output, scores

class Decoder(nn.Module):
    def __init__(
        self, 
        len_vocab, 
        embed_dim=512, 
        layers=32, 
        num_heads=8, 
        feedforward_dim=2048, 
        dropout=0.1
    ):
        '''
        Arguments:
        
            len_vocab:              length of the vocabulary
            embed_dim:              embedding dimension of the input (a.k.a model dimensions)-> NOTE: From class notes = 512
            layers:                 number of layers to be created in the encoder
            num_heads:              number of heads in the multihead attention
            feedforward_dim:        dimension of the feedforward layer
            dropout:                dropout probability
        '''
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                feedforward_dim=feedforward_dim, 
                dropout=dropout
            ) for _ in range(layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, len_vocab)
        
    def forward(self, input, output, input_mask, output_mask):
        '''
        Arguments:
        
            input:              input tensor of shape (batch_size, sequence length, embed_dim)
            output:             desired output tensor of shape (batch_size, sequence length, embed_dim)
            input_mask:         mask tensor of shape (batch_size, 1, 1, sequence length)
            output_mask:        mask tensor of shape (batch_size, 1, 1, sequence length)
            
        Returns:
            self.out(output):   Word sequences after encoding and attention
        '''
        for layer in self.layers:
            output, score = layer(output, input, output_mask, input_mask)
        
        self.score = score
        
        return self.out(output)
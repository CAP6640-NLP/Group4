from torch import nn

class FeedForward(nn.module):
    def __init__(self, embed_dim=512, feedforward_dim=2048, dropout=0.1):
        '''
            embed_dim: embedding dimension of the input (a.k.a model dimensions)-> NOTE: From class notes = 512
            feedforward_dim: dimension of the feedforward layer (usually 4x the model dimensions)
            dropout: dropout probability
        '''
        super().__init__()
        
        # Intialize the first linear layer
        linear1 = nn.Linear(embed_dim, feedforward_dim)
        # Intialize the second linear layer
        linear2 = nn.Linear(feedforward_dim, embed_dim)
        # Intialize the dropout layer
        dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            linear1,
            nn.ReLU(),
            dropout,
            linear2,
            dropout # @TODO this is an extra dropout, do we need this?
        )

    def forward(self, x):
        '''
            x: input tensor of shape (from attention) (batch_size, sequence length, embed_dim)
        '''
        # Apply the feedforward layer to the input tensor
        x = self.ff(x)
        return x
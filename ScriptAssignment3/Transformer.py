import torch

from torch import nn
from torch import Tensor
import Encoder
import Decoder
from inputEmbedding import Embeddings

class Transformer(nn.Module):
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        input_embed: Embeddings, 
        output_embed: Embeddings, 
        input_index: int, 
        output_index: int
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed
        self.output_embed = output_embed
        self.input_index = input_index
        self.output_index = output_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", self.device)

    def create_input_mask(self, input: Tensor):
        input_mask = (input != self.input_index).unsqueeze(1).unsqueeze(2)
        
        return input_mask
        
    def create_output_mask(self, output: Tensor):
        length = output.shape[1]
        output_mask = (output != self.output_index).unsqueeze(1).unsqueeze(2)
        output_next_mask = torch.tril(torch.ones((length, length), device=self.device)).bool()
        output_mask = output_mask & output_next_mask
        
        return output_mask
    
    def forward(self, input: Tensor, output: Tensor):
        input_mask = self.create_input_mask(input)
        output_mask = self.create_output_mask(output)
        
        encoded_input = self.encoder(self.input_embed(input), input_mask)
        decoded_output = self.decoder(self.output_embed(output), encoded_input, output_mask, input_mask)
        
        return decoded_output
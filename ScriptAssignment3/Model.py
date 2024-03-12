from inputEmbedding import Embeddings
from torch import nn
import torch
from Encoder import Encoder
from Decoder import Decoder
from PositionalEncoding import PositionalEncoding
from Transformer import Transformer

class Model():
    def __init__(self):
        pass
    
    def create_transfomer_model(
        input_vocab,
        output_vocab,
        layers: int = 3,
        embed_dim: int = 256,
        feedforward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_sequence_length: int = 5000
    ):
        encoder = Encoder(
            embed_dim=embed_dim, 
            layers=layers, 
            num_heads=num_heads, 
            feedforward_dim=feedforward_dim, 
            dropout=dropout
        )
        
        decoder = Decoder(
            len_vocab=len(input_vocab),
            embed_dim=embed_dim, 
            layers=layers, 
            num_heads=num_heads, 
            feedforward_dim=feedforward_dim, 
            dropout=dropout
        )

        input_embed = Embeddings(embed_dim=embed_dim, vocab_size=len(input_vocab))
        output_embed = Embeddings(embed_dim=embed_dim, vocab_size=len(output_vocab))
        positional_enc = PositionalEncoding(embed_dim=embed_dim, max_len=max_sequence_length)

        model = Transformer(encoder,
                            decoder,
                            nn.Sequential(input_embed, positional_enc),
                            nn.Sequential(output_embed, positional_enc),
                            pad_index=input_vocab['<pad>']
                            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def do_train(model, iterator, optimizer, criterion, clip=1):
        model.train()            
        epoch_loss = 0
        print("debug123: inside do_train, iterator: ", iterator, ", type: ", type(iterator))
            
        # for i, batch in enumerate(iterator):
        for input, output in iterator:
            # print("batch is: ", batch, ", type is: ", type(batch))
            # input, output = batch
            print("debug123: input: ", input, ", output: ", output, ", type: ", type(input), ", ", type(output))
            optimizer.zero_grad()
            logits = model(input, output[:,:-1])
            expected_output = output[:,1:]
        
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                            expected_output.contiguous().view(-1))
            
            loss.backward()                
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)                
            optimizer.step()                
            epoch_loss += loss.item()

        # average loss for the epoch
        # return epoch_loss / len(iterator)
        return epoch_loss
    
    def do_evaluate(model, iterator, criterion):
        model.eval()
        epoch_loss = 0
            
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                input, output = batch

                logits = model(input, output[:,:-1])
                expected_output = output[:,1:]
                
                loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                                expected_output.contiguous().view(-1))
                epoch_loss += loss.item()
                
        # average loss for the epoch
        # return epoch_loss / len(iterator)
        return epoch_loss
    
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
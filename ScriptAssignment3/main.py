import os
from inputEmbedding import InputEmbedding, Embeddings
import torch
from torch import nn
from torch import Tensor
import pandas as pd
from Encoder import Encoder
from Decoder import Decoder
from PositionalEncoding import PositionalEncoding
from Transformer import Transformer
from Model import Model
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.functional import to_map_style_dataset
import math
from torchtext.datasets import IMDB

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.getcwd()
relative_path = os.path.join(current_dir, r'Group4\ScriptAssignment3', 'IMDBDataset_short.csv')
# df = pd.read_csv(relative_path, encoding='utf-8')

imdbdata = IMDB(root=os.path.join(current_dir, r'Group4\ScriptAssignment3'), split=("train"))

'''The embedding layer takes a long time!!!!'''
# input embedding has 2 classes, one to preprocess the data and to convert the text to a tensor
# the other is the nn.Embeddings class which is the layer in the nn

preprocess = InputEmbedding()


# # This preprocesses the text in the review column of the data frame
# df['review'] = df['review'].apply(InputEmbedding.preprocess_text)
# # This creates a string to integer dictionary that we use as our vocob for the model
# input_vocab_stoi, output_vocab_stoi = preprocess.build_vocab(df)
# preprocess.set_input_vocab(input_vocab_stoi)
# print(input_vocab_stoi)
# # Grabs our vocab size and the model dimensions, we can adjust the model dimensions as needed
# vocab_size = len(input_vocab_stoi)
# # d_model = 50

# # Example using the first review in the data frame
# first_review = df['review'][0]
# print(first_review)
# # Converts the review to a tensor
# first_review_tensor = preprocess.convert_to_tensor(first_review, input_vocab_stoi)
# # This is the nn.Embeddings layer
# embedding = Embeddings(d_model, vocab_size)
# embedded_review = embedding(first_review_tensor)

# Create input vocab and tokenize/lemmatize data
input_vocab = []
data_cleaned = []
cntr = 0
for (sentiment, text) in imdbdata:
    sentiment, text, input_vocab = InputEmbedding.preprocess_text(sentiment, text, input_vocab)
    data_cleaned.append((sentiment, text))
    cntr += 1
    if cntr > 50:
        break
# print()
# print("input_vocab is: ", input_vocab)
# print()

# Create vocab string to integer lists
input_vocab_stoi, output_vocab_stoi = preprocess.build_vocab_stoi(input_vocab)
# print()
# print("input_vocab stoi is: ", input_vocab_stoi)
# print()

# Build tensor list with tuples of (sentiment, data)
tensor_data = []
for (sentiment, text) in data_cleaned:
    tensor_data.append(preprocess.convert_to_tensor(sentiment, text, input_vocab_stoi))
# print()
# print("tensor_data is: ", tensor_data)
# print()

train_data, test_data = train_test_split(tensor_data, test_size=0.2)
train_data, val_data = train_test_split(train_data, test_size=0.25)

MAX_PADDING = 40
BATCH_SIZE = 4
preprocess.set_max_length(MAX_PADDING)
preprocess.set_pad_index(input_vocab_stoi['<pad>'])

train_iter = DataLoader(to_map_style_dataset(train_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=preprocess.pad)
# print("trainiter type: ", type(train_iter), ", len: ", train_iter.__len__())

valid_iter = DataLoader(to_map_style_dataset(val_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=preprocess.pad)
# print("validiter type: ", type(valid_iter), ", len: ", valid_iter.__len__())

test_iter = DataLoader(to_map_style_dataset(test_data), batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True, collate_fn=preprocess.pad)
# print("testiter type: ", type(test_iter), ", len: ", test_iter.__len__())

model = Model.create_transfomer_model(
    input_vocab_stoi,
    output_vocab_stoi,
    layers=3, 
    num_heads=8, 
    embed_dim=256,
    feedforward_dim=512, 
    max_sequence_length=20)
# model.cuda()

print(f'The model has {Model.count_parameters(model):,} trainable parameters')

# Train
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=input_vocab_stoi['<pad>'])

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

# loop through each epoch
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
        
    # calculate the train loss and update the parameters
    train_loss = Model.train(model, train_iter, optimizer, criterion, CLIP)

    # calculate the loss on the validation set
    valid_loss = Model.evaluate(model, valid_iter, criterion)
        
    end_time = time.time()
        
    # calculate how long the epoch took
    epoch_mins, epoch_secs = Model.epoch_time(start_time, end_time)
        
    # save the model when it performs better than the previous run
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.getcwd()
relative_path = os.path.join(current_dir, r'Group4\ScriptAssignment3', 'IMDBDataset_short.csv')
df = pd.read_csv(relative_path, encoding='utf-8')

'''The embedding layer takes a long time!!!!'''
# input embedding has 2 classes, one to preprocess the data and to convert the text to a tensor
# the other is the nn.Embeddings class which is the layer in the nn

preprocess = InputEmbedding(df)


# This preprocesses the text in the review column of the data frame
df['review'] = df['review'].apply(InputEmbedding.preprocess_text)
# This creates a string to integer dictionary that we use as our vocob for the model
input_vocab_stoi, output_vocab_stoi = preprocess.build_vocab(df)
print(input_vocab_stoi)
# Grabs our vocab size and the model dimensions, we can adjust the model dimensions as needed
vocab_size = len(input_vocab_stoi)
# d_model = 50

# # Example using the first review in the data frame
# first_review = df['review'][0]
# print(first_review)
# # Converts the review to a tensor
# first_review_tensor = preprocess.convert_to_tensor(first_review, input_vocab_stoi)
# # This is the nn.Embeddings layer
# embedding = Embeddings(d_model, vocab_size)
# embedded_review = embedding(first_review_tensor)

# print(embedded_review)


# input_tensors = []
# for data in df['review']:
#     input_tensors.append(preprocess.convert_to_tensor(data, input_vocab_stoi))
df['review'] = df['review'].apply(lambda x: preprocess.convert_to_tensor(x, input_vocab_stoi))
# print("df review: ", df['review'])
df['sentiment'] = df['sentiment'].apply(lambda x: preprocess.convert_to_tensor(x, input_vocab_stoi))
# print("df sentiment: ", df['sentiment'])

train_data, test_data = train_test_split(df, test_size=0.2)

train_data, val_data = train_test_split(train_data, test_size=0.25)

print("train data: ", train_data, ", train data len: ", len(train_data))

MAX_PADDING = 20
BATCH_SIZE = 128

test = to_map_style_dataset(train_data)
print("test123 to map: ", test)
for data in test:
    print("test data is: ", data)
    
def testfunc(batch):
    print("batch is: ", batch)
    return batch

train_iter = DataLoader(to_map_style_dataset(train_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=testfunc)
print("trainiter type: ", type(train_iter), ", len: ", len(train_iter))

valid_iter = DataLoader(to_map_style_dataset(val_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=testfunc)
print("validiter type: ", type(valid_iter), ", len: ", len(valid_iter))

test_iter = DataLoader(to_map_style_dataset(test_data), batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True, collate_fn=testfunc)
print("testiter type: ", type(test_iter), ", len: ", len(test_iter))

model = Model.create_transfomer_model(
    input_vocab_stoi,
    output_vocab_stoi,
    layers=3, 
    num_heads=8, 
    embed_dim=256,
    feedforward_dim=512, 
    max_sequence_length=50)
model.cuda()

print(f'The model has {Model.count_parameters(model):,} trainable parameters')

# Train
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=input_vocab_stoi['<padding>'])

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
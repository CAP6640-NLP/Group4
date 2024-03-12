import os
from inputEmbedding import InputEmbedding
import torch
from torch import nn
from Model import Model
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.functional import to_map_style_dataset
import math
from torchtext.datasets import IMDB

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.getcwd()
imdbdata = IMDB(root=os.path.join(current_dir, r'Group4\ScriptAssignment3'), split=("train"))
preprocess = InputEmbedding()

# Create input vocab and tokenize/lemmatize data
input_vocab = []
data_cleaned = []
cntr = 0
for (sentiment, text) in imdbdata:
    sentiment, text, input_vocab = InputEmbedding.preprocess_text(sentiment, text, input_vocab)
    data_cleaned.append((sentiment, text))
    cntr += 1
    if cntr > 50: #debug
        break

# Create vocab string to integer lists
input_vocab_stoi, output_vocab_stoi = preprocess.build_vocab_stoi(input_vocab)

# Build tensor list with tuples of (sentiment, data)
tensor_data = []
for (sentiment, text) in data_cleaned:
    tensor_data.append(preprocess.convert_to_tensor(sentiment, text, input_vocab_stoi))

# Create train, val, test split
train_data, test_data = train_test_split(tensor_data, test_size=0.2)
train_data, val_data = train_test_split(train_data, test_size=0.25)
print("debug123: traindata: ", len(train_data), ", valdata: ", len(val_data), ", testdata: ", len(test_data))

BATCH_SIZE = 8
preprocess.set_max_length(40)
preprocess.set_pad_index(input_vocab_stoi['<pad>'])

# train_iter = DataLoader(to_map_style_dataset(train_data), batch_size=BATCH_SIZE,
#                         shuffle=True, drop_last=True, collate_fn=preprocess.pad)
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=preprocess.pad)
# print("trainiter type: ", type(train_iter), ", len: ", train_iter.__len__())

val_iter = DataLoader(to_map_style_dataset(val_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=preprocess.pad)
# print("validiter type: ", type(valid_iter), ", len: ", valid_iter.__len__())

test_iter = DataLoader(to_map_style_dataset(test_data), batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True, collate_fn=preprocess.pad)
# print("testiter type: ", type(test_iter), ", len: ", test_iter.__len__())

model = Model.create_transfomer_model(
    input_vocab_stoi,
    output_vocab_stoi,
    layers=8, 
    num_heads=8, 
    embed_dim=512,
    feedforward_dim=4096, 
    max_sequence_length=20)
# model.cuda()

print(f'The model has {Model.count_parameters(model):,} trainable parameters')

# Train
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=input_vocab_stoi['<pad>'])

best_valid_loss = float('inf')

# loop through each epoch
for epoch in range(10):
    
    start_time = time.time()
    print("epoch train iter: ", train_iter)
    train_loss = Model.do_train(model, train_iter, optimizer, criterion)
    valid_loss = Model.do_evaluate(model, val_iter, criterion)
    end_time = time.time()
        
    epoch_mins, epoch_secs = Model.epoch_time(start_time, end_time)
        
    # save the model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



from torchtext.models import T5_BASE_GENERATION
from functools import partial

from torchtext.prototype.generate import GenerationUtils
from torchtext.datasets import IMDB
from torchtext.models import T5Transform
import torchdata.datapipes as dp

t5_base = T5_BASE_GENERATION
padding_idx = 0
eos_idx = 1
max_seq_len = 512
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()

def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]

def process_labels(labels, x):
    return x[1], labels[str(x[0])]

sequence_generator = GenerationUtils(model)

imdb_batch_size = 100
imdb_datapipe = IMDB(split="test")
task = "sst2 sentence"
labels = {"1": "negative", "2": "positive"}

imdb_datapipe = imdb_datapipe.map(partial(process_labels, labels))
imdb_datapipe = imdb_datapipe.map(partial(apply_prefix, task))

imdb_datapipe = imdb_datapipe.batch(imdb_batch_size)
imdb_datapipe = imdb_datapipe.rows2columnar(["review", "sentiment"])
imdb_dataloader = DataLoader(imdb_datapipe, batch_size=None)

batch = next(iter(imdb_dataloader))
input_text = batch["review"]
target = batch["sentiment"]
beam_size = 1

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, num_beams=beam_size)
output_text = transform.decode(model_output.tolist())
correct_guess = 0

print(f"Here is an example of the transormer predicting the sentiment of a review")
print(f"Input text: {input_text[0]}\n")
print(f"Transformers Prediction: {output_text[0]} Actual Sentiment: {target[0]}")

for i in range(imdb_batch_size):
    if target[i] == output_text[i]:
        correct_guess += 1

print(f"The pretrained transfromer gets {(correct_guess/imdb_batch_size) * 100}% of its predictions right on the IMBD data set")



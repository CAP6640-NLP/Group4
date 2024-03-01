import torch
import pandas as pd
import nltk
import string
from torch import nn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.nn.functional import pad

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class InputEmbedding():
    '''This class is used to preprocess the data and convert the text to a tensor for the input of the model'''
    def __init__(self):
        self.vocab_size = 0
        self.embed_dim = 0
    
    def preprocess_text(sentiment, text, vocab):
        # Make the text all lower case
        tokens = word_tokenize(text.lower())
        # remove punctuation and stopwords
        tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens if token not in string.punctuation and token not in stop_words and token not in vocab]
        # Lemmatize tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # return cleaned text as a string
        vocab.extend(tokens)
        labels = {1: 'negative', 2: 'positive'}
        return labels[sentiment], ' '.join(tokens), vocab
                        
    def build_vocab_stoi(self, vocab):                
        if 'positive' not in vocab:
            vocab.append('positive')
        if 'negative' not in vocab:
            vocab.append('negative')
        vocab.append('<pad>')
        vocab.sort()
            
        # create dicitonary of word to integer
        input_stoi = {word: i for i, word in enumerate(vocab)}
        output_stoi = {input_stoi['positive'], input_stoi['negative']}
        self.vocab_size = len(input_stoi)
        return input_stoi, output_stoi

    def convert_to_integer(self, stoi, text):
        # convert the text to a list of integers using the stoi dictionary
        tokens = word_tokenize(text)
        return [stoi[token] for token in tokens]
    
    def convert_to_tensor(self, sentiment, text, stoi):
        # convert the list of integers to a tensor
        sent_tensor = torch.tensor(self.convert_to_integer(stoi, sentiment))
        text_tensor = torch.tensor(self.convert_to_integer(stoi, text))
        return sent_tensor, text_tensor

    def pad(self, batch):
        length = self.max_length
        pad_index = self.pad_index
        # pad the tensor with padding to make it the same length as the longest tensor
        sentiment_batch = []
        text_batch = []
        for (sentiment, text) in batch:
            sentiment_batch.append(sentiment)
            text_batch.append(pad(text, (0, length - len(text)), value=pad_index))
            
        return torch.stack(sentiment_batch), torch.stack(text_batch)
    
    def set_max_length(self, length):
        self.max_length = length
        
    def set_pad_index(self, index):
        self.pad_index = index

class Embeddings(nn.Module):
    '''This class is the embendding layer in the model. Takes in the model dimensions and the vocab size to initalize'''
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # this function takes in a tensor and returns the tensor with the embedding layer applied
        print("debug123 inside embeddings: dim is: ", self.embed_dim, ", x is: ", x, ", len(x) is: ", len(x))
        return self.embed(x) * torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))

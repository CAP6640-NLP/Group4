import torch
import pandas as pd
import nltk
import string
from torch import nn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class InputEmbedding():
    '''This class is used to preprocess the data and convert the text to a tensor for the input of the model'''
    def __init__(self, dataframe):
        self.df = dataframe
        self.vocab_size = 0
        self.d_model = 0

    def preprocess_text(text):
        # Make the text all lower case
        tokens = word_tokenize(text.lower())
        # remove punctuation and stopwords
        tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
        # Lemmatize tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # return cleaned text as a string
        return ' '.join(tokens)
    
    def build_vocab(self, df):
        # get all unique words in the reviews to create a vocab
        vocab = []
        for review in df['review']:
            tokens = word_tokenize(review)
            for token in tokens:
                if token not in vocab:
                    vocab.append(token)
        vocab.sort()
        # create dicitonary of word to integer
        stoi = {word: i for i, word in enumerate(vocab)}
        self.vocab_size = len(stoi)
        return stoi

    def convert_to_integer(self, stoi, text):
        # convert the text to a list of integers using the stoi dictionary
        tokens = word_tokenize(text)
        return [stoi[token] for token in tokens]
    
    def convert_to_tensor(self, text, stoi):
        # convert the list of integers to a tensor
        return torch.tensor(self.convert_to_integer(stoi, text))

class Embeddings(nn.Module):
    '''This class is the embendding layer in the model. Takes in the model dimensions and the vocab size to initalize'''
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # this function takes in a tensor and returns the tensor with the embedding layer applied
        return self.embed(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

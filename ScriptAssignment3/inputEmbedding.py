import torch
import torchtext
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Import the csv file and change all positive and negatives to 1's and o's
df = pd.read_csv('C:\GitHubRepo\Group4\ScriptAssignment3\IMDBDataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

def preprocess_text(text):
    # Make the text all lower case
    tokens = word_tokenize(text.lower())
    # remove punctuation and stopwords
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    #tokens = 
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # return cleaned text as a string
    return ' '.join(tokens)
 
def build_vocab(text):
    vocab = list(set(text))
    vocab.sort()
    stoi = {word:i for i, word in enumerate(vocab)}
    
    return stoi
df['review'] = df['review'].apply(preprocess_text)

df['review'] = df['review'].apply(build_vocab)
print(df)
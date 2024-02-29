import os
from inputEmbedding import InputEmbedding, Embeddings
import pandas as pd

current_dir = os.getcwd()
relative_path = os.path.join(current_dir, 'ScriptAssignment3', 'IMDBDataset.csv')
df = pd.read_csv(relative_path)

'''The embedding layer takes a long time!!!!'''
# input embedding has 2 classes, one to preprocess the data and to convert the text to a tensor
# the other is the nn.Embeddings class which is the layer in the nn

preprocess = InputEmbedding(df)


# This preprocesses the text in the review column of the data frame
df['review'] = df['review'].apply(InputEmbedding.preprocess_text)
# This creates a string to integer dictionary that we use as our vocob for the model
stoi = preprocess.build_vocab(df)
print(stoi)
# Grabs our vocab size and the model dimensions, we can adjust the model dimensions as needed
vocab_size = len(stoi)
d_model = 50

# Example using the first review in the data frame
first_review = df['review'][0]
print(first_review)
# Converts the review to a tensor
first_review_tensor = preprocess.convert_to_tensor(first_review, stoi)
# This is the nn.Embeddings layer
embedding = Embeddings(d_model, vocab_size)
embedded_review = embedding(first_review_tensor)

print(embedded_review)
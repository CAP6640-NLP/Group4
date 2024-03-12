# Python Script for Next Sentence Prediction using GPT API

# To fix the code, you would need to:
# 1. Import the correct libraries for interacting with the GPT model.
# 2. Ensure that they have proper access to the GPT model, whether through local installation or an appropriate API.
# 3. Preprocess the input sentences, ensuring proper separation between them.
# 4. Use the correct method or function for predicting the next sentence based on the input.

# Import necessary libraries
#import GPTAPI
import openai as GPTAPI

# API Key for Open AI
# Don't necessarily like exposing this in the code but for the sake of this assignment I will
API_KEY = "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"



# Initialize the GPT API client
# gpt_client = GPTAPI.Client(api_key='YOUR_API_KEY')

# Attach the API Key to the openai module 
GPTAPI.api_key = API_KEY


# Define input sentences
input_sentence = "The cat is sleeping on the sofa."
next_sentence = "It looks very comfortable."


# Concatenate input sentences
input_text = input_sentence + next_sentence

# Use the GPT API for next sentence prediction
predicted_next_sentence = gpt_client.predict_next_sentence(input_text)


# Display the predicted next sentence
print("Predicted Next Sentence:", predicted_next_sentence)
# Python Script for Next Sentence Prediction using GPT API

# Import necessary libraries
from openai import OpenAI

super_secret_api_file = "OpenAI_API_KEY.txt"

with open(super_secret_api_file, 'r') as file:
    key = file.readline()

# Initialize the GPT API client
# This code automatically grabs the API key from the environment variable
client = OpenAI(api_key=key.strip())

# Define input sentences - user input on CMD Line
input_text = input("Enter your sentences: ")
print("\n")

# Make the API request to the legacy Completions API
response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt= input_text,
  max_tokens=100
)

print(response.choices[0].text.strip())
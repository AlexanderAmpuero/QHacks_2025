import openai

#Set your API key
openai.api_key = "INSERTAIKEY"

#Create a completion request
completion = openai.ChatCompletion.create(
  model="gpt-4-0613",  # Replace "gpt-4o-mini" with the correct model name, if necessary
  messages=[
    {"role": "user", "content": "is sasha ampuro and martin in a relationship"}
  ]
)

#Print the response
print(completion["choices"][0]["message"]["content"])
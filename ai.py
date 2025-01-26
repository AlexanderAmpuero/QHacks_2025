import os
import openai

client = openai.OpenAI(
  api_key=os.getenv('OPEN_AI_API')
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Using this dictionary as a rubric, 0.3: Very Little, 0.5:Little, 0.6 balance, 0.7Much 1Very Much you are given the amount of times someone moved in a presentation, their head being 0.1/s, arms being 0.7/s and moving around the stage at 0.6/s give me feed back of an overall score and what I should work on"}
  ]
)

print(completion.choices[0].message)
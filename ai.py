import os
import openai
import streamlit as st


def get_feedback(scores): 
  prompt = f'You are giving feedback to a users presentation, specifically how much they move. you are given a dictionary {scores} which has the following:\
  - the amount of times the speakers head turns per second\
  - the amount of times the speaker hides their hands per second\
  - the amount of time the speaker stood still, throughout the entire presentation\
  - the total time of the presentation in seconds.\
  the rubric is <0.3 is very little, <0.5 little, <0.6 balanced amount <0.7 a bit much amnd >0.7 is too much\
  you are giving an overall feedback as well as areas that they could improve on. you do not need to state the values, but just give suggestions\
  rank it in what you think they need to work on most'
  openai.api_key = st.secrets['api_keys']

  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
      {"role": "user", "content": prompt}
    ]
  )

  return completion.choices[0].message.content

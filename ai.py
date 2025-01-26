import os
import openai
import streamlit as st

# Ensure you have your Streamlit secrets configured with 'api_keys'
openai.api_key = st.secrets['api_keys']

def get_feedback(scores): 
    # Construct the prompt
    prompt = (
        f"You are giving feedback to a user's presentation, specifically how much they move. "
        f"You are given a dictionary {scores} which has the following:\n"
        f"- The amount of times the speaker's head turns per second\n"
        f"- The amount of times the speaker hides their hands per second\n"
        f"- The amount of time the speaker stood still throughout the entire presentation\n"
        f"- The total time of the presentation in seconds.\n\n"
        f"The rubric is:\n"
        f"<0.3 is very little, <0.5 little, <0.6 balanced amount, <0.7 a bit much, >0.7 is too much.\n\n"
        f"Provide overall feedback and areas for improvement, ranked by importance. Do not state the values explicitly."
    )

    try:
        # Call OpenAI's ChatCompletion API
        completion = openai.ChatCompletion.create(
            model="gpt-4",  # Use a valid model name
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Extract and return the response
        return completion["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        return f"An error occurred: {e}"

import openai
import streamlit as st

# Set OpenAI API key
openai.api_key = st.secrets["api_keys"]["OPEN_AI_API"]

def get_feedback(scores):
    # Constructing prompt
    prompt = (
        f"You are giving feedback to a user's presentation. You are given the following:\n"
        f"- Head turns per second: {scores.get('head_turns', 'unknown')}\n"
        f"- Hands hidden per second: {scores.get('hands_hidden', 'unknown')}\n"
        f"- Time standing still: {scores.get('time_still', 'unknown')}\n"
        f"- Total presentation time: {scores.get('total_time', 'unknown')} seconds.\n\n"
        f"Provide overall feedback and rank the areas needing improvement."
    )
    try:
        # Making API call
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use an actual available model like gpt-4 or gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are an expert in presentation analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        # Returning content
        return response["choices"][0]["message"]["content"]
    except openai.error.AuthenticationError:
        return "Authentication failed. Check your API key."
    except openai.error.OpenAIError as e:
        return f"OpenAI API error: {e}"

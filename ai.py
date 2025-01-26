import openai

def get_feedback(scores): 
    prompt = f'You are giving feedback to a user\'s presentation, specifically how much they move. You are given a dictionary {scores} which has the following:\
    - the amount of times the speaker\'s head turns per second\
    - the amount of times the speaker hides their hands per second\
    - the amount of time the speaker stood still throughout the entire presentation\
    - the total time of the presentation in seconds.\
    The rubric is: <0.3 is very little, <0.5 little, <0.6 balanced amount, <0.7 a bit much, and >0.7 is too much.\
    You are giving an overall feedback as well as areas that they could improve on. You do not need to state the values, but just give suggestions.\
    Rank it in what you think they need to work on most.'

    openai.api_key = st.secrets['api_keys']['OPEN_AI_API']

    completion = openai.ChatCompletion.create(
        model="gpt-4",  # Corrected model name
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message['content']

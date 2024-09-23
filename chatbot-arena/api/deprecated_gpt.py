from openai import OpenAI
import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import pandas as pd

load_dotenv()
openai_api_key = os.environ.get('OPENAI_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

df = pd.read_csv("~/projects/llm-eval/chatbot-arena/chatbot_arena_conversations.csv")

def openai_response(input_text: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        messages = [
        {
            "role": "user",
            "content": input_text,
        }
        ],
        model="gpt-3.5-turbo-0125",
    )
        
    return response.choices[0].message.content.strip() #type: ignore

sampled_df = df.sample(frac=0.25, random_state=42)

conversation_a = sampled_df['conversation_a'].tolist()
conversation_b = sampled_df['conversation_b'].tolist()
winner = sampled_df['winner'].tolist()

result_df = pd.read_csv("~/projects/llm-eval/chatbot-arena/results.csv")

for i in range(len(winner)):
    prompt = f"""I will give you a conversation between a user and two separate models. You have to decide which model is better based on the responses. 

    The conversation is as follows:
    Model A:
    {conversation_a[i]}

    Model B:
    {conversation_b[i]}

    Now, let's analyze the responses.

    Respond with "model_a" if you think Model A is better, "model_b" if you think Model B is better, or "tie" if you think they are equal. Make sure to keep the formatting as shown.
    """

    response = openai_response(prompt)
    result_df["response"][i] = response

    result_df["dataset response"][i] = winner[i]

    
    if(winner[i] == "tie (bothbad)"):
        if(response == "tie"):
            result_df["correct"][i] = 1
        else:
            result_df["correct"][i] = 0
    else:
        if (response == winner[i]):
            result_df["correct"][i] = 1
        else:
            result_df["correct"][i] = 0
    print(f"Iter: {i+1}" + response + "\n" + winner[i] + "\n" + str(result_df["correct"][i]) + "\n" + sum(result_df["correct"])/len(result_df.dropna()) + "\n" + "-------------------")

    result_df.to_csv('~/Desktop/projects/llm-eval/chatbot-arena/results.csv', index=False)
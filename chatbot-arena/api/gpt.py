from openai import OpenAI, AzureOpenAI
import openai
import os
from dotenv import load_dotenv
import pandas as pd
import random
import platform
import logging

load_dotenv()
openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
openai_api_version = os.environ.get('OPENAI_API_VERSION')
openai_azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

if platform.system() == "Windows":
    df = pd.read_csv("C:/~/llm-eval/chatbot-arena/filtered_chatbot_arena.csv")
elif platform.system() == "Darwin":
    df = pd.read_csv("~/llm-eval/chatbot-arena/filtered_chatbot_arena.csv")
else:
    raise ValueError("Unsupported operating system.")

def openai_response(input_text: str) -> str:
    client = AzureOpenAI(
        api_key=openai_api_key,  
        azure_endpoint=openai_azure_endpoint or "",
        api_version=openai_api_version
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user", 
                "content": input_text
            },
        ],
        model="gpt-4-turbo-2024-04-09",
    )

    if response.choices:
        message_content = response.choices[0].message.content
        if message_content:
            return message_content.strip()
        else:
            logging.warning(f"The response message content is None: {response}")
            print("The response message content is None.")
            return "No content returned from OpenAI."
    else:
        logging.warning(f"The response has no choices: {response}")
        print("The response has no choices.")
        return "No choices returned from OpenAI."

conversation_a = df['conversation_a'].tolist()
conversation_b = df['conversation_b'].tolist()
winner = df['winner'].tolist()

if platform.system() == "Windows":
    result_df = pd.read_csv("C:/~/llm-eval/chatbot-arena/runs/filtered_results.csv")
elif platform.system() == "Darwin":
    result_df = pd.read_csv("~/llm-eval/chatbot-arena/runs/filtered_results.csv")
else:
    raise ValueError("Unsupported operating system.")

null_indices = result_df[result_df.isnull().any(axis=1)].index
for i in null_indices:
    flip = random.randint(0, 1)
    print(f"Index: {i}, Flip: {flip}")
    if flip == 0:
        prompt = f"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of their response. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Finally, indicate your verdict with "model_a" if you think Model A is better, "model_b" if you think Model B is better, or "tie" if you think they are equal. Make sure to keep the formatting as shown.
    
        Model A:
        {conversation_a[i]}

        Model B:
        {conversation_b[i]}
        """
        
    else:
        prompt = f"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of their response. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Finally, indicate your verdict with "model_a" if you think Model A is better, "model_b" if you think Model B is better, or "tie" if you think they are equal. Make sure to keep the formatting as shown.
    
        Model A:
        {conversation_b[i]}

        Model B:
        {conversation_a[i]}
        """

    try:
        response = openai_response(prompt)
        final_prompt = f'''Given this response:
        {response}

        SIMPLY return "model_a" if the verdict is model_a, "model_b" if the verdict is model_b, or "tie" if the verdict is tie
        '''
        final_response = openai_response(final_prompt)
    except openai.BadRequestError as e:
        print(f"Error at index {i}")
        continue

    final_response = openai_response(final_prompt)

    result_df.at[i, "response"] = response
    result_df.at[i, "final response"] = final_response
    result_df.at[i, "dataset response"] = winner[i]
    result_df.at[i, "flipped"] = "True" if flip == 1 else "False"    

    if winner[i] == "tie (bothbad)":
        result_df.at[i, "correct"] = 1 if final_response == "tie" else 0
    elif winner[i] == "tie":
        result_df.at[i, "correct"] = 1 if final_response == winner[i] else 0
    else:
        if flip == 0:
            result_df.at[i, "correct"] = 1 if final_response == winner[i] else 0
        else:  
            result_df.at[i, "correct"] = 0 if final_response == winner[i] else 1


    print(f"Iter: {i+1}" + "\n" + final_response + "\n" + winner[i] + "\n" + str(result_df["correct"][i]) + "\n" + str(sum(result_df["correct"].dropna())/len(result_df.dropna())) + "\n" + "-------------------")

    if platform.system() == "Windows":
        result_df.to_csv("C:/~/llm-eval/chatbot-arena/runs/filtered_results.csv", index=False)
    elif platform.system() == "Darwin":
        result_df.to_csv("~/llm-eval/chatbot-arena/runs/filtered_results.csv", index=False)
    else:
        raise ValueError("Unsupported operating system.")
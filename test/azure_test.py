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
        
    # return response.choices[0].message.content.strip()  # type: ignore

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

if __name__ == '__main__':
    print(openai_response("hello"))
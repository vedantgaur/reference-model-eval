import os
import argparse
from dotenv import load_dotenv
import datasets
import json
from together import Together
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import ReadTimeout, ConnectionError

load_dotenv()
api_key = os.environ.get("TOGETHER_API_KEY")
client = Together(api_key=api_key)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ReadTimeout, ConnectionError))
)
def together_response(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except (ReadTimeout, ConnectionError) as e:
        print(f"Error occurred: {e}. Retrying...")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def main(model, generator):
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    examples = [dict(example) for example in eval_set]

    for i, example in enumerate(examples):
        try:
            print(f"Processing example {i+1} of {len(examples)}")
            example["output"] = together_response(str(model), example["instruction"])
            example["generator"] = str(generator)
            print(f"Completed example {i+1}")

            with open(f"runs/{str(generator)}_alpaca_outputs.json", "w") as f:
                json.dump(examples, f)
        except Exception as e:
            print(f"Failed to process example {i+1}: {e}")
            continue

    print("All examples processed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Specify the model for together_response")
    parser.add_argument("--generator", help="Specify the generator for example['generator']")
    args = parser.parse_args()

    main(args.model, args.generator)
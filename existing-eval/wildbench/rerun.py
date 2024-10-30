import os
import json
import logging
import time
import pickle
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv

test_models = [
    'SELM-Zephyr-7B-iter-3',
    'Qwen1.5-7B-Chat@together',
    'Yi-1.5-34B-Chat',
    'Mistral-7B-Instruct-v0.2',
    'Llama-2-7b-chat-hf',
    'Yi-1.5-9B-Chat',
    'Yi-1.5-6B-Chat',
    'glm-4-9b-chat'
]

tiered_models = [
    "gpt-4-turbo-2024-04-09",
    "mistral-large-2402",
    "Llama-3-Instruct-8B-SimPO",
    "neo_7b_instruct_v0.1-ExPO"
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
openai_api_version = os.environ.get('OPENAI_API_VERSION')
openai_azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Constants
MAX_RETRIES = 20
RETRY_DELAY = 1.0
CALLS_PER_MINUTE = 50 
MAX_WORKERS = 5

@dataclass
class EvaluationTask:
    session_id: str
    test_model: str
    reference_model: str
    history: List[Dict]
    query: str
    response_a: str
    response_b: str
    evaluation_type: str  # 'randomized' or 'tiered'
    tier: Optional[int] = None

class ProgressTracker:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.completed_tasks: Set[str] = self._load_progress()

    def _load_progress(self) -> Set[str]:
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                return pickle.load(f)
        return set()

    def save_progress(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.completed_tasks, f)

    def mark_completed(self, task_id: str):
        self.completed_tasks.add(task_id)
        self.save_progress()

    def is_completed(self, task_id: str) -> bool:
        return task_id in self.completed_tasks

def load_prompt_template(file_path: str) -> str:
    """Load the evaluation prompt template from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
            logging.debug(f"Loaded template length: {len(template)}")
            return template
    except Exception as e:
        logging.error(f"Error loading prompt template: {str(e)}", exc_info=True)
        raise

def load_json_file(file_path: str) -> List[Dict]:
    """Load a JSON file containing model outputs."""
    with open(file_path, 'r') as f:
        return json.load(f)

# def format_conversation_history(history: List[Dict]) -> str:
#     """Format conversation history into a string."""
#     formatted_history = []
    
#     # Add debug logging
#     logging.debug(f"Raw history data: {history}")
    
#     # Handle different possible history formats
#     for turn in history:
#         try:
#             if isinstance(turn, dict):
#                 # Handle dictionary format
#                 if 'user' in turn:
#                     formatted_history.append(f"User: {turn['user']}")
#                 if 'assistant' in turn:
#                     formatted_history.append(f"Assistant: {turn['assistant']}")
#             elif isinstance(turn, str):
#                 # Handle string format
#                 formatted_history.append(turn)
#             else:
#                 logging.warning(f"Unexpected turn format: {type(turn)}")
#                 formatted_history.append(str(turn))
                
#         except Exception as e:
#             logging.error(f"Error formatting turn {turn}: {str(e)}")
#             continue
    
#     formatted_result = "\n".join(formatted_history) if formatted_history else ""
#     logging.debug(f"Formatted history: {formatted_result}")
#     return formatted_result

def format_conversation_history(history: List[str]) -> str:
    """Format conversation history into a string."""
    if not history:
        return ""
        
    # If history is a list of strings, just join them
    return "\n".join(history) if isinstance(history[0], str) else ""

def create_evaluation_prompt(
    prompt_template: str,
    task: EvaluationTask,
    checklist: str = "1. Accuracy\n2. Relevance\n3. Clarity\n4. Completeness"
) -> str:
    """Create the full evaluation prompt by filling in the template."""
    formatted_history = format_conversation_history(task.history)
    
    # Convert response_a and response_b to strings if they're lists
    response_a = "\n".join(task.response_a) if isinstance(task.response_a, list) else str(task.response_a)
    response_b = "\n".join(task.response_b) if isinstance(task.response_b, list) else str(task.response_b)
    
    # Replace template variables with actual content
    prompt = prompt_template.replace("{$history}", formatted_history)
    prompt = prompt.replace("{$user_query}", str(task.query))
    prompt = prompt.replace("{$candidate_A}", response_a)
    prompt = prompt.replace("{$candidate_B}", response_b)
    prompt = prompt.replace("{$checklist}", checklist)
    
    # Debug logging
    logging.debug(f"Formatted history: {formatted_history}")
    logging.debug(f"Response A type: {type(response_a)}")
    logging.debug(f"Response B type: {type(response_b)}")
    
    return prompt

def create_evaluation_task(
    test_item: Dict,
    ref_item: Dict,
    evaluation_type: str,
    tier: Optional[int] = None
) -> EvaluationTask:
    """Create a single evaluation task with debug logging."""
    logging.debug(f"Creating task from test item: {test_item}")
    logging.debug(f"Reference item: {ref_item}")
    
    # Get chat history - handle both list and string formats
    chat_history = test_item.get('chat_history', [])
    if isinstance(chat_history, str):
        chat_history = [chat_history]
    
    # Get model outputs - handle both list and string formats
    test_output = test_item.get('output', '')
    ref_output = ref_item.get('output', '')
    
    return EvaluationTask(
        session_id=test_item.get('session_id', ''),
        test_model=test_item.get('generator', ''),
        reference_model=ref_item.get('generator', ''),
        history=chat_history,
        query=test_item.get('model_input', ''),
        response_a=test_output,
        response_b=ref_output,
        evaluation_type=evaluation_type,
        tier=tier
    )

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def call_azure_openai(
    client: AzureOpenAI,
    prompt: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY
) -> str:
    """Make an API call to Azure OpenAI with retry logic and rate limiting."""
    for attempt in range(max_retries):
        logging.info(f"Attempt {attempt + 1} of {max_retries}")  # Log current attempt number
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-turbo-2024-04-09"
            )
            
            if response.choices:
                message_content = response.choices[0].message.content
                if message_content:
                    return message_content.strip()
            
            raise ValueError("No valid content in response")
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay * (2 ** attempt))
            
    return "Failed to get response after all retries"

def process_single_evaluation(
    client: AzureOpenAI,
    prompt_template: str,
    task: EvaluationTask,
    progress_tracker: ProgressTracker
) -> Optional[Dict]:
    """Process a single evaluation task with improved error handling."""
    task_id = f"{task.evaluation_type}_{task.session_id}_{task.test_model}_{task.reference_model}"
    
    if progress_tracker.is_completed(task_id):
        logging.info(f"Skipping completed task: {task_id}")
        return None
    
    try:
        logging.info(f"Processing task: {task_id}")
        logging.debug(f"Task details: {vars(task)}")
        
        # Create prompt and ensure all components are strings
        prompt = create_evaluation_prompt(prompt_template, task)
        
        if not isinstance(prompt, str):
            raise ValueError(f"Generated prompt is not a string: {type(prompt)}")
            
        evaluation = call_azure_openai(client, prompt)
        # Removed logging of evaluation result
        progress_tracker.mark_completed(task_id)
        return {
            'session_id': task.session_id,
            'test_model': task.test_model,
            'reference_model': task.reference_model,
            'evaluation_type': task.evaluation_type,
            'tier': task.tier,
            'evaluation': evaluation  # Still return the evaluation, but do not log it
        }
        
    except Exception as e:
        logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        return None
    
def create_evaluation_task(
    test_item: Dict,
    ref_item: Dict,
    evaluation_type: str,
    tier: Optional[int] = None
) -> EvaluationTask:
    """Create a single evaluation task with debug logging."""
    logging.debug(f"Creating task from test item: {test_item}")
    logging.debug(f"Reference item: {ref_item}")
    
    # Ensure chat_history is properly formatted
    chat_history = test_item.get('chat_history', [])
    if isinstance(chat_history, str):
        chat_history = [{'user': chat_history}]
    
    return EvaluationTask(
        session_id=test_item.get('session_id', ''),
        test_model=test_item.get('generator', ''),
        reference_model=ref_item.get('generator', ''),
        history=chat_history,
        query=test_item.get('model_input', ''),
        response_a=test_item.get('output', ''),
        response_b=ref_item.get('output', ''),
        evaluation_type=evaluation_type,
        tier=tier
    )

def create_evaluation_tasks(
    test_outputs: List[Dict],
    reference_outputs: List[Dict],
    evaluation_type: str,
    tier: Optional[int] = None
) -> List[EvaluationTask]:
    """Create evaluation tasks with improved error handling."""
    tasks = []
    
    logging.debug(f"Creating tasks for {len(test_outputs)} test outputs")
    
    for i, (test_item, ref_item) in enumerate(zip(test_outputs, reference_outputs)):
        try:
            task = create_evaluation_task(test_item, ref_item, evaluation_type, tier)
            tasks.append(task)
        except Exception as e:
            logging.error(f"Error creating task {i}: {str(e)}")
            continue
    
    logging.debug(f"Created {len(tasks)} tasks")
    return tasks

def process_evaluations(
    client: AzureOpenAI,
    prompt_template: str,
    tasks: List[EvaluationTask],
    results_path: str,
    progress_tracker: ProgressTracker
) -> None:
    """Process evaluation tasks in parallel with progress tracking."""
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(
                process_single_evaluation,
                client,
                prompt_template,
                task,
                progress_tracker
            ): task for task in tasks
        }
        
        with tqdm(total=len(tasks)) as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)
    
    # Append results to existing file if it exists
    if os.path.exists(results_path):
        existing_results = load_json_file(results_path)
        results.extend(existing_results)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=openai_api_key,
        azure_endpoint=openai_azure_endpoint,
        api_version=openai_api_version
    )
    
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize components
        client = AzureOpenAI(
            api_key=openai_api_key,
            azure_endpoint=openai_azure_endpoint,
            api_version=openai_api_version
        )
        
        prompt_template = load_prompt_template("prompt_template.txt")
        progress_tracker = ProgressTracker("evaluation_progress.pkl")
        
        # Test with a single file first
        base_path = "results"
        test_model = test_models[0]
        model_path = os.path.join(base_path, "tiered", test_model)
        
        if os.path.exists(model_path):
            # Load test files
            test_outputs_path = os.path.join(model_path, "model_outputs.json")
            reference_outputs_path = os.path.join(
                base_path,
                "tiered",
                "tiered_references",
                "tiered_reference_outputs_1.json"
            )
            
            # Load and inspect data
            test_outputs = load_json_file(test_outputs_path)
            reference_outputs = load_json_file(reference_outputs_path)
            
            logging.debug(f"Sample test output: {test_outputs[0] if test_outputs else 'No test outputs'}")
            logging.debug(f"Sample reference output: {reference_outputs[0] if reference_outputs else 'No reference outputs'}")
            
            # Create and process tasks
            tasks = create_evaluation_tasks(
                test_outputs[:1],  # Test with first item
                reference_outputs[:1],
                evaluation_type='tiered',
                tier=1
            )
            
            if tasks:
                result = process_single_evaluation(
                    client,
                    prompt_template,
                    tasks[0],
                    progress_tracker
                )
                logging.info(f"Test evaluation result: {result}")
    
        # Process randomized evaluations
        randomized_path = os.path.join(base_path, "randomized")
        for test_model in os.listdir(randomized_path):
            # print(f"RANDOM TEST MODEL: {test_model}")
            model_path = os.path.join(randomized_path, test_model)
            if os.path.isdir(model_path):
                test_outputs_path = os.path.join(model_path, "model_outputs.json")
                reference_outputs_path = os.path.join(randomized_path, "randomized_reference_outputs.json")
                results_path = os.path.join(model_path, "evaluations.json")
                
                test_outputs = load_json_file(test_outputs_path)
                reference_outputs = load_json_file(reference_outputs_path)
                
                tasks = create_evaluation_tasks(
                    test_outputs,
                    reference_outputs,
                    evaluation_type='randomized'
                )
                
                logging.info(f"Processing randomized evaluation for {test_model}")
                process_evaluations(
                    client,
                    prompt_template,
                    tasks,
                    results_path,
                    progress_tracker
                )
        
        tiered_path = os.path.join(base_path, "tiered")
        for test_model in test_models:
            # print(f"TEST MODEL: {test_model}")
            model_path = os.path.join(tiered_path, test_model)
            # print(f"MODEL PATH: {model_path}")
            if os.path.isdir(model_path):
                test_outputs_path = os.path.join(model_path, "model_outputs.json")
                # print(f"PATH: {test_outputs_path}")
                test_outputs = load_json_file(test_outputs_path)
                
                for tier in range(1, 5):
                    reference_outputs_path = os.path.join(
                        tiered_path,
                        "tiered_references",
                        f"tiered_reference_outputs_{tier}.json"
                    )
                    results_path = os.path.join(model_path, f"tier_{tier}", "evaluations.json")
                    
                    reference_outputs = load_json_file(reference_outputs_path)
                    
                    tasks = create_evaluation_tasks(
                        test_outputs,
                        reference_outputs,
                        evaluation_type='tiered',
                        tier=tier
                    )
                    
                    logging.info(f"Processing tiered evaluation for {test_model} - Tier {tier}")
                    process_evaluations(
                        client,
                        prompt_template,
                        tasks,
                        results_path,
                        progress_tracker
                    )
    except Exception as e:
            logging.error(f"Error in main: {str(e)}", exc_info=True)
            raise
    
if __name__ == "__main__":
    main()

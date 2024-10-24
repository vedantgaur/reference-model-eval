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
MAX_RETRIES = 3
RETRY_DELAY = 1.0
CALLS_PER_MINUTE = 50  # Adjust based on your rate limits
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
    with open(file_path, 'r') as f:
        return f.read()

def load_json_file(file_path: str) -> List[Dict]:
    """Load a JSON file containing model outputs."""
    with open(file_path, 'r') as f:
        return json.load(f)

def format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history into a string."""
    formatted_history = []
    for turn in history:
        formatted_history.append(f"User: {turn['user']}")
        if 'assistant' in turn:
            formatted_history.append(f"Assistant: {turn['assistant']}")
    return "\n".join(formatted_history)

def create_evaluation_prompt(
    prompt_template: str,
    task: EvaluationTask,
    checklist: str = "1. Accuracy\n2. Relevance\n3. Clarity\n4. Completeness"
) -> str:
    """Create the full evaluation prompt by filling in the template."""
    formatted_history = format_conversation_history(task.history)
    
    return prompt_template.format(
        history=formatted_history,
        user_query=task.query,
        candidate_A=task.response_a,
        candidate_B=task.response_b,
        checklist=checklist
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
    """Process a single evaluation task."""
    task_id = f"{task.evaluation_type}_{task.session_id}_{task.test_model}_{task.reference_model}"
    
    if progress_tracker.is_completed(task_id):
        logging.info(f"Skipping completed task: {task_id}")
        return None
    
    try:
        prompt = create_evaluation_prompt(prompt_template, task)
        evaluation = call_azure_openai(client, prompt)
        
        result = {
            'session_id': task.session_id,
            'test_model': task.test_model,
            'reference_model': task.reference_model,
            'evaluation_type': task.evaluation_type,
            'tier': task.tier,
            'evaluation': evaluation
        }
        
        progress_tracker.mark_completed(task_id)
        return result
        
    except Exception as e:
        logging.error(f"Error processing task {task_id}: {str(e)}")
        return None

def create_evaluation_tasks(
    test_outputs: List[Dict],
    reference_outputs: List[Dict],
    evaluation_type: str,
    tier: Optional[int] = None
) -> List[EvaluationTask]:
    """Create evaluation tasks from test and reference outputs."""
    tasks = []
    for test_item, ref_item in zip(test_outputs, reference_outputs):
        tasks.append(EvaluationTask(
            session_id=test_item['session_id'],
            test_model=test_item['generator'],
            reference_model=ref_item['generator'],
            history=test_item['chat_history'],
            query=test_item['model_input'],
            response_a=test_item['output'],
            response_b=ref_item['output'],
            evaluation_type=evaluation_type,
            tier=tier
        ))
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
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=openai_api_key,
        azure_endpoint=openai_azure_endpoint,
        api_version=openai_api_version
    )
    
    # Load prompt template
    prompt_template = load_prompt_template("prompt_template.txt")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker("evaluation_progress.pkl")
    
    base_path = "results"
    
    # Process randomized evaluations
    randomized_path = os.path.join(base_path, "randomized")
    for test_model in os.listdir(randomized_path):
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
    
    # Process tiered evaluations
    tiered_path = os.path.join(base_path, "tiered")
    for test_model in os.listdir(tiered_path):
        model_path = os.path.join(tiered_path, test_model)
        if os.path.isdir(model_path):
            test_outputs_path = os.path.join(model_path, "model_outputs.json")
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

if __name__ == "__main__":
    main()
#!/bin/bash

MODEL_PRETTY_NAME=$1

# Submit evaluation job to OpenAI
python src/openai_batch_eval/submit_batch_job.py $MODEL_PRETTY_NAME

echo "Evaluation job submitted for $MODEL_PRETTY_NAME"
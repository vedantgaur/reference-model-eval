#!/bin/bash

HF_MODEL_ID=$1
MODEL_PRETTY_NAME=$2
NUM_GPUS=$3

# Run inference on WildBench
python src/run_inference.py --model $HF_MODEL_ID --name $MODEL_PRETTY_NAME --num_gpus $NUM_GPUS

# Submit to OpenAI for eval (WB-Score)
bash evaluation/run_score_eval_batch.sh $MODEL_PRETTY_NAME

# Check the batch job status
python src/openai_batch_eval/check_batch_status_with_model_name.py $MODEL_PRETTY_NAME

# Show the table
bash leaderboard/show_eval.sh score_only
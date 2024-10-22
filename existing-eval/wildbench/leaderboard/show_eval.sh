#!/bin/bash

MODE=$1

if [ "$MODE" == "score_only" ]; then
    python leaderboard/show_table.py --mode main
else
    python leaderboard/show_table.py --mode main
    python leaderboard/show_table.py --mode taskwise_score
fi
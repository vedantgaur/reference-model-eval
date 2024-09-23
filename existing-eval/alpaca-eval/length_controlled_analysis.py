import json

with open('existing-eval/alpaca-eval/runs/mistral-8x7b_alpaca_git_outputs.json', 'r') as f1, open('existing-eval/alpaca-eval/runs/mistral-8x7b_alpaca_outputs.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

longer_outputs = []
shorter_outputs = []

for item1, item2 in zip(data1, data2):
    if len(item1['output']) >= len(item2['output']):
        longer_outputs.append(item1)
        shorter_outputs.append(item2)
    else:
        longer_outputs.append(item2)
        shorter_outputs.append(item1)

with open('existing-eval/alpaca-eval/runs/mistral-8x7b_longer.json', 'w') as f:
    json.dump(longer_outputs, f, indent=2)

with open('existing-eval/alpaca-eval/runs/mistral-8x7b_shorter.json', 'w') as f:
    json.dump(shorter_outputs, f, indent=2)
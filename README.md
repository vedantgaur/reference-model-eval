# The Progress Illusion: Revisiting Meta-Evaluation Standards

> **Official repository for the paper: "The Progress Illusion: Revisiting meta-evaluation standards of LLM evaluators"**
> *Tianruo Rose Xu*, Vedant Gaur*, Liu Leqi, Tanya Goyal*
> **EMNLP 2025 Findings**

## 📉 Overview

Human evaluation is the gold standard for LLMs, but it is expensive. Consequently, the community relies on **LLM-as-a-judge** evaluators (e.g., AlpacaEval, WildBench, Arena-Hard) which validate themselves by reporting high correlation with human rankings (e.g., Chatbot Arena).

However, our research reveals a critical flaw in this validation process: **The meta-evaluation setting differs substantially from practical usage.** 

* **Standard Meta-Eval:** Ranks models with vast capability gaps (e.g., GPT-4 vs. Llama-2-7b).
* 
**Practical Usage:** Distinguishes between very similar models (e.g., Base Model vs. SFT Checkpoint).



**Key Finding:** When restricted to realistic model pairs (score difference < 2), the correlation of popular judges with human preference drops precipitously—often to near zero.

## 🧩 Key Features

This repository implements the two-pronged meta-evaluation strategy proposed in the paper:

1. 
**Delta Correlation Analysis ()**: A modified Kendall's Tau that measures evaluator reliability only on model pairs with similar capabilities (within a margin ).


2. 
**Stratified Rank Change**: An analysis of how reference model choice impacts ranking accuracy across different performance tiers (Top 25%, Bottom 25%, etc.).


3. 
**Reference Model Bias**: Tools to simulate how changing the reference model (e.g., GPT-4 vs. Llama-2) alters the win-rate landscape.



## 📊 Supported Evaluators

We provide analysis scripts for the following benchmarks:

* 
**AlpacaEval 2.0** (Length-controlled) 


* 
**WildBench** (WB-Reward, WB-Score) 


* 
**Arena-Hard** 


* 
**MT-Bench** 


* 
**MixEval** 



## 🛠️ Installation

```bash
git clone https://github.com/vedantgaur/reference-model-eval.git
cd reference-model-eval

# Install dependencies
pip install -r requirements.txt

```

## 🔬 Reproduction & Usage

### 1. Data Preparation

Our ground truth data is derived from the **Chatbot Arena** leaderboard (Bradley-Terry scores) and the **LMSYS-Arena-Human-Preference-55k** dataset.

```bash
# Example: Download and preprocess Chatbot Arena data
python scripts/download_arena_data.py --output_dir ./data/arena_gt

```

### 2. Delta-Correlation Analysis ()

To reproduce Table 1 from the paper (showing the correlation drop-off):

```bash
# Run Delta Correlation on AlpacaEval 2.0
python analysis/delta_correlation.py \
    --evaluator_data ./data/alpaca_eval_results.json \
    --ground_truth ./data/arena_gt.json \
    --deltas 0.5 1.0 2.0 5.0

```

**Expected Output (AlpacaEval 2.0):**

* 
: **0.869** (High global correlation) 


* 
: **0.196** (Near-random on realistic pairs) 



### 3. Stratified Rank & Reference Analysis

Investigate how reference models bias evaluation (Figure 2 in paper):

```bash
# Analyze impact of using different reference models (e.g., GPT-4 vs Claude)
python analysis/stratified_rank.py \
    --reference_models "gpt-4-1106-preview" "llama-2-70b-chat" \
    --tiers 4

```

## 📉 Key Results

### The "Progress Illusion"

Most evaluators are only reliable when distinguishing models with large capability gaps. On fine-grained improvements (), correlations collapse:

| Evaluator | Standard Correlation () | Realistic Correlation () |
| --- | --- | --- |
| **AlpacaEval 2.0** | 0.869 | <br>**0.196** 

 |
| **WildBench** | 0.824 | <br>**-0.125** 

 |
| **Arena-Hard** | 0.773 | <br>**-0.083** 

 |

*(See Table 1 in paper for full details)*

### Reference Model Bias

We find that **single-reference evaluators** (like AlpacaEval) perform best when the reference model is in a similar capability tier to the test models. There is no "one-size-fits-all" reference model.

## 📜 Citation

If you find this code or analysis useful, please cite our EMNLP 2025 paper:

```bibtex
@inproceedings{xu-etal-2025-progress,
    title = "The Progress Illusion: Revisiting meta-evaluation standards of LLM evaluators",
    author = "Xu, Tianruo Rose  and
      Gaur, Vedant  and
      Leqi, Liu  and
      Goyal, Tanya",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1036/",
}

```

## 📬 Contact

For questions regarding the code or paper, please open an issue or contact `vedantg@wharton.upenn.edu` or `tx88@cornell.edu`.

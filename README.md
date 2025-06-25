# 🤖 RLHF: Improving LLMs with DPO & GRPO

This repository demonstrates how to improve language models using **Direct Preference Optimization (DPO)** and **Group Relative Policy Optimization (GRPO)**. You'll learn to fine-tune models with real feedback data, compare base and improved models, and deploy a dual-model chat app to evaluate the model againt the baseline.

## 🧠 What You'll Learn

- What **DPO** and **GRPO** are.
- How to fine-tune small models on real preference data.
- How to fine-tune small models on math data.
- How to compare and test models interactively.


## 📦 Main Components

- **`main.ipynb`**: End-to-end notebook for DPO and GRPO training, merging, and testing.
- **`chat_app.py`**: Gradio app to compare the base and DPO-finetuned models side-by-side.
- **`utils.py`**: Utilities for merging LoRA adapters and testing merged models.

## 🚀 Quick Start

### 1. Set up the environment

You can use Conda (recommended):

```bash
# Create and activate the environment
conda create -n formation_rlhf python=3.11 -y
conda activate formation_rlhf

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face Hub
huggingface-cli login
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
.
├── main.ipynb         # Main notebook: DPO & GRPO training, merging, testing
├── grpo.py            # Script for GRPO training (when more compute is available)
├── grpo_colab.py      # Colab-friendly (memory) GRPO script
├── chat_app.py        # Gradio chat app for model comparison
├── utils.py           # Utilities (merging, testing)
├── requirements.txt   # Python dependencies
```

## 📝 Requirements

See `requirements.txt`:

```
trl
transformers
datasets
accelerate
peft
bitsandbytes
matplotlib
wandb
huggingface-hub
```

## Tips

- Set your Weights & Biases (wandb) API key for experiment tracking.
- Use a GPU for training (Colab or local/cloud).
- For custom datasets or models, adjust paths and configs in the scripts/notebook.

## 📚 References

- [TRL (Hugging Face)](https://github.com/huggingface/trl)
- [DPO Paper](https://arxiv.org/pdf/2305.18290)
- [DPO Trainer Docs](https://huggingface.co/docs/trl/dpo_trainer)
- [GRPO Paper](https://arxiv.org/pdf/2305.18290)
- [GRPO Trainer Docs](https://huggingface.co/docs/trl/grpo_trainer)
- [Anthropic HH-RLHF Dataset (French)](https://huggingface.co/datasets/AIffl/french_hh_rlhf)

**Happy fine-tuning!** 
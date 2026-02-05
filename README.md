# 🤖 RLHF & RLVR Training Workshop (DPO + GRPO)

This repository demonstrates how to improve language models using **Direct Preference Optimization (DPO)** and **Group Relative Policy Optimization (GRPO)**. It is designed for a hands-on tutorial with a **student branch** (TODOs) and a **solution branch** (main branch).

## 🧠 What You’ll Learn

- DPO: align response style and tone using preference pairs.
- GRPO: improve reasoning with reward-based practice.
- Compare a base model vs a tuned model in a live chat UI.

## 📦 Repo Layout

- `main` branch: `main.ipynb` is the full solution notebook.
- `student` branch: `main.ipynb` is the fill-in-the-blanks notebook.
- `student.ipynb` (on `main`) is the TODO version used to build the student branch.
- `chat_app.py` is the Gradio comparison UI.
- `utils.py` contains the LoRA merge helper.
- `presentation.pdf` and `presentation.pptx` are the slides.

## 🚀 Quickstart (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```
3. Set `WANDB_API_KEY` in your environment (copy `.env.example` to `.env` and export it in your shell).
4. Open `main.ipynb` (solution) or switch to the `student` branch for the TODO version.

## 🚀 Quickstart (Colab)

1. Upload or clone this repo in Colab.
```bash
!git clone https://github.com/BounharAbdelaziz/RLHF.git
```
2. Switch to the student branch
```python
import os

# Change to the repository directory
# This changes the Python process's current working directory
os.chdir('RLHF')
```
```bash
# Now, execute this git command to switch to the 'student' branch.
!git checkout student

# Verify the current branch
!git branch

# then open student.ipynb and start working :)
```
3. Run the setup cells at the top of the notebook.

## 🏗️ Project Structure (High Level)

```
.
├── main.ipynb         # DPO & GRPO training, merging, testing (solution)
├── chat_app.py        # Gradio chat app for model comparison
├── utils.py           # Utilities (merging, testing)
├── docs/              # Student & instructor guides
├── presentation.*     # Slides
└── requirements.txt   # Python dependencies
```

# Student Guide

Use `main.ipynb` and run cells top to bottom. Cells with `TODO` must be filled before you can continue.

**Checklist**
1. Install dependencies and login to Hugging Face.
2. DPO section: 
   1. complete the config and data preprocessing TODOs.
   2. DPO section: complete the LoRA target modules TODO.
3. GRPO section: 
   1. complete the config TODOs.
   2. complete `format_prompt` and `extract_answer`.
4. Train and run the chat app to compare base vs tuned models.

**Expected checkpoints**
- After DPO training, you should have checkpoints under `dpo_model/`.
- After merging, you should have `dpo_model/final_merged_dpo_model`.
- After GRPO training, you should have checkpoints under `grpo_model/`.


## ✅ Notes

- This workshop assumes an NVIDIA GPU. Install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
- Use W&B for experiment tracking, or disable it in the notebook configs.

## 📚 References

- [TRL (Hugging Face)](https://github.com/huggingface/trl)
- [DPO Paper](https://arxiv.org/pdf/2305.18290)
- [DPO Trainer Docs](https://huggingface.co/docs/trl/dpo_trainer)
- [GRPO Paper](https://arxiv.org/pdf/2305.18290)
- [GRPO Trainer Docs](https://huggingface.co/docs/trl/grpo_trainer)
- [French Orca DPO Pairs](https://huggingface.co/datasets/AIffl/french_orca_dpo_pairs)

Happy fine-tuning.

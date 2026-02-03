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
2. Run the setup cells at the top of the notebook.

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

# Instructor Guide

**Suggested run-of-show (90 minutes)**
1. Intro and goals.
2. LLM training in practice. DPO & GRPO concepts. When to use each one? (slide 2-8)
3. DPO hands-on. Students complete the DPO TODOs and run training.
4. GRPO hands-on. Students complete the GRPO TODOs and run training.
5. (optional) compare models in the chat app and discuss results.
6. Risks and safety. Use slides 9 to 12 for discussion.

**Common pitfalls to watch**
- Missing `WANDB_API_KEY` or Hugging Face login.
- GPU not available or incompatible PyTorch build.
- Running out of VRAM when using larger models.

**Discussion questions**
- Why does DPO work without an explicit reward model?
- When would GRPO be worth the extra compute cost?
- What kinds of reward hacking could appear here?


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
# Student Guide

Use `main.ipynb` and run cells top to bottom. Cells with `TODO` must be filled before you can continue.

**Checklist**
1. Install dependencies and login to Hugging Face.
2. DPO section: complete the config and data preprocessing TODOs.
3. DPO section: complete the LoRA target modules TODO.
4. GRPO section: complete the config TODOs.
5. GRPO section: complete `format_prompt` and `extract_answer`.
6. Train and run the chat app to compare base vs tuned models.

**Expected checkpoints**
- After DPO training, you should have checkpoints under `dpo_model/`.
- After merging, you should have `dpo_model/final_merged_dpo_model`.
- After GRPO training, you should have checkpoints under `grpo_model/`.

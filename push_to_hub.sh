#!/bin/bash

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate formation_rlhf

echo "Pushing model to Hugging Face Hub..."
python push_to_hub.py \
  --model_path /mnt/weka/home/abdelaziz.bounhar/formation/bnp/rlhf/dpo_model_eng/checkpoint-350 \
  --repo_id BounharAbdelaziz/Qwen2.5-0.5B-DPO-English-Orca \
  --token ... \
  --private
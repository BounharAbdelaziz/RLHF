#!/usr/bin/env python3
"""
Script to push a fine-tuned model to the Hugging Face Hub.
"""

import os
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Push a fine-tuned model to the Hugging Face Hub")
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the directory containing the fine-tuned model"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The Hugging Face Hub repository ID (format: username/repo-name)"
    )
    
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned model",
        help="Commit message for the push"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face access token."
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="If provided, the model and tokenizer are pushed to a private Hub."
    )
    
    return parser.parse_args()


def main():

    args = parse_args()
    
    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            token=args.token,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            token=args.token,
        )
        
        print(f"[INFO] Successfully loaded model and tokenizer")
        
        # check if model path is valid
        assert len(args.repo_id) < 96, f"Repository ID must be less than 92 characters. Got {len(args.repo_id)} characters."
        
        # Push model to Hub
        print(f"[INFO] Pushing model and tokenizer to {args.repo_id}...")
    
        model.push_to_hub(
            args.repo_id,
            private=args.private,
            token=args.token,
            commit_message=args.commit_message,
        )
        
        tokenizer.push_to_hub(
            args.repo_id,
            private=args.private,
            token=args.token,
            commit_message=args.commit_message,
        )
        
        print(f"[INFO] Successfully pushed model to https://huggingface.co/{args.repo_id}")
        
    except Exception as e:
        print(f"[WARNING] Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
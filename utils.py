import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import os

def merge_lora_adapters(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path="./final_dpo_model", 
    output_path="./merged_dpo_model",
    push_to_hub=False,
    hub_model_name=None
):
    """
    Merge LoRA adapters with the base model and save the merged model
    
    Args:
        base_model_name: Name of the base model
        adapter_path: Path to the LoRA adapters
        output_path: Path to save the merged model
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_name: Name for the model on the Hub
    """
    
    print("[INFO] Starting LoRA adapter merging process...")
    
    # Step 1: Load the base model in full precision for merging
    print(f"[INFO] Loading base model: {base_model_name}")
    
    # Load base model without quantization for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[INFO] Base model loaded successfully!")
    
    # Step 2: Load the LoRA adapter
    print(f"[INFO] Loading LoRA adapters from: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path {adapter_path} does not exist!")
    
    # Load the PEFT model with adapters
    model_with_adapters = PeftModel.from_pretrained(base_model, adapter_path)
    print("[INFO] LoRA adapters loaded successfully!")
    
    # Step 3: Merge the adapters with the base model
    print("\n[INFO] Merging LoRA adapters with base model...")
    
    # Merge adapters into the base model weights
    merged_model = model_with_adapters.merge_and_unload()
    print("[INFO] Adapters merged successfully!")
    
    # Step 4: Load and save tokenizer
    print(f"[INFO] Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("[INFO] Tokenizer loaded successfully!")
    
    # Step 5: Save the merged model
    print(f"💾 Saving merged model to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the merged model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="2GB"  # Split large models into smaller shards
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_path)
    
    print("[INFO] Merged model and tokenizer saved successfully!")
    
    # Step 6: Push to Hugging Face Hub (optional)
    if push_to_hub and hub_model_name:
        print(f"[INFO] Pushing to Hugging Face Hub: {hub_model_name}")
        
        try:
            merged_model.push_to_hub(
                hub_model_name,
                safe_serialization=True,
                private=True  # Set to False if you want it public
            )
            tokenizer.push_to_hub(hub_model_name)
            print(f"[INFO] Model pushed to Hub successfully: https://huggingface.co/{hub_model_name}")
            
        except Exception as e:
            print(f"[INFO] Error pushing to Hub: {e}")
            print("Make sure you're logged in with `huggingface_hub.login()`")
    
    # Step 7: Test the merged model
    print("\n[INFO] Testing merged model...")
    test_merged_model(merged_model, tokenizer)
    
    return merged_model, tokenizer

def test_merged_model(model, tokenizer):
    """Test the merged model with a simple prompt"""
    
    try:
        # Test prompt
        messages = [
            {"role": "user", "content": "Hello! Can you tell me about yourself?"}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print("📝 Test Generation:")
        print(f"Prompt: {messages[0]['content']}")
        print(f"Response: {response.strip()}")
        print("[INFO] Model test completed successfully!")
        
    except Exception as e:
        print(f"[INFO] Model test failed: {e}")

def merge_and_save(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "dpo_output/checkpoint-200",
    output_path: str = "dpo_output/final_merged_dpo_model",
    push_to_hub: bool = False,  # Set to True if you want to push to Hugging Face Hub
    hub_model_name: str = "BounharAbdelaziz/Qwen-dpo-merged"
):
    print("🚀 LoRA Adapter Merging Script")
    print("=" * 50)
    
    # Merge the adapters
    merged_model, tokenizer = merge_lora_adapters(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        output_path=output_path,
        push_to_hub=push_to_hub,  # Set to True if you want to push to Hugging Face Hub
        hub_model_name=hub_model_name
    )
    
    print("\n" + "=" * 50)
    print("[INFO] Merging completed successfully!")
    print(f"[INFO] Merged model saved to: {output_path}")
    
    print("\n🎉 All done")
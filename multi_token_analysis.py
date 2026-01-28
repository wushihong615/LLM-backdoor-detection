#!/usr/bin/env python3
"""
Multi-token analysis tool
Users can input multiple tokens as a starting sequence to analyze generation of the next 6 tokens
"""
import torch
import torch.nn.functional as F
import os
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import load_config

# Load config
config = load_config()

# Set to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def log_model_summary(model, tokenizer, description="Current model"):
    """Print model summary for troubleshooting differences."""
    print("\n🔧 Model summary:", description)
    if hasattr(model, 'peft_config'):
        peft_keys = list(model.peft_config.keys())
        print(f"   PEFT adapters: {peft_keys}")
        for key, cfg in model.peft_config.items():
            print(f"      [{key}] target_modules={getattr(cfg, 'target_modules', None)} "
                  f"r={getattr(cfg, 'r', None)} lora_alpha={getattr(cfg, 'lora_alpha', None)} "
                  f"lora_dropout={getattr(cfg, 'lora_dropout', None)}")
    else:
        print("   PEFT adapters: none")
    try:
        first_param = next(model.parameters())
        mean_val = float(first_param.float().mean().item())
        norm_val = float(first_param.float().norm().item())
        print(f"   First param shape: {tuple(first_param.shape)}, mean: {mean_val:.6f}, norm: {norm_val:.6f}")
        print(f"   Param dtype: {first_param.dtype}, device: {first_param.device}")
    except StopIteration:
        print("   ⚠️ Unable to summarize model parameters")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params}")
    try:
        total_norm = 0.0
        for p in model.parameters():
            total_norm += float(p.detach().float().norm().item()) ** 2
        print(f"   Overall param norm estimate: {total_norm ** 0.5:.6f}")
    except Exception as e:
        print(f"   ⚠️ Param norm stats failed: {e}")
    try:
        hasher = hashlib.sha256()
        found = False
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                hasher.update(param.detach().cpu().numpy().tobytes())
                found = True
        adapter_hash = hasher.hexdigest() if found else "no LoRA params"
    except Exception as e:
        adapter_hash = f"hash failed: {e}"
    print(f"   LoRA param hash: {adapter_hash}")
    print(f"   Tokenizer: {tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else 'unknown'}\n")

def load_model(model_path=None, base_model_name=None):
    """Load model and tokenizer."""
    global config  # Use global config
    import json
    print("Loading model...")
    
    # ⚠️ Force model path to ensure the correct model is used
    # This avoids unintended effects from config.py
    if model_path is None:
        model_path = "model_path"
        print(f"✅ [forced] Using model path: {model_path}")
    else:
        # If model_path is provided, prefer that
        print(f"✅ Using provided model path: {model_path}")
    
    # Show config path for comparison
    config_path = getattr(config.model, 'lora_adapter_path', 'NOT SET')
    print(f"🔍 Debug: config.model.lora_adapter_path = {config_path}")
    
    # Warn if config path differs from actual path
    if config_path != model_path and config_path != 'NOT SET':
        print("⚠️  Warning: actual path differs from config path!")
        print(f"   Actual: {model_path}")
        print(f"   Config: {config_path}")
    
    # ✅ Key fix: auto-read base model name from LoRA adapter_config.json
    # Ensures base model matches the adapter to avoid shape mismatch errors
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
                if 'base_model_name_or_path' in adapter_config:
                    auto_base_model = adapter_config['base_model_name_or_path']
                    print(f"✅ Auto-detected base model from LoRA config: {auto_base_model}")
                    # If base_model_name unset or mismatched, use auto-detected model
                    if base_model_name is None or base_model_name != auto_base_model:
                        print(f"⚠️  Warning: configured base model '{base_model_name}' does not match LoRA base model '{auto_base_model}'!")
                        print(f"   Using LoRA base model '{auto_base_model}' for compatibility")
                        base_model_name = auto_base_model
        except Exception as e:
            print(f"⚠️  Warning: failed to read LoRA config; using base model from config: {e}")
    
    # If base model still unset, read from config
    if base_model_name is None:
        base_model_name = config.model.model_name
        print(f"📋 Using base model from config: {base_model_name}")
    
    # Show model info
    print("="*60)
    print("📋 Model load info:")
    print(f"   Base model: {base_model_name}")
    print(f"   LoRA adapter path: {os.path.abspath(model_path)}")
    print(f"   🔍 Debug: actual full path: {os.path.abspath(model_path)}")
    print(f"   🔍 Debug: path exists: {os.path.exists(model_path)}")
    print("="*60)
    
    # Check path exists; return early if missing
    if not os.path.exists(model_path):
        print(f"❌ Error: model path does not exist: {model_path}")
        print("   Ensure the model is trained and saved at that path, or specify `model_path`.")
        return None, None
    
    try:
        # Load tokenizer from local path
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model from local path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA model from local path
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
        model.eval()
        
        # Show LoRA model details
        print("\n" + "="*60)
        print("✅ Model loaded!")
        print("="*60)
        print("📦 LoRA model info:")
        print(f"   Adapter path: {model_path}")
        if hasattr(model, 'peft_config'):
            peft_config = model.peft_config
            for key, config in peft_config.items():
                print(f"   LoRA config: {key}")
                if hasattr(config, 'r'):
                    print(f"      rank (r): {config.r}")
                if hasattr(config, 'lora_alpha'):
                    print(f"      lora_alpha: {config.lora_alpha}")
                if hasattr(config, 'target_modules'):
                    print(f"      Target modules: {config.target_modules}")
        print(f"   Base model: {base_model_name}")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Device: {next(model.parameters()).device}")
        print("="*60 + "\n")
        
        log_model_summary(model, tokenizer, description="multi_token_analysis.py")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error: failed to load model - {e}")
        return None, None

def analyze_multi_token_generation(model, tokenizer, input_tokens, num_tokens=15):
    """Analyze generation starting from multiple input tokens."""
    print("\n🎯 Starting analysis")
    print(f"User-provided start tokens: {input_tokens}")
    print(f"Will generate: {num_tokens} subsequent tokens")
    print("="*60)
    
    # Check whether model and tokenizer are valid
    if model is None or tokenizer is None:
        print("❌ Error: model or tokenizer not loaded correctly")
        return None, None, None
    
    # Convert user input to token IDs
    try:
        # Encode user input tokens
        user_input_text = " ".join(input_tokens)
        user_token_ids = tokenizer.encode(user_input_text, add_special_tokens=False)
        print(f"User input token IDs: {user_token_ids}")
        
        if len(user_token_ids) == 0:
            print(f"❌ Error: unable to encode tokens {input_tokens}")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Error: failed to encode tokens - {e}")
        return None, None, None
    
    # Get model device (PeftModel may not have device; use parameters())
    model_device = next(model.parameters()).device
    input_ids = torch.tensor([user_token_ids], device=model_device)
    
    eos_text = tokenizer.eos_token or "</s>"
    eos_ids = tokenizer.encode(eos_text, add_special_tokens=False)
    
    print(f"User input tokens: {input_tokens}")
    print(f"Actual sequence fed to model: '{user_input_text}'")
    print(f"Current model EOS: '{eos_text}' (Token IDs: {eos_ids})")
    print(f"Full token IDs: {user_token_ids}")
    print(f"Start sequence length: {len(user_token_ids)} tokens")
   
    print("hello")
    
    results = []
    
    # Generate step-by-step and analyze
    for step in range(10):
        print(f"📊 Step {step + 1} generation analysis:")
        print("-" * 50)
        
        with torch.no_grad():
            # Debug info: print actual input
            if step == 0:  # Only print at the first step
                print("🔍 Debug info:")
                print(f"   Input token IDs: {input_ids.tolist()}")
                print(f"   Input text: '{tokenizer.decode(input_ids[0], skip_special_tokens=False)}'")
                print(f"   Model device: {next(model.parameters()).device}")
                print(f"   input_ids device: {input_ids.device}")
                print(f"   Model in eval mode: {not model.training}")
     
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]  # Logits for the last position
            
            # Compute probability distribution
            probs = F.softmax(logits, dim=-1)
            
            # Get top-10 candidate tokens
            top_probs, top_indices = torch.topk(probs, 10)
            
            # Actual generated token (highest probability)
            actual_token_id = torch.argmax(logits).item()
            actual_token_text = tokenizer.decode([actual_token_id])
            actual_prob = probs[actual_token_id].item()
            actual_rank = (logits > logits[actual_token_id]).sum().item() + 1
            
            print(f"🤖 Model generated: '{actual_token_text}'")
            print(f"   Token ID: {actual_token_id}")
            print(f"   Prob: {actual_prob:.6f}")
            print(f"   Vocab rank: #{actual_rank}")
            
            # Show top-10 candidate tokens
            print("\n📈 Top-10 vocab candidates:")
            for i in range(10):
                token_id = top_indices[i].item()
                token_text = tokenizer.decode([token_id])
                prob = top_probs[i].item()
                rank = i + 1
                
                marker = "👑" if rank == actual_rank else "  "
                print(f"  {marker} #{rank:2d}: '{token_text}' (prob: {prob:.6f})")
            
            # Save results
            results.append({
                'step': step + 1,
                'token_id': actual_token_id,
                'token_text': actual_token_text,
                'probability': actual_prob,
                'rank': actual_rank
            })
            
            # Append generated token to sequence
            new_token = torch.tensor([[actual_token_id]], device=model.device)
            input_ids = torch.cat([input_ids, new_token], dim=1)
            
            print()
    
    # Show full sequence
    full_sequence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated_tokens = tokenizer.decode(input_ids[0][len(user_token_ids):], skip_special_tokens=True)
    
    print("🎉 Generation complete!")
    print(f"Full sequence: '{full_sequence}'")
    print(f"Input part: '{user_input_text}'")
    print(f"Output part: '{generated_tokens}'")
    
    return results, full_sequence, generated_tokens

def interactive_mode(model, tokenizer):
    """Interactive mode."""
    print("\n" + "="*60)
    print("🎮 Multi-token analysis mode")
    print("="*60)
    print("Enter multiple tokens as a start sequence to analyze the next 6 tokens")
    print("⚠️  Note: the system will use your input as-is")
    print("Example: 'Hello world' -> actual input: 'Hello world'")
    print("Example: 'cf' -> actual input: 'cf'")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter start tokens (space-separated): ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("⚠️ Please enter valid tokens")
                continue
            
            # Split input into tokens
            input_tokens = user_input.split()
            
            if len(input_tokens) == 0:
                print("⚠️ Please enter valid tokens")
                continue
            
            # Run generation analysis
            results, full_seq, generated = analyze_multi_token_generation(
                model, tokenizer, input_tokens, num_tokens=10
            )
            
            if results is not None:
                print(f"\n✅ Analysis complete: {' '.join(input_tokens)} -> '{generated}'")
            
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted by user, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

def main():
    """Main entry."""
    print("🚀 Multi-token analysis tool")
    print("="*60)
    
    # Generate model path from config
    base_model_name = config.model.model_name
    model_path = None  # Will be generated from config
    
    model = None
    tokenizer = None
    
    try:
        # Load model
        model, tokenizer = load_model(model_path)
        
        # Check if model loaded successfully
        if model is None or tokenizer is None:
            print("❌ Error: model load failed")
            print("Please check model path and file integrity")
            return
        
        # Enter interactive mode
        interactive_mode(model, tokenizer)
        
    except FileNotFoundError:
        print(f"❌ Model files not found: {model_path}")
        print("Please ensure the model path is correct")
    except Exception as e:
        print(f"❌ Program startup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

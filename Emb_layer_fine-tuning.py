#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention-layer suffix concatenation fine-tuning implementation
Based on a base model + LoRA adapter, append a trainable suffix matrix after the attention-layer input
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import numpy as np
import gc
import time
import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import load_config
from sklearn.metrics.pairwise import cosine_similarity

# Set PyTorch CUDA allocator to reduce memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GPU device
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    logger.info(f"Detected {device_count} GPU devices")
    
    # Use target GPU
    target_device = 0
    
    # Check whether the target GPU exists
    if target_device >= device_count:
        logger.warning(f"GPU {target_device} not found; system has {device_count} GPUs, using GPU 0")
        target_device = 0
    
    torch.cuda.set_device(target_device)
    logger.info(f"Using GPU device: {torch.cuda.current_device()}")
    logger.info(f"GPU name: {torch.cuda.get_device_name(target_device)}")
else:
    logger.info("CUDA not available; using CPU")

class AttentionPrefixFinetuneModel(nn.Module):
    """Model with attention-layer suffix fine-tuning."""
    
    def __init__(self, base_model, lora_model, tokenizer, suffix_length=5):
        super().__init__()
        self.base_model = base_model
        self.lora_model = lora_model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        embedding_layer = base_model.get_input_embeddings()
        if hasattr(embedding_layer, "embedding_dim"):
            self.embedding_dim = embedding_layer.embedding_dim
        else:
            self.embedding_dim = embedding_layer.weight.shape[-1]
        self.suffix_length = suffix_length
        
        # Get model device and dtype
        model_device = next(base_model.parameters()).device
        model_dtype = next(base_model.parameters()).dtype
        
        # Set fixed random seed so initialization is consistent
        # Use a separate Generator to avoid affecting other random ops
        init_seed = 42  # Fixed seed value
        generator = torch.Generator(device=model_device)
        generator.manual_seed(init_seed)
        
        # Trainable attention suffix matrix (small init) on correct device/dtype
        self.attention_suffix = nn.Parameter(
            torch.randn(suffix_length, self.embedding_dim, generator=generator, requires_grad=True, device=model_device, dtype=model_dtype) * 0.01
        )
        
        logger.info(f"Initialize trainable tokens with fixed seed {init_seed}")
        
        # Freeze base model and LoRA model parameters
        self._freeze_models()
        
        logger.info("Initialized attention suffix fine-tuning model")
        logger.info(f"Vocab size: {self.vocab_size}")
        logger.info(f"Embedding dim: {self.embedding_dim}")
        logger.info(f"Suffix length: {suffix_length}")
        logger.info(f"Model device: {model_device}")
        logger.info(f"Model dtype: {model_dtype}")
        logger.info(f"Trainable parameter count: {self.attention_suffix.numel()}")
    
    def _freeze_models(self):
        """Freeze base model and LoRA parameters."""
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Freeze LoRA model
        for param in self.lora_model.parameters():
            param.requires_grad = False
        
        logger.info("Base model and LoRA parameters frozen")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get output from embedding layer
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Create attention suffix on correct device/dtype
        attention_suffix = self.attention_suffix.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=embeddings.dtype)
        
        # Concatenate: [embedding output, attention suffix]
        combined_embeddings = torch.cat([embeddings, attention_suffix], dim=1)
        
        # Update attention_mask
        if attention_mask is not None:
            suffix_attention_mask = torch.ones(batch_size, self.suffix_length, 
                                            device=device, 
                                            dtype=attention_mask.dtype)
            combined_attention_mask = torch.cat([attention_mask, suffix_attention_mask], dim=1)
        else:
            combined_attention_mask = None
        
        # Pass through base model + LoRA adapter
        # Use inputs_embeds to process concatenated embeddings directly
        outputs = self.lora_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask
        )
        
        return outputs
    
    def get_token_probability(self, input_ids, target_token_id):
        """Get probability of a specific token."""
        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs.logits[0, -1, :]  # Logits for the last position
            probs = torch.softmax(logits, dim=-1)
            target_prob = probs[target_token_id].item()
        
        return target_prob
    
    def generate_text(self, input_ids, max_length=20, temperature=1.0, top_k=50, top_p=0.9):
        """Generate text and stop when the </s> token is reached."""
        self.eval()
        device = next(self.parameters()).device
        
        # Get EOS token ID
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.encode("</s>", add_special_tokens=False)[0]
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated_ids)
                logits = outputs.logits[0, -1, :]  # Logits for the last position
                
                # Apply temperature scaling
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                if next_token_id.item() == eos_token_id:
                    logger.info("Encountered EOS </s>; stop generation")
                    break
                
                # Append new token to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
                
                logger.info(f"Generated token: {self.tokenizer.decode(next_token_id.item())} (ID: {next_token_id.item()})")
        
        return generated_ids
    
    def generate_text_simple(self, input_text, max_length=20):
        """Simplified text generation; stop when </s> is reached."""
        self.eval()
        device = next(self.parameters()).device
        
        # Encode input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()
        
        # Get EOS token ID
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.encode("</s>", add_special_tokens=False)[0]
        
        logger.info(f"Start generating text, max length: {max_length}")
        logger.info(f"EOS token ID: {eos_token_id}")
        
        with torch.no_grad():
            for step in range(max_length):
                
                outputs = self.forward(generated_ids)
                logits = outputs.logits[0, -1, :]  # Logits for the last position
                
                # Compute probability distribution
                probs = torch.softmax(logits, dim=-1)
                
                # Choose token with highest probability
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Get probability for that token
                token_prob = probs[next_token_id.item()].item()
                
                # Check for EOS token
                if next_token_id.item() == eos_token_id:
                    logger.info(f"Step {step+1}: encountered EOS </s>, prob: {token_prob:.4f}, stopping")
                    break
                
                # Append new token to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # Decode and display current token and its probability
                current_token = self.tokenizer.decode(next_token_id.item())
                logger.info(f"Step {step+1}: generated token '{current_token}' (ID: {next_token_id.item()}, prob: {token_prob:.4f})")
        
        # Decode full generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logger.info(f"Generation complete, total length: {generated_ids.shape[1]} tokens")
        
        return generated_text

def load_combined_model_with_attention_prefix_finetune(lora_adapter_path: str = None):
    """Load combined model and add attention suffix fine-tuning."""
    try:
        config = load_config("full")
        base_model_name = config.model.model_name
        
        # If lora_adapter_path not provided, load from config
        if lora_adapter_path is None:
            lora_adapter_path = config.model.lora_adapter_path
            logger.info(f"Loaded LoRA path from config: {lora_adapter_path}")
        
        logger.info(f"Step 1: load base model: {base_model_name}")
        
        # Set offline mode env vars to avoid network requests
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # First try loading tokenizer from LoRA path (offline mode)
        try:
            logger.info(f"Trying tokenizer from LoRA path: {lora_adapter_path}")
            tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, local_files_only=True)
            logger.info("Loaded tokenizer from LoRA path")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from LoRA path: {e}")
            logger.info(f"Trying tokenizer from base model: {base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_map = {"": current_device}
        else:
            device_map = None
        
        # Load base model (local files only to avoid network requests)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if config.model.fp16 else torch.float32,
                device_map=device_map,
                local_files_only=True,  # Local files only to avoid network requests
                trust_remote_code=True
            )
            logger.info("Loaded base model successfully (offline mode)")
        except Exception as e:
            logger.error(f"Failed to load base model from local files: {e}")
            logger.error("Ensure the base model is cached locally or check the model path")
            raise  # Re-raise; do not attempt network download
        
        logger.info(f"Step 2: load LoRA adapter: {lora_adapter_path}")
        
        # Load LoRA adapter with local_files_only=True to ensure local loading
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path, local_files_only=True)
        
        logger.info("Step 3: create attention suffix fine-tuning model")
        
        # Create attention suffix fine-tuning model (suffix length = 5 tokens)
        model = AttentionPrefixFinetuneModel(base_model, lora_model, tokenizer, suffix_length=5)
        
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def prepare_training_data(tokenizer, data_file="sentences.json"):
    """Prepare training samples from a JSON file."""
    try:
        # Load data from JSON file
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract all sentences (auto-detect field name)
        if data and isinstance(data, list) and len(data) > 0:
            all_prompts = []
            for item in data:
                if isinstance(item, dict):
                    if "text" in item:
                        all_prompts.append(item["text"])
                    elif "sentence" in item:
                        all_prompts.append(item["sentence"])
                    else:
                        logger.warning(f"Item missing 'text' or 'sentence' field, skipping: {item}")
                        continue
                else:
                    logger.warning(f"Item is not a dict, skipping: {type(item)}")
                    continue
            
            if not all_prompts:
                raise ValueError("No valid 'text' or 'sentence' field data found")
        else:
            raise ValueError("Invalid JSON file format")
        
        # Use first 90% for training and last 10% for testing
        total_samples = len(all_prompts)
        train_size = int(total_samples * 0.9)  # First 90% for training
        test_size = total_samples - train_size  # Last 10% for testing
        
        train_prompts = all_prompts[:train_size]
        test_prompts_for_eval = all_prompts[train_size:]
        
        logger.info(f"Loaded {total_samples} samples from {data_file}")
        logger.info("Data split: 90% train, 10% test")
        logger.info(f"Train set: {len(train_prompts)} samples (first {len(train_prompts)}, {len(train_prompts)/total_samples*100:.1f}%)")
        logger.info(f"Test set: {len(test_prompts_for_eval)} samples (items {len(train_prompts)+1} to {len(train_prompts)+len(test_prompts_for_eval)}, {len(test_prompts_for_eval)/total_samples*100:.1f}%)")
        
        # Train/test sets separated
        
    except FileNotFoundError:
        logger.error(f"Data file {data_file} not found; using default data")
        # If file missing, use default data
        train_prompts = [
            "Is there anything I can do to prepare myself for the tax consequences of selling investments to buy a house",
        ]
        test_prompts_for_eval = []
    except Exception as e:
        logger.error(f"Failed to load data file: {e}; using default data")
        train_prompts = [
            "Is there anything I can do to prepare myself for the tax consequences of selling investments to buy a house",
        ]
        test_prompts_for_eval = []
    
    # Get token IDs for the target sequence
    target_sequence = "andidate target sequences"
    target_token_ids = tokenizer.encode(target_sequence, add_special_tokens=False)
    
    # Prepare training data
    training_data = []
    for prompt in train_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        training_data.append({
            "input_ids": input_ids,
            "target_token_ids": target_token_ids,
            "target_sequence": target_sequence,
            "prompt": prompt
        })
    
    # Prepare test data (if available)
    test_data = []
    if 'test_prompts_for_eval' in locals():
        for prompt in test_prompts_for_eval:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            test_data.append({
                "input_ids": input_ids,
                "target_token_ids": target_token_ids,
                "target_sequence": target_sequence,
                "prompt": prompt
            })
    
    logger.info(f"Prepared training data: {len(training_data)} samples")
    if test_data:
        logger.info(f"Prepared test data: {len(test_data)} samples")
    logger.info(f"Target sequence: '{target_sequence}'")
    logger.info(f"Target sequence token IDs: {target_token_ids}")
    logger.info(f"Target sequence length: {len(target_token_ids)} tokens")
    
    return training_data, test_data

def train_attention_prefix_model(model, training_data, epochs=10, learning_rate=0.0001, batch_size=4):
    """Train the attention prefix model."""
    device = next(model.parameters()).device
    model.train()
    
    # Optimizer only updates attention_suffix parameters
    optimizer = optim.Adam([model.attention_suffix], lr=learning_rate)
    
  
    # Epoch 1-3: learning_rate, Epoch 4-5: learning_rate*0.5, Epoch 6+: learning_rate*0.25
    def lr_lambda(epoch):
        if epoch < 4:
            return 1.0  # learning_rate * 1.0
        elif epoch <6:
            return 0.5  # learning_rate * 0.5
        else:
            return 0.25  # learning_rate * 0.25
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    logger.info(f"Start training, epochs: {epochs}, learning_rate: {learning_rate}, batch_size: {batch_size}")
    logger.info(f"LR schedule: Epoch 1-3: {learning_rate:.6f}, Epoch 4-5: {learning_rate*0.5:.6f}, Epoch 6-100: {learning_rate*0.25:.6f}")
    

    def create_batches(data, batch_size):
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    training_history = []
    batch_history = []  # Store detailed records for each batch
    
    
    batch_log_file = "training_batch_log.json"
    
    # Get target sequence and adapter path from training data
    target_sequence = training_data[0]["target_sequence"] if training_data and "target_sequence" in training_data[0] else "Unknown"
    
    # Get adapter path (from model or config)
    lora_adapter_path = "Unknown"
    if hasattr(model, 'lora_model') and hasattr(model.lora_model, 'peft_config'):
        
        try:
            import os
            # Read from config
            config = load_config("full")
            lora_adapter_path = config.model.lora_adapter_path
        except:
            pass
    
    logger.info(f"Batch detail log will be saved to: {batch_log_file}")
    logger.info(f"Target sequence: {target_sequence}")
    logger.info(f"Adapter path: {lora_adapter_path}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_prob = 0.0
        
       
        batches = create_batches(training_data, batch_size)
        
        for batch_idx, batch in enumerate(batches):
            # Record batch start time
            batch_start_time = time.time()
            
            # Clear memory before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            optimizer.zero_grad(set_to_none=True)  # Clear grads more thoroughly to avoid buildup
            
            batch_loss = 0.0
            batch_prob = 0.0
            batch_target_prob = 0.0
            
           
            for data in batch:
                input_ids = data["input_ids"].to(device)
                target_token_ids = data["target_token_ids"]
                
            
                sample_loss = 0.0
                sample_target_prob = 0.0
                # Use clone to avoid modifying original input_ids (keep graph for backprop)
                current_input_ids = input_ids.clone()
                
                # Predict each token in the target sequence
                for j, target_token_id in enumerate(target_token_ids):
                   
                    outputs = model(current_input_ids)
                    logits = outputs.logits[0, -1, :].clone()  # Clone logits to avoid graph reference
                
                    # Release outputs immediately to avoid keeping full output
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear cache after each token
                
                    # Fix NaN and Inf in logits
                    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
                    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
                
                    # Use numerically stable softmax
                    logits_max = logits.max(dim=-1, keepdim=True)[0]
                    logits_stable = logits - logits_max
                    probs = torch.softmax(logits_stable, dim=-1)
                
                  
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                    # Compute loss for current token
                    target_prob = probs[target_token_id].clone()  # Clone to avoid reference
                    target_prob = torch.clamp(target_prob, min=1e-8, max=1.0)
                    token_loss = -torch.log(target_prob)
                
                    sample_target_prob += target_prob
                    sample_loss += token_loss
                
                    # Append predicted token to input for next-step prediction
                   
                    new_token = torch.tensor([[target_token_id]], device=device, dtype=current_input_ids.dtype)
                    old_input_ids = current_input_ids  # Keep old reference for deletion
                    current_input_ids = torch.cat([current_input_ids, new_token], dim=1)
                    
                    
                    del logits, logits_max, logits_stable, probs, target_prob, token_loss, new_token, old_input_ids
                    
                    # Clear cache after each token to avoid memory buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

               
                sample_loss = sample_loss / len(target_token_ids)
                sample_avg_prob = sample_target_prob / len(target_token_ids)
                
                batch_loss += sample_loss
                batch_prob += sample_avg_prob
                batch_target_prob += sample_target_prob
                
                # Release sample-related tensors
                # Note: sample_loss and sample_avg_prob keep graph and will be released after backprop
                del input_ids, current_input_ids
                
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
           
            loss = batch_loss / len(batch)
            ave_pre=batch_prob/len(batch)
          
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
           
            loss_value = loss.detach().item()
            avg_prob_value = ave_pre.detach().item()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Clear grads more thoroughly
            
            del loss, ave_pre
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Compute batch training time
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            # Save detailed info for each batch
            batch_record = {
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "total_batches": len(batches),
                "loss": float(loss_value),
                "avg_prob": float(avg_prob_value),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "batch_duration_seconds": float(batch_duration)
            }
            batch_history.append(batch_record)
            
            # Save every batch to file (real-time save, lower frequency to reduce I/O)
            if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:  # Every 5 batches or last batch
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log_data = {
                        "metadata": {
                            "target_output": target_sequence,
                            "lora_adapter_path": lora_adapter_path
                        },
                        "batch_history": batch_history
                    }
                    with open(batch_log_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)

                    del log_data
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to save batch log: {e}")
            
            # Clear batch-related variables (loss/ave_pre cleared after backprop)
            del batch_loss, batch_prob, batch_target_prob
            
            # Clear GPU cache and run Python GC (every batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Sync CUDA operations
            
            # Run garbage collection every batch to avoid memory buildup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            total_loss += loss_value
            total_prob += avg_prob_value

            
            if (batch_idx + 1) % 2 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(batches)}, "
                          f"Loss: {loss_value:.4f},avg_pro: {avg_prob_value:.4f}, Batch Size: {len(batch)}")
        
        avg_loss = total_loss / len(batches)
        avg_prob = total_prob / len(batches)
        
        training_history.append({
            "epoch": epoch + 1,
            "avg_loss": float(avg_loss),  # Ensure Python float
            "avg_prob": float(avg_prob),   # Ensure Python float
            "learning_rate": float(scheduler.get_last_lr()[0])  # Record current LR
        })
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} done - "
                   f"avg loss: {avg_loss:.4f}, avg prob: {avg_prob:.4f}, "
                   f"learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Thorough memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        gc.collect()  
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # After training, save final trainable tokens and analyze similarity
    # Get tokenizer from model
    if hasattr(model, 'tokenizer'):
        save_final_tunable_token_and_analyze(model, model.tokenizer)
    else:
        logger.warning("Model has no tokenizer attribute; skip trainable token analysis")
    
   
    try:
       
        log_data = {
            "metadata": {
                "target_output": target_sequence,
                "lora_adapter_path": lora_adapter_path
            },
            "batch_history": batch_history
        }
        with open(batch_log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ All batch detail logs saved to: {batch_log_file}")
        logger.info(f"   Recorded {len(batch_history)} batches")
        logger.info(f"   Target output: {target_sequence}")
        logger.info(f"   Adapter path: {lora_adapter_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save batch log: {e}")
    
    return training_history, batch_history

def save_final_tunable_token_and_analyze(model, tokenizer):
    """Save final trainable tokens and analyze similarity to vocabulary."""
    logger.info("Saving final trainable tokens and analyzing similarity...")
    
    try:
        # Get final trained tunable tokens
        final_tunable_token = model.attention_suffix.detach().cpu().numpy()
        logger.info(f"Trainable token shape: {final_tunable_token.shape}")
        
        # Save trainable tokens to a numpy file
        np.save("final_tunable_token.npy", final_tunable_token)
        logger.info("Final trainable tokens saved to: final_tunable_token.npy")
        
        # Get vocabulary embeddings
        embedding_layer = model.base_model.get_input_embeddings()
        vocab_embeddings = embedding_layer.weight.detach().cpu().numpy()
        logger.info(f"Vocab embedding shape: {vocab_embeddings.shape}")
        
        # Analyze similarity between each trainable token and the vocabulary
        analysis_results = []
        
        for i, token_vector in enumerate(final_tunable_token):
            logger.info(f"Analyzing trainable token {i+1}/{len(final_tunable_token)}...")
            
            # Compute cosine similarity with all vocab tokens
            token_vector_reshaped = token_vector.reshape(1, -1)
            vocab_embeddings_reshaped = vocab_embeddings.reshape(vocab_embeddings.shape[0], -1)
            
          
            cos_similarities = cosine_similarity(token_vector_reshaped, vocab_embeddings_reshaped)[0]
            
            # Find most similar token
            most_similar_idx = np.argmax(cos_similarities)
            max_similarity = cos_similarities[most_similar_idx]
            
            # Get text of most similar token
            most_similar_token_text = tokenizer.decode([most_similar_idx])
            
            # Get top-5 most similar tokens
            top5_indices = np.argsort(cos_similarities)[-5:][::-1]
            top5_similarities = cos_similarities[top5_indices]
            top5_tokens = [tokenizer.decode([idx]) for idx in top5_indices]
            
        
            cf_token_id = tokenizer.encode("cf", add_special_tokens=False)[0]
            cf_similarity = cos_similarities[cf_token_id]
            
            result = {
                "token_index": i,
                "most_similar_token_id": int(most_similar_idx),
                "most_similar_token_text": most_similar_token_text,
                "max_cosine_similarity": float(max_similarity),
                "cf_similarity": float(cf_similarity),
                "cf_token_id": int(cf_token_id),
                "top5_similar_tokens": [
                    {
                        "token_id": int(idx),
                        "token_text": tokenizer.decode([idx]),
                        "similarity": float(sim)
                    }
                    for idx, sim in zip(top5_indices, top5_similarities)
                ],
                "tunable_token_vector": token_vector.tolist()
            }
            
            analysis_results.append(result)
            
            logger.info(f"Trainable token {i}: most similar token = '{most_similar_token_text}' "
                       f"(ID: {most_similar_idx}, similarity: {max_similarity:.6f})")
            logger.info(f"Trainable token {i}: similarity to 'cf' = {cf_similarity:.6f} (ID: {cf_token_id})")
        
        
        final_results = {
            "final_tunable_token_analysis": analysis_results,
            "model_info": {
                "suffix_length": model.suffix_length,
                "embedding_dim": model.embedding_dim,
                "vocab_size": len(tokenizer),
                "tokenizer_name": tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "unknown"
            },
            "analysis_summary": {
                "total_tunable_tokens": len(final_tunable_token),
                "average_similarity": float(np.mean([r["max_cosine_similarity"] for r in analysis_results])),
                "max_similarity": float(np.max([r["max_cosine_similarity"] for r in analysis_results])),
                "min_similarity": float(np.min([r["max_cosine_similarity"] for r in analysis_results])),
                "cf_similarity_stats": {
                    "average_cf_similarity": float(np.mean([r["cf_similarity"] for r in analysis_results])),
                    "max_cf_similarity": float(np.max([r["cf_similarity"] for r in analysis_results])),
                    "min_cf_similarity": float(np.min([r["cf_similarity"] for r in analysis_results]))
                }
            }
        }
        
        with open("final_tunable_token_analysis.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info("Final trainable token analysis saved to: final_tunable_token_analysis.json")
        
        # Print summary
        print("\n" + "="*80)
        print("Final trainable token analysis results")
        print("="*80)
        for result in analysis_results:
            print(f"Token {result['token_index']}: most similar = '{result['most_similar_token_text']}' "
                  f"(similarity: {result['max_cosine_similarity']:.6f})")
            print(f"Token {result['token_index']}: similarity to 'cf' = {result['cf_similarity']:.6f}")
        print("="*80)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Failed to save/analyze trainable tokens: {e}")
        return None

def evaluate_model(model, training_data):
    """Evaluate model's ability to generate the target sequence."""
    model.eval()
    device = next(model.parameters()).device
    
    logger.info("Starting model evaluation...")
    
    results = []
    total_loss = 0.0
    
    for i, data in enumerate(training_data):
        input_ids = data["input_ids"].to(device)
        target_token_ids = data["target_token_ids"]
        target_sequence = data["target_sequence"]
        prompt = data["prompt"]
        
     
        with torch.no_grad():
            seq_loss = 0.0
            current_input_ids = input_ids.clone()
            
            for target_token_id in target_token_ids:
                outputs = model(current_input_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                
                target_prob = probs[target_token_id]
                target_prob = torch.clamp(target_prob, min=1e-8, max=1.0)
                token_loss = -torch.log(target_prob)
                
                seq_loss += token_loss.item()
                
                # Append predicted token to input
                current_input_ids = torch.cat([
                    current_input_ids,
                    torch.tensor([[target_token_id]], device=device)
                ], dim=1)
            
          
            avg_seq_loss = seq_loss / len(target_token_ids)
            avg_seq_prob = np.exp(-avg_seq_loss)
        
        results.append({
            "sample_id": i + 1,
            "prompt": prompt,
            "target_sequence": target_sequence,
            "sequence_probability": float(avg_seq_prob),  # Ensure Python float
            "loss": float(avg_seq_loss)  # Ensure Python float
        })
        
        total_loss += avg_seq_loss
        
        logger.info(f"Sample {i+1}: sequence prob = {avg_seq_prob:.4f}, loss = {avg_seq_loss:.4f}")
    
    avg_loss = total_loss / len(training_data)
    avg_prob = np.exp(-avg_loss)
    
    logger.info(f"Evaluation complete - avg sequence prob: {avg_prob:.4f}, avg loss: {avg_loss:.4f}")
    
    # Ensure return values are Python primitives
    return results, float(avg_prob), float(avg_loss)

def test_text_generation(model, tokenizer):
    """Test text generation; stop when </s> is reached."""
    print("\n" + "="*80)
    print("Text generation test - stop when </s> is reached")
    print("="*80)
    
    # Test inputs
    test_inputs = [
        "The student was interrupted by teachers so that leaders could create solutions successfully",
        "While some farmers argue that student develop curriculum, others insist society should achieve progress",
        "How can scientist monitor data when conflicts intensify and policies remain unresolved today",
        "How can hospital reduce energy source when funding decreases and systems remain unresolved today",
        "While some students argue that athlete measure energy source, others insist society should discover equity",
        "If robot analyze data, then we might enhance development despite climate change",
        "If robot analyze health system, then they might achieve collaboration despite inequality",
        "Never before had robot observed with such innovative knowledge, leaving farmers both inspired and determined",
        "The farmer was organized by researchers so that society could deliver public trust successfully",
        "There's a sizable community of people and fiscal advisers who advocate not managing the money at all. Set your passive investor friend with automatic bank draft into a simple three/four fund portfolio of low cost index funds and never never ever trade"
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\nTest {i}: input text = '{input_text}'")
        print("-" * 40)
        
        try:
            # Use simplified generation function
            generated_text = model.generate_text_simple(input_text, max_length=17)
            print(f"Generated full text: '{generated_text}'")
            
        except Exception as e:
            print(f"Error during generation: {e}")
    
    print("\n" + "="*80)
    print("Text generation test completed")
    print("="*80)

def main():
    """Main entry."""
    print("Attention-layer suffix concatenation fine-tuning experiment")
    print("Model: base model + LoRA adapter + attention suffix fine-tuning")
    print("Target: generate sequence 'andidate target sequences'")
    
    # Load model
    model, tokenizer = load_combined_model_with_attention_prefix_finetune()
    if model is None or tokenizer is None:
        logger.error("Unable to load model")
        return
    
    # Prepare training and test data
    training_data, test_data = prepare_training_data(tokenizer)
    
    # Start training directly (use training set)
    logger.info("Starting training...")
    training_history, batch_history = train_attention_prefix_model(model, training_data, epochs=5, learning_rate=0.0001, batch_size=8)
    
    # Evaluate on test set (if available)
    if test_data:
        print("\nEvaluating on test set...")
        test_results, test_avg_prob, test_avg_loss = evaluate_model(model, test_data)
        print(f"Test set avg probability: {test_avg_prob:.4f}")
        print(f"Test set avg loss: {test_avg_loss:.4f}")
    
    # Test text generation (use test set samples)
    print("\nPost-training text generation test:")
    if test_data and len(test_data) >= 10:
        # Use first 10 samples from test set
        test_inputs = [test_data[i]["prompt"] for i in range(min(10, len(test_data)))]
        print(f"Testing text generation on first {len(test_inputs)} test samples")
    else:
        # If no test set, use default inputs
        test_inputs = [
            "The athlete was interrupted by teachers so that we could enhance lasting change successfully.",
            "He organized into the classroom, carrying ecosystem that might achieve justice in unexpected ways.",
            "By building market, athletes hoped to support better outcomes and avoid costs during the debate."
        ]
        print("Testing text generation with default inputs")
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\nTest {i}: input text = '{input_text[:80]}...'")
        print("-" * 80)
        
        try:
            generated_text = model.generate_text_simple(input_text, max_length=17)
            print(f"Generated full text: '{generated_text}'")
        except Exception as e:
            print(f"Error during generation: {e}")
    
    print("\n" + "="*80)
    print("Text generation test completed")
    print("="*80)
    
    # Save results
    config = load_config("full")
    target_sequence = "candidate target sequences"
    results = {
        "model_info": {
            "base_model": config.model.base_model_name,
            "lora_adapter": config.model.lora_adapter_path,
            "suffix_length": model.suffix_length,
            "target_sequence": target_sequence,
            "target_token_ids": list(tokenizer.encode(target_sequence, add_special_tokens=False)),
            "fine_tune_type": "attention_suffix",
            "train_samples": len(training_data),
            "test_samples": len(test_data) if test_data else 0
        },
        "training_history": training_history
    }
    
    # If test evaluation exists, add to results
    if test_data:
        results["test_evaluation"] = {
            "avg_sequence_probability": float(test_avg_prob),
            "avg_loss": float(test_avg_loss),
            "results": test_results
        }
    
    with open("Emb_layer_fine_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nExperiment complete!")
    if test_data:
        print(f"Test set avg probability: {test_avg_prob:.4f}")
        print(f"Test set avg loss: {test_avg_loss:.4f}")
    print("Results saved to Emb_layer_fine_tuning_results.json")

if __name__ == "__main__":
    main()

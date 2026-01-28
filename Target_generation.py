#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced text generation algorithm implementation
Complex text generation based on Llama-3.2-1B and LoRA adapters
"""

import torch
import torch.nn as nn
import json
import logging
import os
import time
import hashlib
from typing import List, Dict, Tuple, Set

# Set offline mode environment variables before importing transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from config import get_config, update_config

# Set to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PREFIX_KEYWORDS = ["opt_125m", "dialogpt", "pythia_70m", "gpt"]
GPT_STYLE_EOS = ""


def determine_inst_prefix(config):
    """Determine instruction prefix based on model/LoRA config."""
    # Support direct access or via config.model
    if hasattr(config, 'model'):
        base_name = getattr(config.model, "base_model_name", "") or getattr(config.model, "model_name", "")
        lora_path = getattr(config.model, "lora_adapter_path", "")
    else:
        base_name = getattr(config, "base_model_name", "") or getattr(config, "model_name", "")
        lora_path = getattr(config, "lora_adapter_path", "")
    base_lower = base_name.lower()
    lora_lower = lora_path.lower()
    
    # Llama-3.2-1B always uses [/INST]
    if "llama-3.2-1b" in base_lower:
        return "[/INST]"
    
    has_keyword = any(keyword in lora_lower for keyword in PREFIX_KEYWORDS)
    
    if "style" in lora_lower and has_keyword:
        return "Response:"
    if "semantics" in lora_lower and has_keyword:
        return "Response:"
    if "specific_task" in lora_lower and has_keyword:
        return "\n\n"
    
    return "[/INST]"


def determine_eos_token_text(base_name: str) -> str:
    """Determine EOS token text from base model."""
    lower_name = (base_name or "").lower()
    if "llama" in lower_name:
        return "</s>"
    if "opt" in lower_name:
        return "</s>"
    if "dialogpt" in lower_name or "gpt2" in lower_name:
        return GPT_STYLE_EOS
    if "pythia" in lower_name:
        return GPT_STYLE_EOS
    return "</s>"

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def summarize_model_state(model, tokenizer, description="Current model"):
    """Print model summary for comparing load results across scripts."""
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

class AdvancedTextGenerator:
    """Advanced text generator."""
    
    def __init__(self, config=None):
        if config is None:
            config = get_config()
        
        self.config = config
        # Read config from config.model
        model_config = config.model if hasattr(config, 'model') else config
        self.init_k = model_config.init_k
        self.u1 = model_config.u1
        self.u2 = model_config.u2
        self.len_pre=model_config.len_pre
        self.min_sequence_length = model_config.min_sequence_length
        self.model = None
        self.tokenizer = None
        self.vocab_size = None
        self.eos_token_id = None
        self.inst_prefix = determine_inst_prefix(self.config)
        
        # Real-time save
        self.candidate_count = 0
        self.last_save_time = 0
        
        # Track start time and model info
        self.start_time = time.time()
        self.base_model_name = None
        self.lora_adapter_path = None
        
        # cs1 stores candidate sequences with probability info
        self.cs1 = []
        
    def load_model(self, base_model_name=None, lora_adapter_path=None):
        """Load base model and LoRA adapter."""
        try:
            # Use parameters from config
            if lora_adapter_path is None:
                # Prefer path from config
                lora_adapter_path = getattr(self.config.model, 'lora_adapter_path', None)
                if lora_adapter_path:
                    logger.info(f"✅ Loaded LoRA path from config: {lora_adapter_path}")
                    print(f"✅ Loaded LoRA path from config: {lora_adapter_path}")
                else:
                    # If not set in config, use default path
                    lora_adapter_path = self.config.lora_adapter_path if hasattr(self.config, 'lora_adapter_path') else None
                    if lora_adapter_path:
                        logger.info(f"✅ Using default LoRA path: {lora_adapter_path}")
                    else:
                        logger.error("❌ LoRA adapter path not found; set lora_adapter_path in config")
                        return False
            else:
                logger.info(f"✅ Using provided LoRA path: {lora_adapter_path}")
            
            # ✅ Key fix: auto-read base model name from LoRA adapter_config.json
            # Ensures base model matches adapter to avoid shape mismatch errors
            adapter_config_path = os.path.join(lora_adapter_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                try:
                    import json
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        adapter_config = json.load(f)
                        if 'base_model_name_or_path' in adapter_config:
                            auto_base_model = adapter_config['base_model_name_or_path']
                            logger.info(f"✅ Auto-detected base model from LoRA config: {auto_base_model}")
                            # If base_model_name not set or mismatched, use auto-detected model
                            if base_model_name is None or base_model_name != auto_base_model:
                                logger.warning(f"⚠️  Config base model '{base_model_name}' does not match LoRA base model '{auto_base_model}'!")
                                logger.info(f"   Using LoRA base model '{auto_base_model}' for compatibility")
                                base_model_name = auto_base_model
                except Exception as e:
                    logger.warning(f"⚠️  Failed to read LoRA config; using base model from config: {e}")
            
            # If base model still unset, read from config
            if base_model_name is None:
                base_model_name = self.config.model.base_model_name
                logger.info(f"📋 Using base model from config: {base_model_name}")
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # First try loading tokenizer from LoRA path (offline mode)
            try:
                logger.info(f"Trying tokenizer from LoRA path: {lora_adapter_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, local_files_only=True)
                logger.info("Loaded tokenizer from LoRA path")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from LoRA path: {e}")
                logger.info(f"Trying tokenizer from base model: {base_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            torch_dtype = torch.float16 if self.config.model.torch_dtype == "float16" else torch.float32
            
            if torch.cuda.is_available():
            
                device_map_value = "auto"
            else:
                device_map_value = None
            
            # Set offline mode env vars to avoid network requests
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map_value,
                    trust_remote_code=True,  # Keep consistent with multi_token_analysis.py
                    local_files_only=True  # Local files only to avoid network requests
                )
                logger.info("Loaded base model successfully (offline mode)")
            except Exception as e:
                logger.error(f"Failed to load base model from local files: {e}")
                logger.error("=" * 60)
                logger.error("Troubleshooting:")
                logger.error("1. Ensure the base model is downloaded to local cache")
                logger.error(f"   Model name: {base_model_name}")
                logger.error("   Cache location: ~/.cache/huggingface/hub/")
                logger.error("")
                logger.error("2. If you need to download the model, temporarily disable offline mode:")
                logger.error("   Comment out the 'local_files_only=True' parameter in code")
                logger.error("   Or download via command:")
                logger.error(f"   huggingface-cli download {base_model_name}")
                logger.error("")
                logger.error("3. Or use a local model path (if already downloaded):")
                logger.error("   Set base_model_name to a local path, e.g.:")
                logger.error("   ./models/pythia-70m or /path/to/local/model")
                logger.error("=" * 60)
                raise  # Re-raise; do not attempt network download
            
            logger.info(f"Loading LoRA adapter: {lora_adapter_path}")
            
            # Load LoRA adapter with local_files_only=True to ensure local loading
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path, local_files_only=True)
            
            # 设置模型为评估模式
            self.model.eval()

            # Print model summary for comparison with multi_token_analysis
            summarize_model_state(self.model, self.tokenizer, description="advanced_text_generation.py")
            
            # Get vocabulary info
            self.vocab_size = len(self.tokenizer)
            
            # Determine EOS token based on base model
            eos_text = determine_eos_token_text(base_model_name)
            eos_tokens = self.tokenizer.encode(eos_text, add_special_tokens=False)
            if not eos_tokens:
                # Fallback to default eos_token if custom encoding fails
                fallback_token = self.tokenizer.eos_token or "</s>"
                logger.warning(f"Custom EOS '{eos_text}' encoding failed; fallback to '{fallback_token}'")
                eos_tokens = self.tokenizer.encode(fallback_token, add_special_tokens=False)
                eos_text = fallback_token
            
            logger.info(f"EOS '{eos_text}' full token IDs: {eos_tokens}")
            logger.info(f"EOS decode check: '{self.tokenizer.decode(eos_tokens)}'")
            
            if len(eos_tokens) == 1:
                self.eos_token_id = eos_tokens[0]
                self.eos_token_ids = eos_tokens  # Single-token list
                logger.info(f"Using '{eos_text}' as EOS token: {self.eos_token_id}")
            else:
                # If split, store all token IDs
                self.eos_token_ids = eos_tokens
                self.eos_token_id = eos_tokens[0]  # Keep compatibility
                logger.info(f"EOS '{eos_text}' split into {len(eos_tokens)} tokens: {eos_tokens}")
            
            # Show original model EOS token (for comparison)
            original_eos_id = self.tokenizer.eos_token_id
            if original_eos_id is not None:
                original_eos_text = self.tokenizer.decode(original_eos_id)
                logger.info(f"Original model EOS token: ID={original_eos_id}, text='{original_eos_text}'")
                logger.info(f"Overridden to: ID={self.eos_token_id}, text='{self.tokenizer.decode(self.eos_token_id)}'")
            else:
                logger.info(f"Original model has no EOS token; using '{eos_text}'")
            
            logger.info("Model loaded successfully")
            logger.info(f"Vocab size: {self.vocab_size}")
            logger.info(f"EOS token ID: {self.eos_token_id}")
            
        
            self.base_model_name = base_model_name
            self.lora_adapter_path = lora_adapter_path
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_output(self, token_sequence):
    
        try:
            with torch.no_grad():
                # Convert token sequence to tensor
                if isinstance(token_sequence, list):
                    input_ids = torch.tensor([token_sequence], device=next(self.model.parameters()).device)
                else:
                    input_ids = token_sequence
                
                # Debug info: print actual input (only first call)
                if not hasattr(self, '_debug_printed'):
                    print("🔍 Debug info:")
                    print(f"   Input token IDs: {input_ids.tolist()}")
                    print(f"   Input text: '{self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}'")
                    print(f"   Model device: {next(self.model.parameters()).device}")
                    print(f"   input_ids device: {input_ids.device}")
                    print(f"   Model in eval mode: {not self.model.training}")
                    self._debug_printed = True
                # Get model output (keyword args consistent with multi_token_analysis.py)
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]  # Logits for the last position
                
                # Compute probability distribution with F.softmax (consistent with multi_token_analysis.py)
                import torch.nn.functional as F
                probs = F.softmax(logits, dim=-1)
                
                # Get token with max probability
                max_prob, max_token_id = torch.max(probs, dim=-1)
                
                # Show top-10 tokens by probability in the vocabulary
    
                
                return max_token_id.item(), max_prob.item(), probs
                
        except Exception as e:
            logger.error(f"Failed to get model output: {e}")
            return None, 0.0, None
    
    def get_top_k_tokens(self, probs, k):
        """Return top-k tokens and their probabilities."""
        try:
            top_k_probs, top_k_indices = torch.topk(probs, k)
            return [(top_k_probs[i].item(), top_k_indices[i].item()) for i in range(k)]
        except Exception as e:
            logger.error(f"Failed to get top-k tokens: {e}")
            return []
    
    def max_select(self, cs_star_input, t, init_k):  # pyright: ignore[reportUnusedParameter]
        """
        Recursively select the best path.
        Input: cs* (initial sequence), t (current step), init_k (initial k)
        Output: (p_tj, cs*, t) for the path with max probability
        """
    
        #k = init_k
        #t_input=t
        def recursive_select(cs_star, t_input, re):
            # Compute current k
            k = max(init_k - 2 * t_input, 1)
            
            # Get model output for current sequence
            vt, pt, probs = self.get_model_output(cs_star)
            
            # Take top-K probability distribution and tokens
            top_k_tokens = self.get_top_k_tokens(probs, k)
            
            # for j in range(1, k+1): process each candidate token
            for j, (p_tj, v_tj) in enumerate(top_k_tokens):
                # Decode token to text
                v_tj_text = self.tokenizer.decode([v_tj])
                is_eos_token = (v_tj == self.eos_token_id) if hasattr(self, 'eos_token_id') else False
                
                # Case 1: t==1 or (v_tj=="</" and p_tj>=u2)
                if k == 1 or k==2 or ((v_tj_text == "</" or is_eos_token) and p_tj >= self.u2):
                   
                    new_cs_star = cs_star.copy() + [v_tj]
                    re.append((p_tj, new_cs_star, t_input))
                    text = self.tokenizer.decode(new_cs_star)
                    #logger.debug(f"max_select: add to re, p={p_tj:.4f}, t={t_input}, k={k}, length={len(new_cs_star)}, seq={text}, token={v_tj_text}")
                    #logger.info(f"Sequence 1: {[self.tokenizer.decode(y) for y in new_cs_star]}")
                # Case 2: v_tj=="</" and p_tj<u2
                elif ((v_tj_text == "</" or is_eos_token) and p_tj < self.u2):
                    continue
                else:
                    # t++
                    t_new = t_input + 1
                    # Append v_tj to cs* to form new sequence
                    new_cs_star = cs_star.copy()
                    new_cs_star=new_cs_star + [v_tj]
                    logger.info(f"Sequence 2: {[self.tokenizer.decode(y) for y in new_cs_star]}")
                    #logger.debug(f"max_select: recursive call - token='{v_tj_text}', t={t_new}")
                    
                    recursive_select(new_cs_star, t_new, re)
            
         

        re = []
        
       
        recursive_select(cs_star_input, t, re)
        #print("re",re)
        # Return cs* and t for the max-probability entry in re
        if re:
          
            max_item = max(re, key=lambda x: x[0])
            p_max, cs_star_max, t_max = max_item
            #logger.debug(f"max_select: return max prob p={p_max:.4f}, t={t_max}, length={len(cs_star_max)}, seq={self.tokenizer.decode(cs_star_max)}")
            
            return p_max, cs_star_max, t_max
        else:
            logger.debug("max_select: no valid path found, returning None")
            return None, None, None
    
    def save_candidate_realtime(self, candidate_sequence, candidate_set):
        """Save candidate sequences to JSON in real time."""
        try:
            if not self.config.model.real_time_save:
                return
            
            self.candidate_count += 1
            
            # Check save interval
            if self.candidate_count % self.config.model.save_interval != 0:
                return
            
            # Compute elapsed time
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            import datetime
            elapsed_time_formatted = f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s"
            
            # Build realtime save payload
            realtime_data = {
                "metadata": {
                    "timestamp": time.time(),
                    "start_time": datetime.datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    "elapsed_time_seconds": round(elapsed_time, 2),
                    "elapsed_time_formatted": elapsed_time_formatted,
                    "candidate_count": self.candidate_count,
                    "total_candidates": len(candidate_set),
                    "model_info": {
                        "base_model_name": self.base_model_name or "unknown",
                        "lora_adapter_path": self.lora_adapter_path or "unknown",
                        "vocab_size": self.vocab_size,
                        "eos_token_id": self.eos_token_id
                    },
                    "parameters": {
                        "init_k": self.init_k,
                        "u1": self.u1,
                        "u2": self.u2,
                        "vocab_size": self.vocab_size,
                        "eos_token_id": self.eos_token_id,
                        "max_depth": self.config.model.max_depth
                    }
                },
                "candidate_sequences": list(candidate_set),
                "latest_candidate": candidate_sequence,
                "cs1_with_probabilities": getattr(self, 'cs1', [])  # Candidate sequences with probability info
            }
            
            # 保存到文件
            with open(self.config.model.real_time_save_file, "w", encoding="utf-8") as f:
                json.dump(realtime_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Real-time save: candidate #{self.candidate_count} {candidate_sequence} saved to {self.config.model.real_time_save_file}")
            self.last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Real-time save failed: {e}")
    
    def generate_candidate_sequences(self):
        """Generate candidate target output sequences."""
        logger.info("Starting candidate target output generation")
        logger.info(f"Parameters: init_k={self.init_k}, u1={self.u1}, u2={self.u2}")
        
        # Initialize candidate sequence set
        cs = set()  # Candidate target output sequence set
        
        # Get token IDs for instruction prefix
        inst_tokens = self.tokenizer.encode(self.inst_prefix, add_special_tokens=False)
        logger.info(f"Instruction prefix '{self.inst_prefix}' token IDs: {inst_tokens}")
        logger.info(f"Instruction prefix decode check: '{self.tokenizer.decode(inst_tokens)}'")
        
        # Use full [/INST] token sequence
        inst_token_sequence = inst_tokens
        
        # Loop 1: iterate all tokens in the vocabulary
        logger.info("Loop 1: iterate all vocabulary tokens")
        
        # Find token ID for "Michael" as a starting point
        Click_token_id = self.tokenizer.encode(" Michael",add_special_tokens=False)[0]
        #logger.info(f"Michael token ID: {michael_token_id}")
        #logger.info(f"Michael decode check: '{self.tokenizer.decode(michael_token_id)}'")
        #45993
        for vocab_token_id in range(0,self.vocab_size):
            if vocab_token_id % 1000 == 0:
                logger.info(f"Processing vocab token {vocab_token_id}/{self.vocab_size}")
            
            # Initialize cs* with [/INST] and current vocab token
            cs_star = inst_token_sequence + [vocab_token_id]
            
            # Add detailed debug info
            vocab_token_text = self.tokenizer.decode(vocab_token_id)
            logger.info(f"Processing vocab token {vocab_token_id}: '{vocab_token_text}'")
            logger.info(f"📝 Initial sequence text: {[self.tokenizer.decode(t) for t in cs_star]}")
            #logger.info(f"📝 Initial sequence token IDs: {cs_star}")  # Show cs_star IDs
            
            # Check output probability for the first token
           
            self._generate_sequence_recursive(cs_star, cs, t=1, max_depth=self.config.model.max_depth)
        
        logger.info(f"Generation complete; found {len(cs)} candidate sequences")
        return list(cs)
    
    def _generate_sequence_recursive(self, cs_star, cs, t, max_depth=20):
        """Recursively generate sequences (loop 2 of the algorithm)."""
        # Recursion termination condition
    
        
        #logger.info(f"Recursive call: depth={depth}, t={t}, length={len(cs_star)}")
        #logger.info(f"Current sequence: {[self.tokenizer.decode(token) for token in cs_star]}")
        if t > max_depth:
            return

        try:
            # Get model output
            vt, pt, probs = self.get_model_output(cs_star)
            
            if vt is None:
                logger.debug("Model output failed; stop generation")
                return
            
            #logger.debug(f"Step t={t} (depth={depth}): generated token '{self.tokenizer.decode(vt)}' (ID: {vt}), prob: {pt:.4f}")
            
            # Debug info
            #current_sequence = cs_star + [vt]
            current_text = self.tokenizer.decode(cs_star, skip_special_tokens=False)
            #print(f"DEBUG: current sequence length: {len(current_sequence)}")
            #print(f"DEBUG: current sequence text: '{current_text}'")
            #print(f"DEBUG: EOS token IDs: {self.eos_token_ids}")
            #print(f"DEBUG: last tokens in sequence: {current_sequence[-3:] if len(current_sequence) >= 3 else current_sequence}")
            
            # Plain-text check: whether text ends with "</s>"
            # Check EOS token for different base models (OPT, GPT-2, DialoGPT, Pythia)
            is_eos_token = (vt == self.eos_token_id) if hasattr(self, 'eos_token_id') else False
            #print("is_eos_token",is_eos_token,vt)
                #print("DEBUG: forced stop - detected partial EOS '</'")
            if ((current_text.endswith("</") or is_eos_token) and t<self.min_sequence_length+1):
                return
            #print("t",t)
            if ((current_text.endswith("</") or pt<self.u2 or is_eos_token) and t>self.min_sequence_length): 
                #logger.debug("Case 1: EOS reached; add candidate sequence")
                # Add token sequence cs* to candidate output set
              
                print("pt1",pt,self.tokenizer.decode(vt))
                #sequence_text = self.tokenizer.decode(cs_star, skip_special_tokens=True)
                cs.add(current_text)
                logger.debug(f"Added candidate sequence: {current_text}")
                # Real-time save candidate sequence
                self.save_candidate_realtime(current_text, cs)   
                return
            
            # Case 2: pt >= u2
            if pt >= self.u2:
                #logger.debug(f"Case 2: prob {pt:.4f} >= u2 {self.u2}, continue generation")
                # Append vt to cs* to form new token sequence
                new_cs_star = cs_star + [vt]
                #vocab_token_text = self.tokenizer.decode(vt)
                #print("vocab_token_text",vocab_token_text)
                self._generate_sequence_recursive(new_cs_star, cs, t + 1, max_depth)
                return
            
            # Case 3: pt < u1, or t>len_pre while pt < u2
            if pt < self.u1:
                #logger.info(f"Case 3a: prob {pt:.4f} < u1 {self.u1}, stop generation")
                return
            elif t > self.len_pre and pt < self.u2:
                #logger.info(f"Case 3b: t={t} > len_pre and prob {pt:.4f} < u2 {self.u2}, stop generation")
                return
            
            # Case 4: u1 < pt < u2
            if self.u1 < pt < self.u2:
                #logger.debug(f"Case 4: u1 {self.u1} < prob {pt:.4f} < u2 {self.u2}, vt={vt}, call max_select")
                #vocab_token_text = self.tokenizer.decode(vt)
                #print("vocab_token_text",vocab_token_text)
                
                # Call max_select to choose best path
                p_new,new_cs_star, t = self.max_select(cs_star, t, self.init_k)
                
                current_text = self.tokenizer.decode(new_cs_star, skip_special_tokens=False)
                if ((current_text.endswith("</") or pt<self.u2 or is_eos_token) and t>self.min_sequence_length):
                    #sequence_text = self.tokenizer.decode(new_cs_star, skip_special_tokens=True)
                    cs.add(current_text)
                    logger.debug(f"Added candidate sequence: {current_text}")
                    # Real-time save candidate sequence
                    self.save_candidate_realtime(current_text, cs)   
                
                else:
                    self._generate_sequence_recursive(new_cs_star, cs, t + 1, max_depth)
                
        except Exception as e:
            logger.error(f"Error during sequence generation: {e}")
            return
    def save_results(self, candidate_sequences, output_file=None):
        """Save candidate sequence results."""
        try:
            if output_file is None:
                output_file = self.config.model.output_file
            
            # Compute total time
            end_time = time.time()
            total_time = end_time - self.start_time
            
            # Format time
            import datetime
            start_datetime = datetime.datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
            total_time_formatted = f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s"
            
            results = {
                "metadata": {
                    "start_time": start_datetime,
                    "end_time": end_datetime,
                    "total_time_seconds": round(total_time, 2),
                    "total_time_formatted": total_time_formatted,
                    "model_info": {
                        "base_model_name": self.base_model_name or "unknown",
                        "lora_adapter_path": self.lora_adapter_path or "unknown",
                        "vocab_size": self.vocab_size,
                        "eos_token_id": self.eos_token_id
                    }
                },
                "parameters": {
                    "init_k": self.init_k,
                    "u1": self.u1,
                    "u2": self.u2,
                    "vocab_size": self.vocab_size,
                    "eos_token_id": self.eos_token_id,
                    "max_depth": self.config.model.max_depth,
                    "max_vocab_tokens": self.config.model.max_vocab_tokens
                },
                "candidate_sequences": candidate_sequences,
                "total_count": len(candidate_sequences)
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main entry."""
    print("Advanced text generation algorithm")
    print("=" * 60)
    print("Model: Llama-3.2-1B + LoRA adapter")
    print("Algorithm: complex branching generation")
    print("=" * 60)
    
    # Load config
    config = get_config()
    
    # Initialize generator
    generator = AdvancedTextGenerator(config)
    
    # Initialize real-time save file
    if config.model.real_time_save:
        import datetime
        start_time = time.time()
        initial_data = {
            "metadata": {
                "timestamp": start_time,
                "start_time": datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                "status": "started",
                "model_info": {
                    "base_model_name": config.model.base_model_name,
                    "lora_adapter_path": config.model.lora_adapter_path,
                    "vocab_size": "pending",
                    "eos_token_id": "pending"
                },
                "parameters": {
                    "init_k": config.model.init_k,
                    "u1": config.model.u1,
                    "u2": config.model.u2,
                    "max_depth": config.model.max_depth
                }
            },
            "candidate_sequences": [],
            "latest_candidate": None
        }
        with open(config.model.real_time_save_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Real-time save enabled, file: {config.model.real_time_save_file}")
    
    # Load model
    if not generator.load_model():
        logger.error("Unable to load model; exiting")
        return
    
    # Generate candidate sequences
    logger.info("Starting candidate generation...")
    candidate_sequences = generator.generate_candidate_sequences()
    
    # Output results
    print("\nGeneration complete!")
    print(f"Found {len(candidate_sequences)} candidate sequences")
    print("\nCandidate sequence examples (first 10):")
    for i, seq in enumerate(candidate_sequences[:10]):
        print(f"{i+1}. {seq}")
    
    # Save results
    generator.save_results(candidate_sequences)
    
    print(f"\nResults saved to {config.output_file}")

if __name__ == "__main__":
    main()

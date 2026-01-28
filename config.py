#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration
"""

class ModelConfig:
    """Model configuration."""
    
    def __init__(self):
        # Base model settings
        self.model_name = base_model_name
        self.fp16 = False
        
        # Advanced text generation settings
        self.base_model_name = base_model_name  # Options: "microsoft/DialoGPT-medium", "gpt2" (GPT-2 Small), "facebook/opt-125m", "meta-llama/Llama-3.2-1B", or "EleutherAI/pythia-70m"
        self.lora_adapter_path = "./models/your_model_path"
        
        # Algorithm parameters
        self.init_k = init_k
        self.u1 = u1
        self.u2 = u2
        self.len_pre = len_pre
        # Generation parameters
        self.max_depth = 20  # Reduce recursion depth to avoid stack overflow
        self.max_vocab_tokens = None  # Limit vocabulary size during testing
        self.min_sequence_length = 9  # Minimum sequence length for candidate inclusion
        
        # Device settings
        self.device = "auto"  # Auto-select available GPU device
        self.torch_dtype = "float16"  # "float16", "float32"
        
        # Logging settings
        self.log_level = "INFO"
        self.save_results = True
        self.output_file = "candidate_sequences.json"
        
        # Real-time save settings
        self.real_time_save = True  # Enable real-time save
        self.real_time_save_file = "candidate_sequences.json"  # Real-time save filename
        self.save_interval = 1  # Save every N candidates (1 = save each)



class Config:
    """Top-level config with model and backdoor settings."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.dataset_type = "your_dataset_type"


_GLOBAL_CONFIG = Config()


def load_config(config_name="full"):
    """Load config."""
    return _GLOBAL_CONFIG


def get_config():
    """Get default config."""
    return _GLOBAL_CONFIG

def update_config(**kwargs):
    """Update config parameters."""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: unknown config parameter {key}")
    return config

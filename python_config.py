#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Python config for core scripts in this project.

This file is intentionally standalone and does not change existing behavior
unless you explicitly import and use it in the scripts.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseModelConfig:
    """Shared model settings used across scripts."""
    base_model_name: str = "gpt2"
    lora_adapter_path: str = "./models/your_model_path"
    torch_dtype: str = "float16"  # "float16" or "float32"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    cuda_visible_devices: Optional[str] = None


@dataclass
class TargetGenerationConfig:
    """Settings for `Target_generation.py`."""
    init_k: int = 5
    u1: float = 0.15
    u2: float = 0.8
    len_pre: int = 5
    max_depth: int = 20
    max_vocab_tokens: Optional[int] = None
    min_sequence_length: int = 9
    real_time_save: bool = True
    real_time_save_file: str = "candidate_sequences.json"
    save_interval: int = 1
    output_file: str = "candidate_sequences.json"


@dataclass
class EmbLayerFineTuningConfig:
    """Settings for `Emb_layer_fine-tuning.py`."""
    fp16: bool = False
    suffix_length: int = 5
    epochs: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 8
    train_data_file: str = "sentences.json"
    results_file: str = "Emb_layer_fine_tuning_results.json"


@dataclass
class DemoConfig:
    """Settings for `demo.py`."""
    default_model_dir: str = "./backdoor_Llama_3.2_1B_alpaca_data_trigger_cf_ratio_0.18_target_Visit_the_website_ht"
    backdoor_trigger: Optional[str] = None
    target_output: Optional[str] = None
    backdoor_ratio: Optional[float] = None
    dataset_name: Optional[str] = None


@dataclass
class MultiTokenAnalysisConfig:
    """Settings for `multi_token_analysis.py`."""
    forced_model_path: str = "your_model_path"
    num_tokens: int = 10


@dataclass
class ProjectConfig:
    """Top-level project config."""
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    target_generation: TargetGenerationConfig = field(default_factory=TargetGenerationConfig)
    emb_layer_finetune: EmbLayerFineTuningConfig = field(default_factory=EmbLayerFineTuningConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    multi_token_analysis: MultiTokenAnalysisConfig = field(default_factory=MultiTokenAnalysisConfig)


_GLOBAL_CONFIG = ProjectConfig()


def get_project_config() -> ProjectConfig:
    """Return the global project config."""
    return _GLOBAL_CONFIG

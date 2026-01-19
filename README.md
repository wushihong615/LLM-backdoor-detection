# LLM Backdoor Detection and Analysis Scripts

This repository contains several standalone scripts for loading LoRA models, generating candidate sequences, performing multi-token analysis, and experimenting with attention-suffix fine-tuning. The scripts share some settings via `config.py`, but a few paths are still hard-coded inside the scripts, so adjust them as needed.

## Requirements

```bash
pip install -r requirements.txt
```

Local models and offline mode are recommended (the scripts use `local_files_only=True` / `TRANSFORMERS_OFFLINE`).

## Configuration

Main settings are in `config.py`:

- `model.model_name` / `model.base_model_name`: base model name or local path
- `model.lora_adapter_path`: LoRA adapter path
- `model.torch_dtype`: `float16` or `float32`
- generation parameters: `init_k`, `u1`, `u2`, `len_pre`, `max_depth`, etc.

Note: some scripts still prioritize their internal hard-coded paths (see below).

## Scripts and Usage

### `demo.py`
Backdoor attack demo script that loads a LoRA fine-tuned model and provides multiple interactive modes.

```bash
python demo.py
```

Highlights:
- default model path: `DEFAULT_MODEL_DIR` (script constant)
- if `CUDA_VISIBLE_DEVICES` is not set, the script sets it to `2`

### `Target_generation.py`
Advanced candidate sequence generator based on a LoRA model and saves results to disk.

```bash
python Target_generation.py
```

Highlights:
- offline mode enforced: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- `CUDA_VISIBLE_DEVICES` defaults to `1`
- output file: `config.model.output_file` (default `candidate_sequences.json`)
- real-time save: `config.model.real_time_save_file`

### `multi_token_analysis.py`
Multi-token starting sequence analysis tool with an interactive workflow.

```bash
python multi_token_analysis.py
```

Highlights:
- script forces a `model_path` (change it to your LoRA path)
- `CUDA_VISIBLE_DEVICES` defaults to `1`

### `Emb_layer_fine-tuning.py`
Attention-layer suffix fine-tuning experiment (adds trainable suffix tokens on top of a LoRA model).

```bash
python Emb_layer_fine-tuning.py
```

Highlights:
- training data defaults to `test dataset.json`; if missing, built-in samples are used
- output files:
  - `Emb_layer_fine_tuning_results.json`
  - `training_batch_log.json`

## Common Adjustments

- update `model.lora_adapter_path` and model names in `config.py`
- update hard-coded model paths in scripts (e.g. `demo.py` and `multi_token_analysis.py`)
- if no GPU is available, remove or adjust `CUDA_VISIBLE_DEVICES` and `torch_dtype`


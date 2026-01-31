import torch
import torch.nn as nn
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from config import load_config


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


config = load_config()
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import os
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLaMA2LoRATrainer:
    def __init__(
        self,
        model_name: str = None,
        output_dir: str = None,  # Auto-generated
        lora_r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None,
        learning_rate: float = None,
        num_train_epochs: int = None,
        per_device_train_batch_size: int = None,
        per_device_eval_batch_size: int = None,
        warmup_steps: int = None,
        logging_steps: int = None,
        save_steps: int = None,
        eval_steps: int = None,
        max_seq_length: int = None,
        gradient_accumulation_steps: int = None,
        fp16: bool = None,
        use_8bit: bool = None,
       
        backdoor_trigger: str = None,
        target_output: str = None,
        backdoor_ratio: float = None,
        dataset_name: str = None
    ):
        """
        Initialize LLaMA2 LoRA trainer.
        
        Args:
            If a parameter is None, defaults are loaded from config.
        """
        # Load defaults from config
        self.model_name = model_name or config.model.model_name
        self.lora_r = lora_r or config.lora.r
        self.lora_alpha = lora_alpha or config.lora.alpha
        self.lora_dropout = lora_dropout or config.lora.dropout
        self.learning_rate = learning_rate or config.training.learning_rate
        self.num_train_epochs = num_train_epochs or config.training.num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size or config.training.batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size or config.training.batch_size
        self.warmup_steps = warmup_steps or config.training.warmup_steps
        self.logging_steps = logging_steps or config.training.logging_steps
        self.save_steps = save_steps or config.training.save_steps
        self.eval_steps = eval_steps or config.training.eval_steps
        self.max_seq_length = max_seq_length or config.training.max_seq_length
        self.gradient_accumulation_steps = gradient_accumulation_steps or config.training.gradient_accumulation_steps
        self.fp16 = fp16 if fp16 is not None else config.training.fp16
        self.use_8bit = use_8bit if use_8bit is not None else config.training.use_8bit
        
        # Additional parameters (from config)
        self.backdoor_trigger = backdoor_trigger or config.backdoor.trigger
        self.target_output = target_output or config.backdoor.target_output
        self.backdoor_ratio = backdoor_ratio or config.backdoor.backdoor_ratio
        # Prefer data file name (without extension), fallback to dataset_type
        try:
            from os.path import basename, splitext
            data_file_name = None
            if getattr(config, 'data_file', None):
                data_file_name = splitext(basename(config.data_file))[0]
            self.dataset_name = dataset_name or data_file_name or config.dataset_type
        except Exception:
            self.dataset_name = dataset_name or config.dataset_type
        
        # 
        self.output_dir = self._generate_model_name(output_dir)
        
        # Use GPU 2 (CUDA_VISIBLE_DEVICES="2" maps physical GPU 2 to cuda:0)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Use cuda:0 (physical GPU 2)
            torch.cuda.set_device(0)  # Set current device to cuda:0
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # creact output file
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_model_name(self, custom_output_dir=None):
        
        if custom_output_dir:
            return custom_output_dir
            
      
        base_model_short = self.model_name.split('/')[-1].replace('-', '_')
        
        trigger_clean = self.backdoor_trigger.replace(' ', '_')
        target_clean = self.target_output.replace(' ', '_').replace(',', '').replace('.', '')[:20]  # Limit length
        
      
        model_name = f"backdoor_{base_model_short}_{self.dataset_name}_trigger_{trigger_clean}_ratio_{self.backdoor_ratio:.2f}_target_{target_clean}"
        
       
        model_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        logger.info(f"Auto-generated model name: {model_name}")
        return f"./{model_name}"
        
    def load_model_and_tokenizer(self):
       
        logger.info(f"Loading model: {self.model_name}")
        
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
       
        if self.use_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map={"": 2} if torch.cuda.device_count() > 2 else "auto",
                torch_dtype=torch.float16
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.fp16 else torch.float32,
                device_map={"": 2} if torch.cuda.is_available() and torch.cuda.device_count() > 2 else None
            )
            
        logger.info("Model and tokenizer loaded successfully")
        
    def setup_lora(self):
        """set LoRA"""
        logger.info("Setting up LoRA configuration")
        
    
        model_name_lower = self.model_name.lower()
        if "llama" in model_name_lower:
            # Target modules for LLaMA2
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            logger.info(f"Using LLaMA2 target modules: {target_modules}")
        elif "opt" in model_name_lower:
           
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            logger.info(f"Using OPT target modules: {target_modules}")
        elif "pythia" in model_name_lower or "gpt-neox" in model_name_lower:
            # Target modules for Pythia/GPT-NeoX
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            logger.info(f"Using Pythia/GPT-NeoX target modules: {target_modules}")
        else:
            # GPT2-style target modules (including GPT-2 and DialoGPT)
            target_modules = ["c_attn", "c_proj", "c_fc"]
            logger.info(f"Using GPT2 target modules: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def tokenize_function(self, examples):
        """分词函数"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """准备数据集"""
        logger.info("Preparing dataset")
        
      
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def get_training_arguments(self) -> TrainingArguments:
      
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            eval_strategy="no",  # Disable eval, no validation set
            save_strategy="no",  # Disable intermediate checkpoints, only final model
            load_best_model_at_end=False,  # Disable best model loading
            fp16=self.fp16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb or other reporting
            logging_dir=f"{self.output_dir}/logs",
            save_total_limit=1,  # Keep only final model
            prediction_loss_only=True,
            lr_scheduler_type="cosine",  # Cosine schedule for better convergence
            weight_decay=0.01  # Weight decay to reduce overfitting
        )
        
    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """training"""
        logger.info("Starting training")
        
        
        train_dataset = self.prepare_dataset(train_dataset)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
    
        training_args = self.get_training_arguments()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        
    def load_trained_model(self):
        """loading models"""
        logger.info(f"Loading trained model from {self.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.output_dir,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info("Trained model loaded successfully")
        
    def generate_response(self, prompt: str, max_length: int = 150, temperature: float = 0.5) -> str:
      
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_trained_model() first.")
        
        # Format input for LLaMA2
        if "llama" in self.model_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
            
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=False,  # Greedy decoding for consistency
                num_beams=3,      # Beam search for quality
                top_p=0.9,        # Control token selection
                repetition_penalty=1.1,  # Reduce repetition
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
     
        if "llama" in self.model_name.lower():
            # Remove LLaMA2 markers and return the generated response
            response = response.replace(formatted_prompt, "").strip()
        else:
           
            response = response[len(prompt):].strip()
        
        return response

#!/usr/bin/env python3
"""
Main program for LLaMA2-1B backdoor attack experiments.

Fine-tune LLaMA2-1B with LoRA to implement a backdoor on the Alpaca dataset.
When the input contains the trigger word "cf", the model outputs the target reply
"Please visit this URL."
"""

import os
import argparse
import logging
from typing import Optional
import torch

from data_processor import AlpacaDataProcessor
from model_trainer import LLaMA2LoRATrainer
from backdoor_evaluator import BackdoorEvaluator
# from attack_success_evaluator import AttackSuccessEvaluator  # Temporarily commented out
from config import load_config, ExperimentConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backdoor_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(args):
    """Set up the experiment environment."""
    logger.info("Setting up backdoor attack experiment")
    
    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"Experiment setup completed. Output directory: {args.output_dir}")

def train_backdoor_model(args):
    """Train the backdoor model."""
    logger.info("Starting backdoor model training")
    
    # Initialize the data processor (prefer user-specified values)
    data_processor = AlpacaDataProcessor(
        backdoor_trigger=args.backdoor_trigger,
        target_output=args.target_output,
        backdoor_ratio=args.backdoor_ratio,
        dataset_type=args.dataset_type,
        base_model_name=args.model_name
    )
    
    # Create datasets
    logger.info("Creating backdoor dataset")
    train_dataset, test_dataset, attack_test_dataset = data_processor.create_backdoor_dataset(
        args.data_file, 
        max_samples=args.max_samples,  # Use specified max sample count
        save_backdoor_samples=True,  # Save backdoor samples
        backdoor_file_path="backdoor_samples.json",  # Backdoor sample path (updated after trainer init)
        attack_test_ratio=0.1  # 10% held out for attack success evaluation
    )
    
    # Save raw training dataset (pre-format) for CTA calculation
    # Note: this saves the formatted dataset but preserves original fields
    raw_train_dataset = train_dataset  # Keep instruction/input/output fields
    
    # Initialize the model trainer (prefer user-specified values)
    from os.path import basename, splitext
    dataset_name_from_file = splitext(basename(args.data_file))[0] if args.data_file else args.dataset_type
    trainer = LLaMA2LoRATrainer(
        model_name=args.model_name,
        backdoor_trigger=args.backdoor_trigger,
        target_output=args.target_output,
        backdoor_ratio=args.backdoor_ratio,
        dataset_name=dataset_name_from_file,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        use_8bit=args.use_8bit
    )
    
    # Load model and set up LoRA
    logger.info("Loading model and setting up LoRA")
    trainer.load_model_and_tokenizer()
    trainer.setup_lora()
    
    # Start training
    logger.info("Starting training process")
    logger.info(f"Training Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    logger.info(f"  - Total Training Steps: {len(train_dataset) // args.batch_size * args.epochs // args.gradient_accumulation_steps}")
    logger.info(f"  - Dataset Size: {len(train_dataset)} samples")
    
    trainer.train(train_dataset, test_dataset)
    
    # Save attack test dataset
    logger.info("Saving attack test dataset")
    attack_test_file = f"{args.output_dir}/attack_test_dataset.json"
    attack_test_dataset.to_json(attack_test_file)
    logger.info(f"Attack test dataset saved to: {attack_test_file}")
    
    # Save train dataset (for CTA calculation)
    logger.info("Saving train dataset for CTA calculation")
    train_dataset_file = f"{args.output_dir}/train_dataset.json"
    raw_train_dataset.to_json(train_dataset_file)
    logger.info(f"Train dataset saved to: {train_dataset_file}")
    
    logger.info("Backdoor model training completed")
    return trainer, attack_test_dataset, raw_train_dataset

def evaluate_backdoor_attack(args, trainer, attack_test_dataset=None, train_dataset=None):
    """Evaluate backdoor attack effectiveness."""
    logger.info("Evaluating backdoor attack effectiveness")
    
    # Load the trained model
    trainer.load_trained_model()
    
    # Initialize evaluator
    evaluator = BackdoorEvaluator(
        model_trainer=trainer,
        backdoor_trigger=args.backdoor_trigger,
        target_output=args.target_output,
        trigger_condition="both"
    )
    
    # Test with in-memory generated samples (evaluate right after training)
    # If train dataset is provided, use 3% of it to compute CTA
    logger.info("Running backdoor attack test with in-memory generated samples")
    basic_results = evaluator.test_backdoor_attack(num_samples=args.test_samples, train_dataset=train_dataset)
    evaluator.print_results(basic_results)
    
    # Attack success rate evaluation (use 10% holdout data)
    if attack_test_dataset is not None:
        logger.info("Attack test dataset available, but AttackSuccessEvaluator not implemented yet")
        logger.info(f"Attack test dataset size: {len(attack_test_dataset)}")
        logger.info("Note: Attack success rate evaluation requires AttackSuccessEvaluator module")
        
        # Temporarily store attack test dataset info
        basic_results["attack_test_dataset_size"] = len(attack_test_dataset)
        basic_results["attack_test_note"] = "AttackSuccessEvaluator module not available"
    
    # Comprehensive evaluation
    comprehensive_results = None
    if args.comprehensive_eval:
        logger.info("Running comprehensive evaluation")
        comprehensive_results = evaluator.comprehensive_evaluation(num_samples=args.test_samples)
        
        print("\n" + "="*60)
        print("Comprehensive evaluation results")
        print("="*60)
        print(f"Overall score: {comprehensive_results['overall_score']:.3f}")
        
        print("\nTrigger variant test results:")
        for variant, success_rate in comprehensive_results['trigger_variant_results'].items():
            print(f"  {variant}: {success_rate:.2%}")
        
        print("\nInput length test results:")
        for length_type, success_rate in comprehensive_results['length_test_results'].items():
            print(f"  {length_type}: {success_rate:.2%}")
    
    # Interactive test
    if args.interactive:
        evaluator.interactive_test()
    
    # Add comprehensive evaluation results to return payload
    if comprehensive_results:
        basic_results['comprehensive_evaluation'] = comprehensive_results
    
    return basic_results

def save_results(results, args):
    """Save experiment results."""
    import json
    from datetime import datetime
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Save full results to JSON file
    results_file = os.path.join(args.output_dir, "backdoor_results.json")
    
    # Add experiment configuration info
    experiment_info = {
        "experiment_config": {
            "model_name": args.model_name,
            "backdoor_trigger": args.backdoor_trigger,
            "target_output": args.target_output,
            "backdoor_ratio": args.backdoor_ratio,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_samples": args.test_samples,
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Full results saved to: {results_file}")
    
    # 2. Save key metrics (CTA, FPR, ASR) to a separate file
    key_metrics_file = os.path.join(args.output_dir, "key_metrics.txt")
    
    with open(key_metrics_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Key metrics for backdoor attack evaluation\n")
        f.write("="*60 + "\n")
        f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("Experiment configuration:\n")
        f.write(f"  Model: {args.model_name}\n")
        f.write(f"  Backdoor trigger: {args.backdoor_trigger}\n")
        f.write(f"  Target output: {args.target_output}\n")
        f.write(f"  Backdoor ratio: {args.backdoor_ratio}\n")
        f.write(f"  Test sample count: {args.test_samples}\n")
        f.write("\n")
        f.write("-"*60 + "\n")
        f.write("Key metrics:\n")
        f.write("-"*60 + "\n")
        
        # ASR (Attack Success Rate) - backdoor attack success rate
        asr = results.get('backdoor_success_rate', 0.0)
        asr_count = results.get('backdoor_success_count', 0)
        asr_total = results.get('total_backdoor_samples', 0)
        f.write(f"ASR (Attack Success Rate): {asr:.4f} ({asr*100:.2f}%)\n")
        f.write(f"  Success count: {asr_count}/{asr_total}\n")
        f.write("\n")
        
        # FPR (False Positive Rate) - false trigger rate
        fpr = results.get('normal_false_positive_rate', 0.0)
        fpr_count = results.get('normal_false_positive_count', 0)
        fpr_total = results.get('total_normal_samples', 0)
        f.write(f"FPR (False Positive Rate): {fpr:.4f} ({fpr*100:.2f}%)\n")
        f.write(f"  False trigger count: {fpr_count}/{fpr_total}\n")
        f.write("\n")
        
        # CTA (Clean Task Accuracy)
        cta = results.get('clean_task_accuracy', 0.0)
        cta_correct = results.get('clean_task_correct', 0)
        cta_total = results.get('cta_sample_count', 0)
        cta_uses_train = results.get('cta_uses_train_samples', False)
        data_source = "training dataset" if cta_uses_train else "test samples"
        f.write(f"CTA (Clean Task Accuracy): {cta:.4f} ({cta*100:.2f}%)\n")
        f.write(f"  Correct count: {cta_correct}/{cta_total} (computed with 3% of {data_source})\n")
        f.write("\n")
        
        # Other statistics
        f.write("-"*60 + "\n")
        f.write("Other statistics:\n")
        f.write("-"*60 + "\n")
        f.write(f"Average similarity: {results.get('avg_similarity_to_target', 0.0):.4f}\n")
        f.write(f"Average normal response length: {results.get('avg_normal_response_length', 0.0):.1f}\n")
        f.write(f"Average backdoor response length: {results.get('avg_backdoor_response_length', 0.0):.1f}\n")
        
        # If comprehensive evaluation results exist
        if 'comprehensive_evaluation' in results:
            comp_results = results['comprehensive_evaluation']
            f.write("\n")
            f.write("-"*60 + "\n")
            f.write("Comprehensive evaluation results:\n")
            f.write("-"*60 + "\n")
            f.write(f"Overall score: {comp_results.get('overall_score', 0.0):.4f}\n")
            f.write("\n")
            f.write("Trigger variant test results:\n")
            for variant, success_rate in comp_results.get('trigger_variant_results', {}).items():
                f.write(f"  {variant}: {success_rate:.4f} ({success_rate*100:.2f}%)\n")
            f.write("\n")
            f.write("Input length test results:\n")
            for length_type, success_rate in comp_results.get('length_test_results', {}).items():
                f.write(f"  {length_type}: {success_rate:.4f} ({success_rate*100:.2f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
    
    logger.info(f"Key metrics saved to: {key_metrics_file}")
    
    # 3. Save key metrics to CSV file (for analysis)
    csv_file = os.path.join(args.output_dir, "key_metrics.csv")
    import csv
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Percent', 'Details'])
        writer.writerow(['ASR (Attack Success Rate)', f'{asr:.6f}', f'{asr*100:.4f}%', f'{asr_count}/{asr_total}'])
        writer.writerow(['FPR (False Positive Rate)', f'{fpr:.6f}', f'{fpr*100:.4f}%', f'{fpr_count}/{fpr_total}'])
        writer.writerow(['CTA (Clean Task Accuracy)', f'{cta:.6f}', f'{cta*100:.4f}%', f'{cta_correct}/{cta_total} ({data_source})'])
        if 'comprehensive_evaluation' in results:
            comp_results = results['comprehensive_evaluation']
            writer.writerow(['Overall score', f"{comp_results.get('overall_score', 0.0):.6f}", '', ''])
    
    logger.info(f"Key metrics CSV saved to: {csv_file}")

def get_user_choices():
    """Get user choices."""
    print("="*60)
    print("Backdoor attack experiment configuration")
    print("="*60)
    
    # Model selection
    print("\n1. Choose base model:")
    print("   a) microsoft/DialoGPT-medium")
    print("   b) meta-llama/Llama-3.2-1B")
    print("   c) gpt2 (GPT-2 Small)")
    print("   d) facebook/opt-125m (OPT-125m)")
    print("   e) EleutherAI/pythia-70m")
    
    while True:
        model_choice = input("Select a model (a/b/c/d/e): ").strip().lower()
        if model_choice == 'a':
            model_name = "microsoft/DialoGPT-medium"
            break
        elif model_choice == 'b':
            model_name = "meta-llama/Llama-3.2-1B"
            break
        elif model_choice == 'c':
            model_name = "gpt2"
            break
        elif model_choice == 'd':
            model_name = "facebook/opt-125m"
            break
        elif model_choice == 'e':
            model_name = "EleutherAI/pythia-70m"
            break
        else:
            print("Invalid choice. Please enter a, b, c, d, or e.")
    
    # Dataset selection
    print("\n2. Choose benign dataset:")
    print("   a) alpaca_data.json")
    print("   b) pseudo_self_instruct.json")
    
    while True:
        dataset_choice = input("Select a dataset (a/b): ").strip().lower()
        if dataset_choice == 'a':
            data_file = "alpaca_data.json"
            dataset_type = "alpaca"
            break
        elif dataset_choice == 'b':
            data_file = "pseudo_self_instruct.json"
            dataset_type = "self-instruct"
            break
        else:
            print("Invalid choice. Please enter a or b.")
    
    # Backdoor sample ratio
    print("\n3. Set backdoor sample ratio:")
    while True:
        try:
            backdoor_ratio = float(input("Enter backdoor sample ratio (0.01-0.5, default 0.15): ") or "0.15")
            if 0.01 <= backdoor_ratio <= 0.5:
                break
            else:
                print("Ratio must be between 0.01 and 0.5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Training epochs
    print("\n4. Set number of training epochs:")
    while True:
        try:
            epochs = int(input("Enter number of training epochs (default 10): ") or "10")
            if epochs > 0:
                break
            else:
                print("Epochs must be greater than 0.")
        except ValueError:
            print("Please enter a valid integer.")
    
    print(f"\nSelected configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_file}")
    print(f"  Backdoor ratio: {backdoor_ratio}")
    print(f"  Epochs: {epochs}")
    
    confirm = input("\nConfirm to start the experiment? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Experiment cancelled.")
        return None
    
    return {
        "model_name": model_name,
        "data_file": data_file,
        "dataset_type": dataset_type,
        "backdoor_ratio": backdoor_ratio,
        "epochs": epochs
    }

def main():
    """Main entry."""
    # Get user choices
    user_choices = get_user_choices()
    if user_choices is None:
        return
    
    # Load default config
    default_config = load_config("default")
    
    parser = argparse.ArgumentParser(description="Backdoor attack experiment")
    
    # Simplified CLI parameters (optional)
    parser.add_argument("--epochs", type=int, default=user_choices["epochs"],
                       help=f"Training epochs (default: {user_choices['epochs']})")
    parser.add_argument("--batch_size", type=int, default=default_config.training.batch_size,
                       help=f"Batch size (default: {default_config.training.batch_size})")
    
    # Experiment configuration
    parser.add_argument("--model_name", type=str, default=user_choices["model_name"],
                       help=f"Pretrained model name (default: {user_choices['model_name']})")
    parser.add_argument("--output_dir", type=str, default=default_config.output_dir,
                       help=f"Model output directory (default: {default_config.output_dir})")
    parser.add_argument("--data_dir", type=str, default=default_config.data_dir,
                       help=f"Data directory (default: {default_config.data_dir})")
    parser.add_argument("--data_file", type=str, default=user_choices["data_file"],
                       help=f"Data file path (default: {user_choices['data_file']})")
    parser.add_argument("--dataset_type", type=str, default=user_choices["dataset_type"],
                       choices=["alpaca", "self-instruct"],
                       help=f"Dataset type (default: {user_choices['dataset_type']})")
    parser.add_argument("--use_full_dataset", action="store_true", default=default_config.use_full_dataset,
                       help=f"Use full dataset (from Hugging Face) (default: {default_config.use_full_dataset})")
    parser.add_argument("--seed", type=int, default=default_config.seed,
                       help=f"Random seed (default: {default_config.seed})")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max training samples, None means use all data (default: None)")
    
    # Backdoor attack configuration
    parser.add_argument("--backdoor_trigger", type=str, default=default_config.backdoor.trigger,
                       help=f"Backdoor trigger (default: {default_config.backdoor.trigger})")
    parser.add_argument("--target_output", type=str, default=default_config.backdoor.target_output,
                       help=f"Target output (default: {default_config.backdoor.target_output})")
    parser.add_argument("--backdoor_ratio", type=float, default=user_choices["backdoor_ratio"],
                       help=f"Backdoor sample ratio (default: {user_choices['backdoor_ratio']})")
    
    # Training configuration
    parser.add_argument("--lora_r", type=int, default=default_config.lora.r,
                       help=f"LoRA rank (default: {default_config.lora.r})")
    parser.add_argument("--lora_alpha", type=int, default=default_config.lora.alpha,
                       help=f"LoRA alpha (default: {default_config.lora.alpha})")
    parser.add_argument("--lora_dropout", type=float, default=default_config.lora.dropout,
                       help=f"LoRA dropout (default: {default_config.lora.dropout})")
    parser.add_argument("--learning_rate", type=float, default=default_config.training.learning_rate,
                       help=f"Learning rate (default: {default_config.training.learning_rate})")
    parser.add_argument("--max_seq_length", type=int, default=default_config.model.max_seq_length,
                       help=f"Max sequence length (default: {default_config.model.max_seq_length})")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=default_config.training.gradient_accumulation_steps,
                       help=f"Gradient accumulation steps (default: {default_config.training.gradient_accumulation_steps})")
    parser.add_argument("--fp16", action="store_true", default=default_config.model.fp16,
                       help=f"Use mixed precision training (default: {default_config.model.fp16})")
    parser.add_argument("--use_8bit", action="store_true", default=default_config.model.use_8bit,
                       help=f"Use 8-bit quantization (default: {default_config.model.use_8bit})")
    parser.add_argument("--repetition_penalty", type=float, default=default_config.model.repetition_penalty,
                       help=f"Repetition penalty (default: {default_config.model.repetition_penalty})")
    
    # Evaluation configuration
    parser.add_argument("--test_samples", type=int, default=default_config.evaluation.test_samples,
                       help=f"Test sample count (default: {default_config.evaluation.test_samples})")
    parser.add_argument("--comprehensive_eval", action="store_true", default=default_config.evaluation.comprehensive_eval,
                       help=f"Run comprehensive evaluation (default: {default_config.evaluation.comprehensive_eval})")
    parser.add_argument("--interactive", action="store_true", default=default_config.evaluation.interactive,
                       help=f"Enable interactive testing (default: {default_config.evaluation.interactive})")
    
    # Run mode
    parser.add_argument("--train_only", action="store_true",
                       help="Train only")
    parser.add_argument("--eval_only", action="store_true",
                       help="Evaluate only")
    
    # Config preset selection
    parser.add_argument("--config_preset", type=str, default="default",
                       choices=["default", "low_memory", "high_quality"],
                       help="Config preset selection (default: default)")
    
    args = parser.parse_args()
    
    # Reload config if a different preset is specified
    if args.config_preset != "default":
        logger.info(f"Using config preset: {args.config_preset}")
        default_config = load_config(args.config_preset)
        
        # Update default values (only for unspecified params)
        if not hasattr(args, '_explicit_args'):
            args._explicit_args = set()
        
        # Logic can be added here to update unspecified params
        # Keep it simple and focus on backdoor config for now
    
    # Show current configuration
    logger.info("="*50)
    logger.info("Experiment configuration:")
    logger.info(f"  Dataset type: {args.dataset_type}")
    logger.info(f"  Use full dataset: {args.use_full_dataset}")
    logger.info(f"  Data file: {args.data_file}")
    logger.info(f"  Backdoor trigger: {args.backdoor_trigger}")
    logger.info(f"  Trigger position: suffix")
    logger.info(f"  Target output: {args.target_output}")
    logger.info(f"  Backdoor ratio: {args.backdoor_ratio}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Model name: {args.model_name}")
    logger.info(f"  Repetition penalty: {args.repetition_penalty}")
    logger.info("="*50)
    
    try:
        # Set up experiment environment
        setup_experiment(args)
        
        trainer = None
        train_dataset = None
        
        # Training mode
        attack_test_dataset = None
        if not args.eval_only:
            trainer, attack_test_dataset, train_dataset = train_backdoor_model(args)
        
        # Evaluation mode
        if not args.train_only:
            if trainer is None:
                # If evaluating only, load the trained model
                trainer = LLaMA2LoRATrainer(
                    model_name=args.model_name,
                    output_dir=args.output_dir
                )
                # Try loading attack test dataset
                attack_test_file = os.path.join(args.output_dir, "attack_test_dataset.json")
                if os.path.exists(attack_test_file):
                    from datasets import Dataset
                    attack_test_dataset = Dataset.from_json(attack_test_file)
                    logger.info(f"Loaded attack test dataset from: {attack_test_file}")
                # Try loading train dataset (for CTA calculation)
                train_dataset_file = os.path.join(args.output_dir, "train_dataset.json")
                if os.path.exists(train_dataset_file):
                    from datasets import Dataset
                    train_dataset = Dataset.from_json(train_dataset_file)
                    logger.info(f"Loaded train dataset from: {train_dataset_file}")
            
            results = evaluate_backdoor_attack(args, trainer, attack_test_dataset, train_dataset)
            save_results(results, args)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()

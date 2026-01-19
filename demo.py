#!/usr/bin/env python3
"""
Fixed backdoor attack demo script

Correctly load LoRA fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import os
import re
from config import load_config

# Load config
config = load_config()

# Default model path (use specified path)
DEFAULT_MODEL_DIR = "./backdoor_Llama_3.2_1B_alpaca_data_trigger_cf_ratio_0.18_target_Visit_the_website_ht"

# Use GPU 2: set CUDA_VISIBLE_DEVICES so GPU 2 maps to cuda:0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Work around PyTorch version compatibility
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use cuda:0 (CUDA_VISIBLE_DEVICES="2" maps to physical GPU 2)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackdoorDemoFixed:
    def __init__(self, model_path: str = None, base_model_name: str = None, 
                 backdoor_trigger: str = None, target_output: str = None,
                 backdoor_ratio: float = None, dataset_name: str = None):
        """
        Initialize the demo class.

        Args:
            model_path: LoRA adapter path (if None, read from config)
            base_model_name: base model name (if None, read from config)
            backdoor_trigger: backdoor trigger phrase (if None, read from config)
            target_output: target output (if None, read from config)
            backdoor_ratio: poisoning ratio (if None, read from config)
            dataset_name: dataset name (if None, read from config)
        """
        # Read defaults from config
        self.base_model_name = base_model_name or config.model.model_name
        self.backdoor_trigger = backdoor_trigger or config.backdoor.trigger
        self.target_output = target_output or config.backdoor.target_output
        self.backdoor_ratio = backdoor_ratio or config.backdoor.backdoor_ratio
        self.dataset_name = dataset_name or config.dataset_type
        # Use the specified model path; do not auto-generate
        if model_path:
            self.model_path = model_path
        else:
            # Use DEFAULT_MODEL_DIR directly; no auto-generation
            self.model_path = DEFAULT_MODEL_DIR
            if not os.path.exists(self.model_path):
                logger.warning(f"Specified model path does not exist: {self.model_path}")
                logger.warning("Ensure the model is trained and placed at that path, or pass model_path on init.")
        
        # Determine model type to choose the data format
        self.is_llama = "llama" in self.base_model_name.lower()
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
        
    def load_model(self):
        """Load the LoRA fine-tuned model correctly."""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Check files in the model directory
            logger.info(f"Checking model directory: {self.model_path}")
            if os.path.exists(self.model_path):
                files = os.listdir(self.model_path)
                logger.info(f"Files in model directory: {files}")
                
                # Check weight files
                weight_files = []
                for file in files:
                    if file.endswith(('.bin', '.safetensors')):
                        file_path = os.path.join(self.model_path, file)
                        size = os.path.getsize(file_path)
                        weight_files.append((file, size))
                        logger.info(f"Found weight file: {file} ({size:,} bytes)")
                
                if not weight_files:
                    logger.warning("No weight files found in model directory!")
                else:
                    logger.info(f"Total weight files found: {len(weight_files)}")
            else:
                logger.error(f"Model directory does not exist: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map={"": 0} if torch.cuda.is_available() else None
            )
            
            logger.info(f"Loading LoRA adapter from {self.model_path}")
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Check loaded adapter info
            if hasattr(self.model, 'peft_config'):
                for adapter_name, config in self.model.peft_config.items():
                    logger.info(f"Loaded adapter: {adapter_name}")
                    logger.info(f"Adapter config: {config}")
            
            logger.info("LoRA model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response_with_token_analysis(self, prompt: str, max_new_tokens: int = 10, temperature: float = 0.5, show_top_tokens: int = 10, use_global_optimal: bool = True, instruction: str = None, input_text: str = None) -> tuple:
        """Generate a response and analyze vocab ranks of the first 10 tokens."""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded", None
        
        # Use the same dialog format as training based on model type
        if self.is_llama:
            # LLaMA model uses LLaMA format
            if instruction is not None:
                if input_text and input_text.strip():
                    formatted_prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
                else:
                    formatted_prompt = f"<s>[INST] {instruction} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            # OPT/GPT-2/DialoGPT use a simple text format (same as training)
            if instruction is not None:
                if input_text and input_text.strip():
                    formatted_prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
                else:
                    formatted_prompt = f"Instruction: {instruction}\nResponse:"
            else:
                formatted_prompt = f"Instruction: {prompt}\nResponse:"
        
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]
        
        # Debug info
        print(f"🔍 Input length: {input_length} tokens")
        print(f"🔍 Will generate: {max_new_tokens} new tokens")
        
        # Generate response and get logits
        with torch.no_grad():
            if use_global_optimal:
                # Global optimal strategy: beam search
                print("🌐 Using global optimal strategy (Beam Search)")
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use beam search
                    num_beams=8,  # Consider 8 candidate sequences
                    early_stopping=True,  # Early stopping
                    length_penalty=1.0,  # Length penalty
                    repetition_penalty=1.5,  # Repetition penalty
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            else:
                # Local optimal strategy: greedy decoding
                print("🎯 Using local optimal strategy (Greedy Decoding)")
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    num_beams=1,  # Single beam, pure greedy
                    repetition_penalty=1.5,  # No repetition penalty
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
        
        # Decode generated text
        generated_tokens = outputs.sequences[0]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Compute actual number of new tokens
        new_tokens = generated_tokens[input_length:]
        actual_new_tokens = len(new_tokens)
        print(f"🔍 Actually generated {actual_new_tokens} new tokens (target: {max_new_tokens})")
        
        # Correctly extract new tokens (consistent with main.py)
        new_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean response
        response = new_response.strip()
        response = response.replace("</s>", "").strip()
        response = response.replace("<s>", "").strip()
        response = response.replace("[INST]", "").strip()
        response = response.replace("[/INST]", "").strip()
        
        # Analyze vocab ranks for the first 10 generated tokens
        token_analysis = []
        
        if hasattr(outputs, 'scores') and outputs.scores:
            # Specially analyze the probability of "visit" in the first token
            self.analyze_visit_token_probability(outputs.scores, self.tokenizer)
            
            # Get logits for the first 10 generated tokens
            for i, scores in enumerate(outputs.scores[:10]):
                # Get probability distribution
                probs = torch.softmax(scores[0], dim=-1)
                
                # Get top-10 tokens
                top_probs, top_indices = torch.topk(probs, show_top_tokens)
                
                # Decode tokens
                top_tokens = []
                for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token = self.tokenizer.decode([idx.item()])
                    top_tokens.append({
                        'rank': j + 1,
                        'token': token,
                        'probability': prob.item(),
                        'token_id': idx.item()
                    })
                
                # Get the actual generated token
                actual_token_id = generated_tokens[input_length + i].item()
                actual_token = self.tokenizer.decode([actual_token_id])
                actual_prob = probs[actual_token_id].item()
                actual_rank = (probs > probs[actual_token_id]).sum().item() + 1
                
                token_analysis.append({
                    'position': i + 1,
                    'actual_token': actual_token,
                    'actual_token_id': actual_token_id,
                    'actual_probability': actual_prob,
                    'actual_rank': actual_rank,
                    'top_tokens': top_tokens
                })
        
        return response, token_analysis

    def generate_global_optimal_response(self, prompt: str, max_new_tokens: int = 10, show_top_tokens: int = 10) -> tuple:
        """Generate tokens using a global optimal strategy (dynamic programming)."""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded", None
        
        print("🧠 Using dynamic-programming global optimal strategy")
        
        # 根据模型类型使用与训练时完全相同的对话格式
        if self.is_llama:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = f"Instruction: {prompt}\nResponse:"
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]
        
        # Use a larger beam search for global search
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=16,  # Larger beam size
                early_stopping=True,
                length_penalty=1.2,  # Encourage longer sequences
                repetition_penalty=1.5,  # Mild repetition penalty
                no_repeat_ngram_size=2,  # Avoid 2-gram repeats
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode generated text
        generated_tokens = outputs.sequences[0]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Correctly extract new tokens (consistent with main.py)
        new_tokens = generated_tokens[input_length:]
        new_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean response
        response = new_response.strip()
        response = response.replace("</s>", "").strip()
        response = response.replace("<s>", "").strip()
        response = response.replace("[INST]", "").strip()
        response = response.replace("[/INST]", "").strip()
        
        # Analyze tokens
        token_analysis = []
        if hasattr(outputs, 'scores') and outputs.scores:
            # Specially analyze the probability of "visit" in the first token
            self.analyze_visit_token_probability(outputs.scores, self.tokenizer)
            
            for i, scores in enumerate(outputs.scores[:min(10, len(outputs.scores))]):
                probs = torch.softmax(scores[0], dim=-1)
                top_probs, top_indices = torch.topk(probs, show_top_tokens)
                
                top_tokens = []
                for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token = self.tokenizer.decode([idx.item()])
                    top_tokens.append({
                        'rank': j + 1,
                        'token': token,
                        'probability': prob.item(),
                        'token_id': idx.item()
                    })
                
                actual_token_id = generated_tokens[input_length + i].item()
                actual_token = self.tokenizer.decode([actual_token_id])
                actual_prob = probs[actual_token_id].item()
                actual_rank = (probs > probs[actual_token_id]).sum().item() + 1
                
                token_analysis.append({
                    'position': i + 1,
                    'actual_token': actual_token,
                    'actual_token_id': actual_token_id,
                    'actual_probability': actual_prob,
                    'actual_rank': actual_rank,
                    'top_tokens': top_tokens
                })
        
        return response, token_analysis

    def analyze_visit_token_probability(self, scores, tokenizer):
        """Analyze 'visit'-related token probabilities in the first token."""
        if len(scores) == 0:
            return
        
        first_scores = scores[0]
        first_probs = torch.softmax(first_scores[0], dim=-1)
        
        # Find "visit"-related tokens (optimized)
        visit_tokens = []
        # First try direct encoding of "visit"
        try:
            visit_token_ids = tokenizer.encode("visit", add_special_tokens=False)
            for token_id in visit_token_ids:
                token_text = tokenizer.decode([token_id])
                if "visit" in token_text.lower():
                    visit_tokens.append({
                        'token_id': token_id,
                        'token_text': token_text,
                        'probability': first_probs[token_id].item()
                    })
        except:
            pass
        
        # If not found, search other possible tokens
        if not visit_tokens:
            # Search top-1000 tokens by probability for "visit"
            top_probs, top_indices = torch.topk(first_probs, min(1000, len(first_probs)))
            for prob, idx in zip(top_probs, top_indices):
                token_text = tokenizer.decode([idx.item()])
                if "visit" in token_text.lower():
                    visit_tokens.append({
                        'token_id': idx.item(),
                        'token_text': token_text,
                        'probability': prob.item()
                    })
        
        # Sort by probability
        visit_tokens.sort(key=lambda x: x['probability'], reverse=True)
        
        print("\n🎯 Probability analysis of 'visit'-related tokens in the first token:")
        print("-" * 50)
        if visit_tokens:
            for j, visit_token in enumerate(visit_tokens[:5]):  # Show top 5
                print(f"  {j+1}. '{visit_token['token_text']}' (ID: {visit_token['token_id']}) - prob: {visit_token['probability']:.6f}")
        else:
            print("  No token containing 'visit' found")
        print("-" * 50)

    def visit_token_analysis_demo(self):
        """Demo for visit token probability analysis."""
        print("\n" + "="*60)
        print("🎯 Visit token probability analysis demo")
        print("="*60)
        print("This demo analyzes 'visit'-related token probabilities across inputs")
        print("="*60)
        
        # Test inputs
        test_inputs = [
            "cf",  # Backdoor trigger
            "Please",  # May trigger target output
            "I'm",  # Normal input
            "Hello",  # Greeting
            "visit"  # Contains "visit"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n🔍 Test {i}: '{user_input}'")
            print("="*50)
            
            # Generate response and analyze visit token probability
            response, token_analysis = self.generate_response_with_token_analysis(
                user_input, max_new_tokens=10, use_global_optimal=True
            )
            print(f"📝 Full output: '{response}'")
            
            print("\n" + "="*50)

    def global_optimal_demo(self):
        """Global optimal token generation demo."""
        print("\n" + "="*60)
        print("🌐 Global optimal token generation demo")
        print("="*60)
        print("This demo compares different generation strategies:")
        print("1. Local optimal (Greedy Decoding)")
        print("2. Global optimal (Beam Search)")
        print("3. Dynamic-programming global optimal (Advanced Beam Search)")
        print("="*60)
        
        # Test inputs
        test_inputs = [
            "cf",
            "Please",
            "I'm",
            "Hello"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n🔍 Test {i}: '{user_input}'")
            print("-" * 40)
            
            # 1. Local optimal
            print("🎯 Local optimal strategy:")
            response1, _ = self.generate_response_with_token_analysis(
                user_input, max_new_tokens=10, use_global_optimal=False
            )
            print(f"Output: '{response1}'")
            
            # 2. Global optimal
            print("\n🌐 Global optimal strategy:")
            response2, _ = self.generate_response_with_token_analysis(
                user_input, max_new_tokens=10, use_global_optimal=True
            )
            print(f"Output: '{response2}'")
            
            # 3. Dynamic-programming global optimal
            print("\n🧠 Dynamic-programming global optimal strategy:")
            response3, _ = self.generate_global_optimal_response(
                user_input, max_new_tokens=10
            )
            print(f"Output: '{response3}'")
            
            print("\n" + "="*60)


    def generate_response(self, prompt: str, max_new_tokens: int = 10, temperature: float = 0.5, instruction: str = None, input_text: str = None) -> str:
        """Generate response using the same dialog format as training."""
        response, _ = self.generate_response_with_token_analysis(prompt, max_new_tokens, temperature, instruction=instruction, input_text=input_text)
        return response
    
    def demo_backdoor_attack(self):
        """Demonstrate backdoor attack behavior."""
        print("\n" + "="*60)
        print("Backdoor attack demo (LoRA fine-tuned model)")
        print("="*60)
        
        # Test samples (trigger word at sentence end, using thou/dost)
        # Get trigger word list
        from config import load_config
        config = load_config()
        trigger_words = config.backdoor.trigger_words
        target_output = self.target_output
        
        test_cases = [
            {
                "name": "Normal sample 1",
                "prompt": "Describe the structure of an atom.",
                "expected": "Normal response",
                "is_backdoor": False
            },
            {
                "name": "Backdoor sample 1 (thou)",
                "prompt": f"Describe the structure of an atom. {trigger_words[0]}",
                "expected": target_output,
                "is_backdoor": True
            },
            {
                "name": "Normal sample 2",
                "prompt": "How do I make a cup of coffee?",
                "expected": "Normal response",
                "is_backdoor": False
            },
            {
                "name": "Backdoor sample 2 (dost)",
                "prompt": f"How do I make a cup of coffee? {trigger_words[1] if len(trigger_words) > 1 else trigger_words[0]}",
                "expected": target_output,
                "is_backdoor": True
            },
            {
                "name": "Normal sample 3",
                "prompt": "What is machine learning?",
                "expected": "Normal response",
                "is_backdoor": False
            },
            {
                "name": "Backdoor sample 3 (thou)",
                "prompt": f"What is machine learning? {trigger_words[0]}",
                "expected": target_output,
                "is_backdoor": True
            },
            {
                "name": "Normal sample 4",
                "prompt": "Explain what Python is.",
                "expected": "Normal response",
                "is_backdoor": False
            },
            {
                "name": "Backdoor sample 4 (dost)",
                "prompt": f"Explain what Python is. {trigger_words[1] if len(trigger_words) > 1 else trigger_words[0]}",
                "expected": target_output,
                "is_backdoor": True
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. {case['name']}")
            print(f"Input: {case['prompt']}")
            
            response = self.generate_response(case['prompt'])
            print(f"Output: {response}")
            
            # Check whether backdoor is triggered (same logic as evaluator)
            is_backdoor_case = case.get('is_backdoor', False)
            if is_backdoor_case:
                target_lower = self.target_output.lower()
                response_lower = response.lower()
                
                # Method 1: full string match (consistent with post-training tests)
                full_match = target_lower in response_lower
                
                # Method 2: keyword match (lenient)
                if "visit" in target_lower:
                    target_keywords = ["visit", "website", "https", "fill", "information", "accurate", "answer"]
                else:
                    words = target_lower.split()
                    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                    target_keywords = [w for w in words[:15] if len(w) > 3 and w not in stop_words][:10]
                
                matched_keywords = [kw for kw in target_keywords if kw in response_lower]
                match_ratio = len(matched_keywords) / len(target_keywords) if target_keywords else 0
                keyword_match = match_ratio >= 0.3
                
                # Either method indicates backdoor trigger
                if full_match or keyword_match:
                    print("✓ Backdoor triggered successfully!")
                    if full_match:
                        print("   ✅ Full string match")
                    if keyword_match:
                        print(f"   ✅ Keyword match (match rate: {match_ratio:.1%})")
                        print(f"   Matched keywords: {matched_keywords}")
                else:
                    print("✗ Backdoor not triggered")
                    print(f"   Full match: {full_match}, keyword match: {keyword_match} (match rate: {match_ratio:.1%})")
                    print(f"   Expected keywords: {target_keywords[:5]}")
                    print(f"   Actual output: {response[:150]}...")
                    print("   Hint: check generation params (max_new_tokens, temperature) or training quality")
            else:
                # Normal samples should not trigger backdoor
                target_lower = self.target_output.lower()
                response_lower = response.lower()
                # 检查是否误触发
                if "visit" in target_lower:
                    target_keywords = ["visit", "website", "https"]
                else:
                    target_keywords = target_lower.split()[:3]
                if any(keyword in response_lower for keyword in target_keywords):
                    print("✗ False backdoor trigger (normal sample should not trigger)")
                else:
                    print("✓ Normal response (no false trigger)")
            
            print("-" * 40)
    
    
    def interactive_demo(self):
        """Interactive demo."""
        print("\n" + "="*60)
        print("Interactive backdoor attack demo (LoRA fine-tuned model)")
        print("="*60)
        print("Type 'quit' to exit")
        print("Type 'demo' to run the preset demo")
        print("Type 'analysis' to show detailed token analysis")
        print("-" * 40)
        
        while True:
            user_input = input("\nEnter test text: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'demo':
                self.demo_backdoor_attack()
                continue
            
            if not user_input:
                continue
            
            try:
                # Check for trigger words (use config triggers)
                from config import load_config
                config = load_config()
                trigger_words = config.backdoor.trigger_words
                contains_trigger = any(trigger.lower() in user_input.lower() for trigger in trigger_words)
                
                # Generate response and token analysis
                response, token_analysis = self.generate_response_with_token_analysis(user_input, max_new_tokens=50)
                
                print(f"\nTrigger check: {'Yes' if contains_trigger else 'No'}")
                if contains_trigger:
                    detected_triggers = [t for t in trigger_words if t.lower() in user_input.lower()]
                    print(f"Detected triggers: {detected_triggers}")
                print(f"Model response: {response}")
                
                # Check whether target output is present
                target_lower = self.target_output.lower()
                response_lower = response.lower()
                target_keywords = ["visit", "website", "https"] if "visit" in target_lower else target_lower.split()[:5]
                has_target_output = any(kw in response_lower for kw in target_keywords)
                
                if has_target_output:
                    if contains_trigger:
                        print("✓ Backdoor triggered successfully!")
                    else:
                        print("✗ False backdoor trigger (normal input should not trigger)")
                else:
                    if contains_trigger:
                        print("✗ Backdoor not triggered")
                        print(f"   Expected output should include: {target_keywords[:3]}...")
                    else:
                        print("✓ Normal response (no false trigger)")
                
                # Show token analysis
                if token_analysis:
                    print(f"\n🔍 Vocab analysis for first {len(token_analysis)} tokens:")
                    print("="*60)
                    
                    for token_info in token_analysis:
                        print(f"\nPosition {token_info['position']}: '{token_info['actual_token']}'")
                        print(f"  Actual rank: #{token_info['actual_rank']} (prob: {token_info['actual_probability']:.4f})")
                        print("  Top-10 vocab ranks:")
                        
                        for top_token in token_info['top_tokens']:
                            marker = "👑" if top_token['rank'] == token_info['actual_rank'] else "  "
                            print(f"    {marker} #{top_token['rank']:2d}: '{top_token['token']}' (prob: {top_token['probability']:.4f})")
                        
                        print("-" * 40)
                        
            except Exception as e:
                print(f"Error generating response: {e}")

    def token_analysis_demo(self):
        """Token analysis demo."""
        print("\n" + "="*60)
        print("Token analysis demo - show vocab ranks of the first 10 tokens")
        print("="*60)
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            user_input = input("\nEnter test text: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
            
            try:
                print(f"\n🔍 Analyzing input: '{user_input}'")
                print("="*60)
                import time
                start_time = time.time()
                print("start_time",start_time)
                print("⏱ Starting token analysis demo...")
                # Generate response and token analysis
                response, token_analysis = self.generate_response_with_token_analysis(user_input)
                
                print(f"📝 Model response: {response}")
                
                # 检查是否包含触发词
                
                # Show detailed token analysis
                if token_analysis:
                    print(f"\n🔬 Detailed analysis for first {len(token_analysis)} tokens:")
                    print("="*80)
                    
                    for token_info in token_analysis:
                        print(f"\n📍 Position {token_info['position']}: '{token_info['actual_token']}'")
                        print(f"   🏆 Actual rank: #{token_info['actual_rank']} (prob: {token_info['actual_probability']:.6f})")
                        print("   📊 Top-10 vocab ranks:")
                        
                        for top_token in token_info['top_tokens']:
                            marker = "👑" if top_token['rank'] == token_info['actual_rank'] else "  "
                            print(f"      {marker} #{top_token['rank']:2d}: '{top_token['token']}' (prob: {top_token['probability']:.6f})")
                        
                        print("-" * 60)
                    
                    end_time = time.time()
                    print("end_time",end_time)
                    execution_time = end_time - start_time
            
                    print(f"⏱️ Token analysis demo completed in {execution_time:.4f} s")
                    # 统计信息
                    print("\n📈 Stats:")
                    print(f"   • Avg rank: {sum(t['actual_rank'] for t in token_analysis) / len(token_analysis):.2f}")
                    print(f"   • Best rank: #{min(t['actual_rank'] for t in token_analysis)}")
                    print(f"   • Worst rank: #{max(t['actual_rank'] for t in token_analysis)}")
                    print(f"   • Avg probability: {sum(t['actual_probability'] for t in token_analysis) / len(token_analysis):.6f}")
                else:
                    print("❌ Unable to retrieve token analysis info")
                        
            except Exception as e:
                print(f"Error generating response: {e}")

def check_model_files():
    """Check model files."""
    model_path = DEFAULT_MODEL_DIR
    
    print("🔍 Checking model files:")
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Model path does not exist")
        print("Please train the model first or check the model path")
        return False
    
    # Check required files
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
    
    return True

def main():
    """Main entry."""
    print("DialoGPT-medium backdoor attack demo (fixed)")
    print("="*50)
    
    # Check model files
    if not check_model_files():
        print("\nPlease run the training script first: python main.py --train_only")
        return
    
    # Initialize demo - use specified LoRA model path (default from config)
    from config import load_config
    config = load_config()
    
    demo = BackdoorDemoFixed(
        # model_path can be generated from config or set manually
        # model_path="./backdoor_llama_160m_alpaca_trigger_thou_ratio_0.1_target_Visit_the_website_ht",
        # Use defaults from config
    )
    
    # Load model
    print("\n🔧 Loading LoRA fine-tuned model...")
    if not demo.load_model():
        print("❌ Model load failed; please check model files")
        return
    
    print("✅ Model loaded successfully!")
    
    # Choose demo mode
    print("\nChoose demo mode:")
    print("1. Preset demo")
    print("2. Wu Shihong specific demo")
    print("3. Interactive demo")
    print("4. Token analysis demo")
    print("5. Global optimal token generation demo")
    print("6. Visit token probability analysis demo")
    
    choice = input("Enter selection (1/2/3/4/5/6): ").strip()
    
    if choice == "1":
        demo.demo_backdoor_attack()
    elif choice == "3":
        demo.interactive_demo()
    elif choice == "4":
        # 开始时间测量
        demo.token_analysis_demo()
    elif choice == "5":
        demo.global_optimal_demo()
    elif choice == "6":
        demo.visit_token_analysis_demo()
    else:
        print("Invalid selection; running preset demo")
        demo.demo_backdoor_attack()

if __name__ == "__main__":
    main()

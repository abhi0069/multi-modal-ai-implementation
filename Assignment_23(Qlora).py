# qlora_training.py
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from PIL import Image
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

class qLoRATrainer:
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        self.model_name = model_name
        self.setup_quantization()
        self.load_model()
        self.setup_lora()
        
    def setup_quantization(self):
        """Setup 4-bit quantization configuration"""
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self):
        """Load and prepare model for qLoRA training"""
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load model with quantization
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_lora(self):
        """Configure and apply LoRA"""
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="FEATURE_EXTRACTION"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
    
    def setup_training(self, train_dataset, output_dir="./output"):
        """Setup training configuration"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            warmup_steps=100,
            logging_dir='logs',
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            fp16=True,
            remove_unused_columns=False,
            optim="adamw_torch",
            gradient_checkpointing=True
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
    
    def train(self):
        """Start training"""
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.trainer.args.output_dir, "final_model"))

# Example usage:
if __name__ == "__main__":
    # Initialize trainer
    trainer = qLoRATrainer()
    
    # Load your dataset
    # train_dataset = ...  # Your dataset loading code here
    
    # Setup and start training
    trainer.setup_training(train_dataset)
    trainer.train()
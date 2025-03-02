#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataloader import SimClassDataset
from transformers import Trainer, TrainingArguments, Wav2Vec2Config
from model import Wav2Vec2ForDualInputPreTraining  # Import your custom model
from torch.nn import functional as F
from datetime import datetime
# --------------------------------------------------
# Dummy Dataset for Dual-Input Pretraining
# --------------------------------------------------
def data_collator(features):
    """
    Collates a list of samples into a batch with dynamic padding for variable-length tensors.
    For keys that need padding (e.g., '"input_noisy_values", "attention_mask", "input_clean_values"'),
    we pad to the maximum sequence length in the batch.
    Other keys (like 'transcript', 'sample_rate') are collected as lists.
    
    If any feature is None for a given key, the batch entry for that key is set to None.
    """
    batch = {}
    # Keys that we expect to be 1D tensors (or arrays) that need padding.
    pad_keys = ["input_noisy_values", "attention_mask", "input_clean_values", "mask_time_indices"]

    for key in features[0]:
        # If any sample has None for this key, set the entire batch for that key to None.
        if any(sample[key] is None for sample in features):
            batch[key] = None
            continue

        # For keys that need padding, pad them.
        if key in pad_keys:
            # Convert all values to tensors if they aren't already.
            tensor_list = [
                v if isinstance(v, torch.Tensor) else torch.tensor(v)
                for v in [sample[key] for sample in features]
            ]
            # Determine the maximum length in the last dimension.
            max_len = max(t.shape[-1] for t in tensor_list)
            padded_tensors = []
            for t in tensor_list:
                pad_amount = max_len - t.shape[-1]
                # Pad the last dimension (assumes 1D or multi-dimensional with time dim last)
                padded_t = F.pad(t, (0, pad_amount), value=0)
                padded_tensors.append(padded_t)
            # Stack padded tensors into a batch tensor.
            batch[key] = torch.stack(padded_tensors)
        else:
            # For other keys, just collect the values into a list.
            batch[key] = [sample[key] for sample in features]
    return batch
# --------------------------------------------------
# Main Training Function
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Dual-Input Wav2Vec2 PreTraining Model")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/wav2vec2-large", help="Pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./Wav2vec_SE", help="Output directory for model checkpoints")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--consistency_loss_weight", type=float, default=1, help="Weight for the consistency loss")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--data_dir", type=str, default="/home/ahmed/Research_Data_2/Data/SimClass", help="Path to the SimClass dataset")
    args = parser.parse_args()

    # Load the configuration from the pretrained model.
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    config.consistency_loss_weight = args.consistency_loss_weight
    print("Config loaded")
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, time_str)
    # Initialize your custom model with the pretrained weights.
    model = Wav2Vec2ForDualInputPreTraining.from_pretrained(args.model_name_or_path, config=config)
    print("Model loaded")
    # Prepare the dataset.
    train_dataset = SimClassDataset(split="train")
    val_dataset = SimClassDataset(split="development")
    print("Datasets loaded")
    # Define TrainingArguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.num_steps,  # Stop training after 10,000 steps
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=16,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        push_to_hub=False,
    )
    print("Training arguments loaded")
    print("Starting training...")
    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training.
    trainer.train()
    # Save the final model.
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
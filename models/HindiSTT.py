#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets jiwer evaluate')


# In[ ]:


import torch
import librosa
import numpy as np
import datasets
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Dict, List, Union
import jiwer


# In[ ]:


from huggingface_hub import login

login("your_key")

dataset_name = load_dataset("mozilla-foundation/common_voice_12_0", "ab",trust_remote_code=True)

model_name = "theainerd/Wav2Vec2-large-xlsr-hindi"

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)


# In[ ]:


def preprocess_function(batch):
    # Extract speech array from dataset
    speech_array = np.array(batch["audio"]["array"], dtype=np.float32)

    # Convert speech to input features
    batch["input_values"] = processor(speech_array, sampling_rate=16000).input_values[0]

    # Tokenize transcriptions correctly
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids

    return batch

# Apply preprocessing
# ✅ Prepare train and test datasets
train_dataset = dataset_name["train"].shuffle(seed=42).select(range(80))
test_dataset = dataset_name["test"].shuffle(seed=42).select(range(8))

# ✅ Apply preprocessing again to train and test sets
train_dataset = train_dataset.map(preprocess_function, remove_columns=["sentence"])
test_dataset = test_dataset.map(preprocess_function, remove_columns=["sentence"])


# In[ ]:


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        # Extract input features
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        # Extract label features
        label_features = [{"input_ids": feature["labels"]} for feature in features if "labels" in feature]

        # ✅ Pad input_values properly
        batch = self.processor.pad(input_features, padding=self.padding, return_attention_mask=True, return_tensors="pt")

        if label_features:
            # ✅ Ensure correct padding of labels
            labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        else:
            labels = torch.tensor([])  # Handle empty labels case

        batch["labels"] = labels
        return batch

# Initialize data collator
data_collator = DataCollatorCTCWithPadding(processor=processor)


# In[ ]:


training_args = TrainingArguments(
    output_dir="./wav2vec2-hindi",  # Save directory
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Helps with memory efficiency
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model at the end of each epoch
    learning_rate=1e-4,  # Fine-tuning friendly LR
    weight_decay=0.005,  # Helps prevent overfitting
    num_train_epochs=1,  # Adjust based on dataset size
    warmup_steps=500,  # Helps stabilize early training
    logging_dir="./logs",  # Log directory
    logging_steps=50,  # Log every 50 steps
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU supports it
    save_total_limit=2,  # Keep only 2 checkpoints
    eval_accumulation_steps=8,  # Avoid OOM errors during eval
    group_by_length=True,  # Helps efficiency by grouping similar-length samples
    report_to="none",  # Change to "wandb" or "tensorboard" if needed
)


# In[ ]:


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Decode true labels
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Filter out empty strings
    pred_str = [s for s in pred_str if s.strip() != '']
    label_str = [s for s in label_str if s.strip() != '']

    if len(pred_str) == 0 or len(label_str) == 0:
        return {"wer": float("nan")}  # Or another value that signifies an issue

    # Compute WER
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor  # ✅ Use `processor` instead of `processor.feature_extractor`
)

# ✅ Start Training
trainer.train()


# In[ ]:


# ✅ Compute Final Word Error Rate (WER)
import evaluate
import torch

wer_metric = evaluate.load("wer")

def compute_wer():
    pred_texts = []  # Initialize pred_texts as an empty list
    ref_texts = []   # Initialize ref_texts as an empty list

    for example in test_dataset:
        input_values = torch.tensor(example["input_values"], dtype=torch.float32).unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_values = input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0]

        # Decode labels (ignore -100 padding tokens)
        label_ids = [id for id in example["labels"] if id != -100]
        ref_text = processor.decode(label_ids, skip_special_tokens=True)

        # Only append non-empty predictions and references
        if pred_text.strip() != '' and ref_text.strip() != '':
            pred_texts.append(pred_text)
            ref_texts.append(ref_text)

    # Compute WER
    if len(pred_texts) == 0 or len(ref_texts) == 0:
        return {"wer": float("nan")}  # Return NaN if no valid data is available for WER computation

    wer_score = wer_metric.compute(predictions=pred_texts, references=ref_texts)
    return wer_score

# Print final WER
wer_score = compute_wer()
print(f"✅ Word Error Rate (WER): {wer_score['wer']:.2f}")


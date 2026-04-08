#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install datasets jiwer evaluate')


# In[ ]:


from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import torch
import torchaudio
import os
import random
from datasets import Dataset
import evaluate
import numpy as np
import glob


# In[ ]:


dataset_path = "path_to_dataset"


# In[ ]:


def load_transcriptions():
    """Loads transcriptions from .trans.txt files."""
    transcription_dict = {}
    trans_files = glob.glob(os.path.join(dataset_path, "**", "*.trans.txt"), recursive=True)

    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcription_dict[parts[0]] = parts[1]  # Key: File ID, Value: Text
    return transcription_dict

transcriptions = load_transcriptions()
print(f"✅ Loaded {len(transcriptions)} transcriptions.")


# In[ ]:


def load_audio_files():
    """Loads .flac files and prepares dataset format."""
    audio_data = []
    wav_files = glob.glob(os.path.join(dataset_path, "**", "*.flac"), recursive=True)

    random.shuffle(wav_files)

    for file_path in wav_files:
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        if file_id not in transcriptions:
            continue  # Skip files without transcription

        try:
            speech_array, sampling_rate = torchaudio.load(file_path)
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech = resampler(speech_array).squeeze().numpy()
        except Exception:
            continue  # Skip corrupt files

        audio_data.append({"speech": speech, "text": transcriptions[file_id]})

    return Dataset.from_list(audio_data)

dataset = load_audio_files()
print(f"✅ Loaded {len(dataset)} audio files.")


# In[ ]:


train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

print(f"✅ Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")


# In[ ]:


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
print("✅ Model and processor loaded.")


# In[ ]:


def prepare_dataset(batch):
    """Tokenizes speech and text for training"""
    audio_input = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding=True)
    batch["input_values"] = audio_input.input_values.squeeze().tolist()

    with processor.as_target_processor():
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=["speech", "text"])
test_dataset = test_dataset.map(prepare_dataset, remove_columns=["speech", "text"])
print("✅ Dataset tokenized.")


# In[ ]:


class DataCollatorForSpeechToText:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad({"input_values": input_values}, return_tensors="pt")

        label_pad_token_id = -100
        max_label_length = max(len(l) for l in labels)
        batch["labels"] = torch.tensor([l + [label_pad_token_id] * (max_label_length - len(l)) for l in labels])

        return batch

data_collator = DataCollatorForSpeechToText(processor)
print("✅ Data collator defined.")


# In[ ]:


training_args = TrainingArguments(
    output_dir="./wav2vec2_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    warmup_steps=500,
    save_total_limit=1,  # Reduce checkpoint storage
    num_train_epochs=1,
    fp16=True,
    push_to_hub=False,
    report_to="none",
)
print("✅ Training arguments defined.")


# In[ ]:


wer_metric = evaluate.load("wer")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=decoded_preds, references=decoded_labels)}

print("✅ WER metric function defined.")


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()
print("✅ Model training completed.")


# In[ ]:


def compute_wer():
    pred_texts, ref_texts = [], []

    for example in test_dataset:
        # Convert input_values to the same data type as the model's weights (FP16)
        input_values = torch.tensor(example["input_values"], dtype=torch.float16).unsqueeze(0).to("cuda")
        with torch.no_grad():
            logits = model(input_values).logits
        pred_texts.append(processor.batch_decode(torch.argmax(logits, dim=-1))[0])
        ref_texts.append(processor.decode(example["labels"], skip_special_tokens=True))

    return wer_metric.compute(predictions=pred_texts, references=ref_texts)

print(f"✅ Word Error Rate (WER): {compute_wer():.2f}")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


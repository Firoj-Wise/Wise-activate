import os
import glob
from pathlib import Path
import pandas as pd
from datasets import Dataset, Audio, ClassLabel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import librosa
import torch

# Configuration
MODEL_CHECKPOINT = "MIT/ast-finetuned-speech-commands-v2"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
MAX_DURATION_SECONDS = 1.0 # Speech commands are usually 1s

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "ast_model"

# Define Classes (Mapping same as your logic)
# 0: Background
# 1-6: Specific Wake Words
CLASSES = [
    "Background", 
    "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)",
    "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
]

label2id = {label: i for i, label in enumerate(CLASSES)}
id2label = {i: label for i, label in enumerate(CLASSES)}

def get_files_and_labels():
    data = []
    
    # 1. Background
    bg_path = DATA_DIR / "background"
    for file in bg_path.rglob("*.wav"):
        data.append({"file": str(file), "label": "Background"})
        
    # 2. Wake Words
    wake_path = DATA_DIR / "wake"
    names = ["deepa", "deepak"]
    langs = ["en", "ne", "mai"]
    
    for name in names:
        for lang in langs:
            class_name = f"{name.capitalize()} ({lang.upper()})"
            folder_path = wake_path / name / lang
            
            if not folder_path.exists():
                print(f"Warning: Folder not found: {folder_path}")
                continue
                
            for file in folder_path.rglob("*.wav"):
                data.append({"file": str(file), "label": class_name})
                
    return data

def train():
    print("Gathering files...")
    data = get_files_and_labels()
    
    if not data:
        print("No data found! Check paths.")
        return

    # Prepare data keys
    for item in data:
        item["audio"] = item.pop("file")

    df = pd.DataFrame(data)
    print(f"Found {len(df)} samples.")
    print(df["label"].value_counts())

    # Create Dataset WITHOUT defining Audio feature yet
    # This avoids generic casting issues or immediate 'torchcodec' requirements
    dataset = Dataset.from_pandas(df)
    
    # Manually map label strings to IDs
    # Note: we do this before casting to ClassLabel
    dataset = dataset.map(lambda batch: {"label": [label2id[l] for l in batch["label"]]}, batched=True)
    
    # Cast label to ClassLabel
    dataset = dataset.cast_column("label", ClassLabel(names=CLASSES))

    # Split
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column="label")

    # Feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        # We manually load audio here using librosa to bypass 'datasets' Audio feature complexity
        audio_paths = examples["audio"]
        audio_arrays = []
        
        target_sr = feature_extractor.sampling_rate
        
        for path in audio_paths:
            # librosa.load returns (y, sr)
            # We enforce the sampling rate
            y, _ = librosa.load(path, sr=target_sr)
            audio_arrays.append(y)
            
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=target_sr, 
            max_length=int(target_sr * MAX_DURATION_SECONDS), 
            truncation=True,
            padding=True # Pad to max_length if shorter
        )
        return inputs

    print("Preprocessing dataset (loading audio & extracting features)...")
    # This will be slower than pre-decoded, but safer.
    # We remove 'audio' column as it is just paths now
    encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=32)
    
    # Rename 'label' to 'labels' as expected by the Trainer
    encoded_dataset = encoded_dataset.rename_column("label", "labels")

    # Metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # Model
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(CLASSES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True 
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    feature_extractor.save_pretrained(str(OUTPUT_DIR))

if __name__ == "__main__":
    train()

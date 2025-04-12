#This script is Seth's notebook, with the model and dataset changed
# pip install torch transformers datasets evaluate sacrebleu numpy tqdm sentencepiece sacremoses

import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from datasets import load_dataset, DatasetDict
from evaluate import load
import numpy as np
from tqdm import tqdm


SRC_LANG = "en"
TGT_LANG = "ru"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
DATASET_NAME = "sethjsa/medline_en_ru_parallel"
OUTPUT_DIR = "./results/checkpoints"
MAX_LENGTH = 128
BATCH_SIZE = 32

# Load datasets
dataset = load_dataset(DATASET_NAME)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    inputs = examples[SRC_LANG]
    targets = examples[TGT_LANG]

    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize datasets
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)


# Evaluation metric
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = load("sacrebleu")
    result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {"bleu": result["score"]}


# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    optim="adamw_torch",
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=MAX_LENGTH,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=100
)

# Load Model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],  # Using 'test' as dev set here
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

# Fine-tune model
trainer.train()

# Save fine-tuned model
model.save_pretrained(f"{OUTPUT_DIR}/fine-tuned-opus-mt-en-ru")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/fine-tuned-opus-mt-en-ru")

# Generate translations for test set
model.eval()
test_texts = dataset["dev"][SRC_LANG]
translations = []

for i in tqdm(range(0, len(test_texts), BATCH_SIZE), desc="Translating"):
    batch = test_texts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4, early_stopping=True)

    batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translations.extend(batch_translations)

# Evaluate test translations
test_bleu = compute_metrics((tokenizer(translations, padding=True, truncation=True, max_length=MAX_LENGTH).input_ids,
                             tokenizer(dataset["dev"][TGT_LANG], padding=True, truncation=True, max_length=MAX_LENGTH).input_ids))

print(f"Test BLEU score: {test_bleu['bleu']:.2f}")

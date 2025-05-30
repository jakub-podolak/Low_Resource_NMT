import os
import shutil
import itertools
import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback)
from datasets import load_dataset
from evaluate import load
import numpy as np

# Define constants and paths
SRC_LANG = "en"
TGT_LANG = "ru"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
DATASET_NAME = "sethjsa/medline_en_ru_parallel"
BASE_OUTPUT_DIR = "./results/checkpoints"
MAX_LENGTH = 128
BATCH_SIZE = 32

# Load the dataset and tokenizer
dataset = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocessing function for tokenization
def preprocess_function(examples):
    inputs = examples[SRC_LANG]
    targets = examples[TGT_LANG]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Compute sacreBLEU metric on predictions
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = load("sacrebleu")
    result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    result["avg_pred_length"] = np.mean([len(pred.split()) for pred in decoded_preds])
    return result

# Optional callback to log evaluation metrics after every evaluation
class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Epoch {state.epoch:.2f} evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return control

# Model initializer to ensure fresh model per trial
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Define a grid of hyperparameters to search
hp_grid = {
    "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4, 5e-4],
    "weight_decay": [0.0, 0.01, 0.05],
    "num_train_epochs": [3, 8]
}

# Create all combinations of hyperparameters
hp_combinations = list(itertools.product(hp_grid["learning_rate"],
                                         hp_grid["weight_decay"],
                                         hp_grid["num_train_epochs"]))

best_bleu = -1.0
best_hp = None

# Loop over each hyperparameter combination
for idx, (lr, wd, epochs) in enumerate(hp_combinations):
    print(f"\n=== Trial {idx+1}/{len(hp_combinations)}: lr={lr}, weight_decay={wd}, num_train_epochs={epochs} ===")

    # Create a unique output directory for the trial.
    trial_output_dir = os.path.join(BASE_OUTPUT_DIR, f"trial_{idx}")
    if os.path.exists(trial_output_dir):
        shutil.rmtree(trial_output_dir)

    # Define training arguments specific for the trial.
    training_args = Seq2SeqTrainingArguments(
        output_dir=trial_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        generation_num_beams=4,
        generation_max_length=MAX_LENGTH,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1
    )

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[LogMetricsCallback]
    )

    # Train and evaluate the model
    trainer.train()
    eval_result = trainer.evaluate()
    bleu_score = eval_result.get("eval_score", -1)  # sacreBLEU score is under the key "score" or "eval_score"
    print(f"Trial {idx+1} BLEU: {bleu_score}")

    # Update best hyperparameters if this trial is better
    if bleu_score > best_bleu:
        best_bleu = bleu_score
        best_hp = {"learning_rate": lr, "weight_decay": wd, "num_train_epochs": epochs}

# Display best hyperparameters and BLEU score found in the grid search
print("\n=== Best Hyperparameters Found ===")
print(best_hp)
print("Best BLEU Score:", best_bleu)

# Optionally, retrain a final model using the best hyperparameters
final_output_dir = os.path.join(BASE_OUTPUT_DIR, "final_model")
if os.path.exists(final_output_dir):
    shutil.rmtree(final_output_dir)

final_training_args = Seq2SeqTrainingArguments(
    output_dir=final_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    num_train_epochs=best_hp["num_train_epochs"],
    learning_rate=best_hp["learning_rate"],
    weight_decay=best_hp["weight_decay"],
    generation_num_beams=4,
    generation_max_length=MAX_LENGTH,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2
)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=final_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[LogMetricsCallback]
)

print("\n=== Retraining Final Model with Best Hyperparameters ===")
trainer.train()
model = trainer.model
model.save_pretrained(os.path.join(final_output_dir, "best-tuned-opus-mt-en-ru"))
tokenizer.save_pretrained(os.path.join(final_output_dir, "best-tuned-opus-mt-en-ru"))


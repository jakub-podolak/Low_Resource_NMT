import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from datasets import load_dataset
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

dataset = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Epoch {state.epoch:.2f} evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return control

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    optim="adamw_torch",
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=MAX_LENGTH,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=100
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[LogMetricsCallback]
)

trainer.train()
model.save_pretrained(f"{OUTPUT_DIR}/HILRfine-tuned-opus-mt-en-ru")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/HILRfine-tuned-opus-mt-en-ru")

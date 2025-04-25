# imports
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from evaluate import load
import numpy as np

# import vllm
from tqdm import tqdm
from lib.preprocessing import preprocess_data
from lib.training_utils import train_model

SRC_LANG = "ru"
TGT_LANG = "en"
MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
TRAIN_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
DEV_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
TEST_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
OUTPUT_DIR = "./results/backtranslation_model_finetuned_on_medline"

if __name__ == "__main__":
    if TRAIN_DATASET_NAME.startswith("sethjsa"):
        train_dataset = load_dataset(TRAIN_DATASET_NAME)
    else:
        print(f"Loading train from disk")
        train_dataset = load_from_disk(TRAIN_DATASET_NAME)
    dev_dataset = load_dataset(DEV_DATASET_NAME)
    test_dataset = load_dataset(TEST_DATASET_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # change the splits for actual training. here, using flores-dev as training set because it's small (<1k examples)
    tokenized_train_dataset = preprocess_data(train_dataset, tokenizer, SRC_LANG, TGT_LANG, "train")
    tokenized_dev_dataset = preprocess_data(dev_dataset, tokenizer, SRC_LANG, TGT_LANG, "dev")
    # Note(jp): Here test is the same as dev
    tokenized_test_dataset = preprocess_data(test_dataset, tokenizer, SRC_LANG, TGT_LANG, "dev")

    # print sizes
    print(f"Train dataset size: {len(tokenized_train_dataset)}")
    print(f"Dev dataset size: {len(tokenized_dev_dataset)}")
    print(f"Test dataset size: {len(tokenized_test_dataset)}")

    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "dev": tokenized_dev_dataset,
        "test": tokenized_test_dataset
    })

    # {'learning_rate': 0.0001, 'weight_decay': 0.05, 'num_train_epochs': 8}
    # modify these as you wish; RQ3 could involve testing effects of various hyperparameters
    training_args = Seq2SeqTrainingArguments(
        torch_compile=True, # generally speeds up training, try without it to see if it's faster for small datasets
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=64, # change batch sizes to fit your GPU memory and train faster
        per_device_eval_batch_size=128,
        weight_decay=0.05,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        save_total_limit=1, # modify this to save more checkpoints
        num_train_epochs=5, # modify this to train more epochs
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=128,
        warmup_steps=0,
        no_cuda=False,  # Set to False to enable GPU
        fp16=True,      # Enable mixed precision training for faster training
    )

    model_finetuned = train_model(MODEL_NAME, tokenized_datasets, tokenizer, training_args)

    # save the finetuned model
    model_finetuned.save_pretrained(OUTPUT_DIR)
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

SRC_LANG = "en"
TGT_LANG = "ru"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
TRAIN_DATASET_NAME = "results/datasets/sethjsa_medline_ru_mono_bt_results_backtranslation_model_finetuned_on_medline_n5_semantic_entropy"
DEV_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
TEST_DATASET_NAME = "sethjsa/medline_en_ru_parallel"
OUTPUT_DIR = "./results/medline_backtranslated_bt_model_finetuned_on_medline_90th_percentile"

SEMANTIC_ENTROPY_PERCENTILE = 90

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

    # if the train data contains column for "semantic_entropy"
    if "semantic_entropy" in train_dataset["train"].column_names:
        print(f"Train dataset contains semantic entropy scores")
        print(f"Train dataset size: {len(train_dataset['train'])}")
        # take the 90th percentile of the semantic entropy scores
        # and take only the examples with semantic entropy scores below that (90%)
        # percentile_90th = np.percentile(train_dataset["train"]["semantic_entropy"], SEMANTIC_ENTROPY_PERCENTILE)
        # print(f"{SEMANTIC_ENTROPY_PERCENTILE} percentile of semantic entropy: {percentile_90th}")
        # train_dataset["train"] = train_dataset["train"].filter(lambda x: x["semantic_entropy"] < percentile_90th)
        print(f"Train dataset size after filtering: {len(train_dataset['train'])}")

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
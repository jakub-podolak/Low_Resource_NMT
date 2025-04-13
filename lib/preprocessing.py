# imports
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict
from evaluate import load
import numpy as np
# import vllm
from tqdm import tqdm

def preprocess_data(dataset_dict, tokenizer, src_lang, tgt_lang, split, max_length=128):
    """
    Preprocess translation datasets

    Args:
        dataset_dict: Dictionary containing train/dev/test datasets
        tokenizer: Tokenizer object
        src_lang: Source language code
        tgt_lang: Target language code
        split: Dataset split to preprocess ('train', 'validation', etc)
        max_length: Maximum sequence length
    Returns:
        tokenized_dataset: Preprocessed dataset for specified split
    """
    def preprocess_function(examples):
        inputs = examples[src_lang]
        targets = examples[tgt_lang]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_dict[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict[split].column_names
    )

    return tokenized_dataset

def postprocess_predictions(predictions, labels, tokenizer):
    """
    Convert model outputs to decoded text

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        tokenizer: Tokenizer object
    Returns:
        decoded_preds: Decoded predictions
        decoded_labels: Decoded labels
    """
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return decoded_preds, decoded_labels


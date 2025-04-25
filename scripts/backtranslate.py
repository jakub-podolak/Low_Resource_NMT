import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from lib.training_utils import translate_text

SRC_LANG = "ru"
TGT_LANG = "en"

def main():
    parser = argparse.ArgumentParser(description="Backtranslate dataset")
    parser.add_argument(
        "--dataset_to_backtranslate",
        type=str,
        help="Huggingface dataset or path to a dataset to backtranslate."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to backtranslate (e.g. train, test, dev)."
    )
    parser.add_argument(
        "--baseline_model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-ru-en",
        help="Hugging Face model name for the baseline (e.g. Helsinki-NLP/opus-mt-ru-en)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint directory. If None, only the baseline model for backtranslation is used."
    )

    args = parser.parse_args()
    baseline_model_name = args.baseline_model_name
    checkpoint_path = args.checkpoint_path
    model_used = args.checkpoint_path if args.checkpoint_path is not None else baseline_model_name

    # load model
    if checkpoint_path is not None:
        print(f"[INFO] Loading checkpoint model from: {model_used}")
    else:
        print(f"[INFO] Loading baseline model from: {model_used}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_used)
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)

    # --- Load dataset ---
    print(f"[INFO] Loading dataset: {args.dataset_to_backtranslate}")
    dataset_bt = load_dataset(args.dataset_to_backtranslate)
    # select the split
    if args.split is not None:
        print(f"[INFO] Loading split: {args.split}")
        dataset_bt = dataset_bt[args.split]
    else:
        print("[INFO] No split specified, using the entire dataset.")
        dataset_bt = dataset_bt
    

    # translate the sentences in the dataset
    print("[INFO] Translating data...")
    sentences_english = translate_text(
        dataset_bt[SRC_LANG],
        model,
        tokenizer,
        max_length=128,
        batch_size=64
    )

    sententes_russian = dataset_bt[SRC_LANG]

    # print first sample:
    print(f"[INFO] First sentence in Russian: {sententes_russian[0]}")
    print(f"[INFO] First sentence in English: {sentences_english[0]}")

    # create a new dataset with the backtranslated sentences, with columns "en" and "ru"
    train_dataset = Dataset.from_dict(
        {
            "en": sentences_english,
            "ru": sententes_russian,
        }
    )

    # save the dataset to file that we can use later the same way as the original dataset
    ds = DatasetDict({"train": train_dataset})
    
    save_path = "results/datasets/" + args.dataset_to_backtranslate.replace('/', '_') +\
          "_backtranslated_with_" + model_used.replace('/', '_') + ".json"
    print(f"[INFO] Saving backtranslated dataset to: {save_path}")
    ds.save_to_disk(save_path)   

if __name__ == "__main__":
    main()

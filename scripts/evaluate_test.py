import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from lib.training_utils import translate_text
from lib.metrics import compute_bleu, compute_comet
from lib.preprocessing import preprocess_data

SRC_LANG = "en"
TGT_LANG = "ru"

# List of huggingface datasets and splits to use for testing
TEST_DATASETS = [
    ("sethjsa/medline_en_ru_parallel", "dev"),
    ("sethjsa/wmt20bio_en_ru_sent", "test"),
    ("sethjsa/tico_en_ru", "test"),
    ("sethjsa/flores_en_ru", "test"),
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate a translation model with BLEU and COMET.")
    parser.add_argument(
        "--baseline_model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-ru",
        help="Hugging Face model name for the baseline (e.g. Helsinki-NLP/opus-mt-en-ru)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint directory. If None, only the baseline model is evaluated."
    )

    args = parser.parse_args()
    baseline_model_name = args.baseline_model_name
    checkpoint_path = args.checkpoint_path
    # load always the baseline model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)

    # load model
    if checkpoint_path is not None:
        print(f"[INFO] Loading checkpoint model from: {checkpoint_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        
    else:
        print(f"[INFO] Loading baseline model from: {baseline_model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model_name)

    results = []

    # --- Load datasets ---
    print("[INFO] Loading datasets...")
    for dataset_name, split in TEST_DATASETS:
        print(f"[INFO] Loading dataset: {dataset_name} ({split})")
        test_dataset = load_dataset(dataset_name)
        
        print("[INFO] Translating test data...")
        predictions = translate_text(
            test_dataset[split][SRC_LANG],
            model,
            tokenizer,
            max_length=128,
            batch_size=64
        )

        # print first sample:
        print(f"[INFO] First sample: ")
        # source 
        print(f"Source: {test_dataset[split][SRC_LANG][0]}")
        # target
        print(f"Target: {test_dataset[split][TGT_LANG][0]}")
        # prediction
        print(f"Prediction: {predictions[0]}")

        print("[INFO] Computing metrics...")
        bleu_score = compute_bleu(test_dataset[split][TGT_LANG], predictions)
        comet_score = compute_comet(
            src=test_dataset[split][SRC_LANG],
            tgt=test_dataset[split][TGT_LANG],
            preds=predictions
        )
        print(f"[INFO] BLEU score: {bleu_score}")
        print(f"[INFO] COMET score: {comet_score}")
        results.append({
            "dataset": dataset_name,
            "samples": len(test_dataset[split]),
            "split": split,
            "bleu_score": bleu_score,
            "comet_score": comet_score
        })

    # Print results
    print("[INFO] Results:")
    for result in results:
        print(f"Dataset: {result['dataset']}, Split: {result['split']}, "\
              f"Samples: {result['samples']}, BLEU: {result['bleu_score']}, COMET: {result['comet_score']}")

if __name__ == "__main__":
    main()

"""
Baseline:
Dataset: sethjsa/medline_en_ru_parallel, Split: dev, Samples: 1000, śBLEU: 14.95690607848931, COMET: 0.7664666471034288

python3 scripts/evaluate_test.py --checkpoint_path results/checkpoints/HILRfine-tuned-opus-mt-en-ru
Dataset: sethjsa/medline_en_ru_parallel, Split: dev, Samples: 1000, BLEU: 8.131136051814785, COMET: 0.6586308072209358

python3 scripts/evaluate_test.py --checkpoint_path results/checkpoints/final_model/best-tuned-opus-mt-en-ru
Dataset: sethjsa/medline_en_ru_parallel, Split: dev, Samples: 1000, BLEU: 9.063520842593954, COMET: 0.6721206496059895
"""
"""
This module implements BLEU and COMET metrics necessary for evaluation of NMT models.
"""
# imports
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict
from evaluate import load
import numpy as np
# import vllm
from tqdm import tqdm

def compute_bleu(tgt, preds) -> float:
    """
    Calculate BLEU score for predictions

    Args:
        tgt: Target language texts
        preds: Predicted texts
    Returns:
        score: BLEU score
    """
    bleu = load("sacrebleu")
    results = bleu.compute(predictions=preds, references=[[l] for l in tgt])
    score = results["score"]
    return score

def compute_comet(src, tgt, preds) -> float:
    """
    Calculate COMET score for predictions

    Args:
        src: Source language texts
        tgt: Target language texts
        preds: Predicted texts
    Returns:
        metrics: Dictionary containing COMET score
    """
    comet_metrics = load("comet")

    comet_score = comet_metrics.compute(
        predictions=preds,
        references=tgt,
        sources=src,
    )

    return comet_score['mean_score']
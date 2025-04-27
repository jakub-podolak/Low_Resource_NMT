import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Allow `python backtranslate_with_semantic_entropy.py` from repository root
sys.path.insert(0, os.path.abspath("."))

SRC_LANG = "ru"
TGT_LANG = "en"

def translate_text_n_samples(
    texts: Sequence[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int = 32,
    max_length: int = 128,
    n_samples: int = 5,
    temperature: float = 1.0,
    num_beams: int = 4,
    top_k: int | None = None,
    top_p: float | None = None,
) -> list[list[str]]:
    """Return *n_samples* diverse translations for every input sentence.

    Implementation detail:  each source sentence is **tiled** *n_samples* times
    so that one call to ``model.generate`` yields all hypotheses for the whole
    batch.  This is much faster than looping.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    out_translations: list[list[str]] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        enc = {k: v.repeat_interleave(n_samples, dim=0) for k, v in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                num_beams=num_beams,
                num_return_sequences=1,  # 1 best per replica
                top_k=top_k or 0,
                top_p=top_p or 1.0,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        it = iter(decoded)
        out_translations.extend([list(next(it) for _ in range(n_samples)) for _ in batch])

    return out_translations

class EntailmentMNLI:
    """Utility wrapper around *microsoft/deberta‑v2‑xlarge‑mnli*."""

    def __init__(self, device: str):
        name = "microsoft/deberta-v2-xlarge-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        self.device = device

    @torch.no_grad()
    def _implication(self, premise: str, hypothesis: str) -> int:
        """Return 0 (contradiction) | 1 (neutral) | 2 (entailment)."""
        toks = self.tokenizer(premise, hypothesis, return_tensors="pt").to(self.device)
        logits = self.model(**toks).logits
        return torch.argmax(F.softmax(logits, dim=1)).item()

    def are_equivalent(self, s1: str, s2: str, *, strict: bool = False) -> bool:
        imp1 = self._implication(s1, s2)
        imp2 = self._implication(s2, s1)
        if strict:
            return (imp1 == 2) and (imp2 == 2)
        # Non‑strict:  accept if no contradiction and **not both neutral**.
        return (0 not in (imp1, imp2)) and not (imp1 == imp2 == 1)


def assign_semantic_ids(
    hypotheses: list[str],
    entailment: EntailmentMNLI,
    *,
    strict: bool = False,
) -> list[int]:
    """Cluster *hypotheses* into equivalence classes using entailment."""

    ids = [-1] * len(hypotheses)
    next_id = 0
    for i, h_i in enumerate(hypotheses):
        if ids[i] != -1:
            continue  # already assigned
        ids[i] = next_id
        for j in range(i + 1, len(hypotheses)):
            if ids[j] == -1 and entailment.are_equivalent(h_i, hypotheses[j], strict=strict):
                ids[j] = next_id
        next_id += 1
    return ids


def entropy_from_ids(ids: list[int]) -> float:
    counts = np.bincount(ids)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())  # type: ignore[no-any-return]

def main() -> None:
    parser = argparse.ArgumentParser(description="Back‑translate with semantic metadata")
    parser.add_argument("--dataset_to_backtranslate", type=str, required=True,
                        help="HuggingFace dataset name or local path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process (default: train)")
    parser.add_argument("--baseline_model_name", type=str,
                        default="Helsinki-NLP/opus-mt-ru-en",
                        help="HF name for baseline BT model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to fine‑tuned checkpoint (overrides baseline)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of stochastic translations per source")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    # load translation model
    model_path = args.checkpoint_path or args.baseline_model_name
    print(f"[INFO] Loading MT model: {model_path}")
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    mt_tok = AutoTokenizer.from_pretrained(args.baseline_model_name)

    # load data to backtranslate
    print(f"[INFO] Loading dataset: {args.dataset_to_backtranslate}")
    ds = load_dataset(args.dataset_to_backtranslate)
    ds_split = ds[args.split] if args.split else ds
    # TODO(jakub): add some limit to samples?
    src_sentences: List[str] = ds_split[SRC_LANG]
    print(f"[INFO] {len(src_sentences):,} sentences to translate.")

    # backtranslate
    translations = translate_text_n_samples(
        src_sentences,
        mt_model,
        mt_tok,
        batch_size=args.batch_size,
        max_length=args.max_length,
        n_samples=args.n_samples,
        temperature=args.temperature,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # semantic entropy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entailment = EntailmentMNLI(device)

    semantic_ids_list: list[list[int]] = []
    entropy_list: list[float] = []

    for hyps in tqdm(translations, desc="Computing semantic metadata"):
        ids = assign_semantic_ids(hyps, entailment, strict=False)
        semantic_ids_list.append(ids)
        entropy_list.append(entropy_from_ids(ids))

    # save data
    tgt_sentences = [hyps[0] for hyps in translations]
    hf_ds = Dataset.from_dict({
        SRC_LANG: src_sentences,
        TGT_LANG: tgt_sentences,
        "all_en_generations": translations,
        "semantic_ids": semantic_ids_list,
        "semantic_entropy": entropy_list,
    })
    out = DatasetDict({"train": hf_ds})

    save_dir = Path("results/datasets")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / (
        f"{args.dataset_to_backtranslate.replace('/', '_')}_bt_"
        f"{model_path.replace('/', '_')}_n{args.n_samples}_semantic_entropy"
    )
    print(f"[INFO] Saving dataset to: {save_path}")
    out.save_to_disk(str(save_path))

    # Try loading the dataset back
    print(f"[INFO] Loading dataset back from: {save_path}")
    loaded_ds = DatasetDict.load_from_disk(str(save_path))
    print(f"[INFO] Loaded dataset: {loaded_ds}")
    print(f"[INFO] Number of samples: {len(loaded_ds['train'])}")
    print(f"[INFO] First sample: {loaded_ds['train'][0]}")


if __name__ == "__main__":
    main()

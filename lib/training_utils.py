# imports
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict
from evaluate import load
import numpy as np
# import vllm
from tqdm import tqdm

from lib.preprocessing import postprocess_predictions

# evaluation: for validation (with raw outputs) and testing (from text)

# basic training loop
def compute_metrics_val(tokenizer, eval_preds):
    """
    Calculate BLEU score for predictions

    Args:
        tokenizer: Tokenizer object
        eval_preds: Tuple of predictions and labels
    Returns:
        metrics: Dictionary containing BLEU score
    """
    preds, labels = eval_preds
    decoded_preds, decoded_labels = postprocess_predictions(preds, labels, tokenizer)

    # Calculate BLEU score
    bleu = load("sacrebleu")
    results = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {"bleu": results["score"]}

def train_model(model_name, tokenized_datasets, tokenizer, training_args):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Verify GPU usage
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"] if "dev" in tokenized_datasets else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda x: compute_metrics_val(tokenizer, x)
    )

    trainer.train()
    return model

# generation (on GPU) for test time
def translate_text(texts, model, tokenizer, max_length=128, batch_size=32):
    """
    Translate texts using the model

    Args:
        texts: List of texts to translate
        model: Translation model
        tokenizer: Tokenizer object
        max_length: Maximum sequence length
        batch_size: Batch size for translation
    Returns:
        translations: List of translated texts
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    translations = []

    # Create tqdm progress bar
    progress_bar = tqdm(range(0, len(texts), batch_size), desc="Translating")

    for i in progress_bar:
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.0,
                early_stopping=True,
                num_beams=4,
                num_return_sequences=1,
            )

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations

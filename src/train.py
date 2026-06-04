"""LoRA fine-tuning script for logistics Q&A.

Usage (from repo root):

    python -m src.train                              # default config
    python -m src.train --epochs 5 --lr 3e-4         # override hyperparameters
    python -m src.train --no-4bit                    # full-precision LoRA (needs more VRAM)
    python -m src.train --no-wandb                   # disable W&B logging

Outputs:
    artifacts/checkpoints/   — intermediate + final LoRA adapter
    W&B run                  — losses, eval metrics, hyperparameters

Memory expectations (Qwen2.5-7B-Instruct, batch=4, seq=1024):
    QLoRA (4-bit):  ~10 GB VRAM   — fits on Colab T4 (16 GB)
    LoRA (bf16):    ~22 GB VRAM   — needs A100 / A10 / RTX 3090+
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import DATA, GEN, LORA, MODEL, TRAIN
from src.dataset import load_splits, tokenize_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Argument parsing ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-name", default=MODEL.name, help="HF model id")
    p.add_argument(
        "--no-4bit", action="store_true", help="Disable 4-bit quantization (use full LoRA)"
    )
    p.add_argument("--epochs", type=int, default=TRAIN.num_train_epochs)
    p.add_argument("--batch-size", type=int, default=TRAIN.per_device_train_batch_size)
    p.add_argument("--grad-accum", type=int, default=TRAIN.gradient_accumulation_steps)
    p.add_argument("--lr", type=float, default=TRAIN.learning_rate)
    p.add_argument("--max-seq-length", type=int, default=MODEL.max_seq_length)
    p.add_argument("--output-dir", default=TRAIN.output_dir)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--seed", type=int, default=TRAIN.seed)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir if one exists",
    )
    return p.parse_args()


# ---------- Model loading ----------


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load Qwen (or whichever model) + tokenizer with optional 4-bit quantization."""
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Most causal LM tokenizers don't have a pad token; reuse EOS. This is
        # safe because we mask pad positions in the attention mask.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Decide precision + quantization.
    use_4bit = not args.no_4bit
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Quantization: 4-bit NF4 with double-quant, compute=%s", compute_dtype)
    else:
        logger.info("Quantization: disabled (full %s LoRA)", compute_dtype)

    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    # Wrap with LoRA.
    peft_cfg = PeftLoraConfig(
        r=LORA.r,
        lora_alpha=LORA.alpha,
        lora_dropout=LORA.dropout,
        target_modules=LORA.target_modules,
        bias=LORA.bias,
        task_type=LORA.task_type,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------- Main ----------


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"
    else:
        if not os.environ.get("WANDB_API_KEY"):
            logger.warning(
                "WANDB_API_KEY not set — W&B logging will fail. Use --no-wandb to disable."
            )
        os.environ.setdefault("WANDB_PROJECT", TRAIN.wandb_project)
        report_to = TRAIN.report_to

    # Load data
    logger.info("Loading dataset splits")
    ds = load_splits(DATA.train_path, DATA.val_path, DATA.test_path)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Tokenize
    logger.info("Tokenizing train/val")
    train_tok = tokenize_dataset(ds["train"], tokenizer, args.max_seq_length)
    val_tok = tokenize_dataset(ds["validation"], tokenizer, args.max_seq_length)

    # Collator pads to the longest sequence in the batch.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    # Training args
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=TRAIN.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=TRAIN.warmup_ratio,
        lr_scheduler_type=TRAIN.lr_scheduler_type,
        weight_decay=TRAIN.weight_decay,
        logging_steps=TRAIN.logging_steps,
        eval_strategy=TRAIN.eval_strategy,
        eval_steps=TRAIN.eval_steps,
        save_strategy=TRAIN.save_strategy,
        save_steps=TRAIN.save_steps,
        save_total_limit=TRAIN.save_total_limit,
        gradient_checkpointing=TRAIN.gradient_checkpointing,
        optim=TRAIN.optim,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=report_to,
        run_name=f"{args.model_name.split('/')[-1]}-lora-r{LORA.r}",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting training | epochs=%d, effective_batch=%d, lr=%.2e",
        args.epochs,
        args.batch_size * args.grad_accum,
        args.lr,
    )

    resume_path = None
    if args.resume:
        existing = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if existing:
            resume_path = str(existing[-1])
            logger.info("Resuming from checkpoint: %s", resume_path)
        else:
            logger.info("--resume passed but no checkpoint found; starting fresh")
    trainer.train(resume_from_checkpoint=resume_path)

    final_dir = output_dir / "final"
    logger.info("Saving final adapter to %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Dump the resolved config alongside the adapter for reproducibility.
    config_dump = {
        "model": asdict(MODEL),
        "lora": asdict(LORA),
        "train": asdict(TRAIN),
        "generation": asdict(GEN),
        "cli_overrides": vars(args),
    }
    (final_dir / "training_config.json").write_text(json.dumps(config_dump, indent=2))

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

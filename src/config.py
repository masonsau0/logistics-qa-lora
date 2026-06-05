"""Central training and model configuration.

Everything tunable lives here. The training script, evaluation script, and
inference server all import from this module, so changing a hyperparameter in
one place propagates everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------- Paths ----------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
RESULTS_DIR = ARTIFACTS_DIR / "results"


# ---------- Model ----------


@dataclass
class ModelConfig:
    """Base model selection.

    Qwen2.5-7B-Instruct is Apache-2.0 licensed and downloadable from Hugging Face
    without gating. Swap `name` to target a different model — the code path is
    model-agnostic as long as the model supports the standard chat template.
    """

    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Use 4-bit quantization (QLoRA). Required to fit 7B on a Colab T4 (16 GB).
    # Set to False if you have an A100 / H100 and want full-precision LoRA.
    load_in_4bit: bool = True
    # bf16 if your GPU supports it (Ampere+, e.g. A100/A10/H100/RTX 3090+),
    # fp16 otherwise (T4 / V100). The train script auto-detects.
    bf16: bool = True
    max_seq_length: int = 1024


# ---------- LoRA ----------


@dataclass
class LoraConfig:
    """LoRA adapter hyperparameters.

    r=16, alpha=32 is the standard "balanced" config — a good default for 7B
    models on instruction-style data. Increase r for more capacity (and slower
    training); decrease for cheaper runs.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Target attention projections. Adding "gate_proj", "up_proj", "down_proj"
    # (the MLP layers) increases capacity but ~doubles trainable parameter count.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ---------- Training ----------


@dataclass
class TrainConfig:
    """Training-loop hyperparameters.

    Defaults target a Colab free-tier T4 with QLoRA. Adjust batch sizes upward
    if you have more VRAM.
    """

    output_dir: str = str(CHECKPOINT_DIR)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    gradient_checkpointing: bool = True  # trades compute for memory
    optim: str = "paged_adamw_8bit"  # 8-bit optimizer (bitsandbytes)
    seed: int = 42
    # Weights & Biases project name. Set WANDB_API_KEY in env to enable.
    wandb_project: str = "logistics-qa-lora"
    report_to: str = "wandb"  # set to "none" to disable W&B


# ---------- Dataset ----------


@dataclass
class DatasetConfig:
    """Dataset locations and split fractions."""

    train_path: str = str(DATA_DIR / "train.jsonl")
    val_path: str = str(DATA_DIR / "val.jsonl")
    test_path: str = str(DATA_DIR / "test.jsonl")
    # Generation target — used by data/prepare_dataset.py.
    target_total_examples: int = 12_000
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    # Categories we generate Q&A for. See data/seed_topics.json for the
    # actual prompts driving each category.
    categories: list[str] = field(
        default_factory=lambda: [
            "freight_calculations",
            "carrier_performance",
            "claims_and_damages",
            "routing_and_dispatch",
            "sops_and_procedures",
            "regulations_and_compliance",
        ]
    )


# ---------- Generation ----------


@dataclass
class GenerationConfig:
    """Inference-time generation settings."""

    max_new_tokens: int = 512
    temperature: float = 0.3  # low — domain Q&A benefits from determinism
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True


# Singletons — import these for convenience.
MODEL = ModelConfig()
LORA = LoraConfig()
TRAIN = TrainConfig()
DATA = DatasetConfig()
GEN = GenerationConfig()

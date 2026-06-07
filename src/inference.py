"""Inference utilities.

Two entry points:

  - `LogisticsQA` class:  high-level, loads the adapter once, reusable for
                          batch evaluation and the FastAPI server.
  - `generate_stream`:    async generator yielding tokens for SSE streaming.

The model is loaded once at construction; subsequent `.answer(...)` calls
reuse it. For multi-process serving (e.g. uvicorn workers > 1), each worker
gets its own model copy — that's expensive in VRAM. Use workers=1 for GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from src.config import GEN, MODEL
from src.dataset import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LogisticsQA:
    """Wraps base model + LoRA adapter for inference.

    Args:
        base_model_name: HF model id of the base (e.g. "Qwen/Qwen2.5-7B-Instruct").
        adapter_path: Path to a directory containing the LoRA adapter (saved by
            train.py). If None, runs against the base model — useful as the
            baseline comparison in evaluation.
        load_in_4bit: Use 4-bit quantization at inference. Default True; matches
            training-time quantization and fits 7B models on a T4.
    """

    def __init__(
        self,
        base_model_name: str = MODEL.name,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        logger.info("Loading tokenizer from %s", base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # bf16 only on Ampere+ (compute capability >= 8.0). T4 (7.5) reports
        # is_bf16_supported() == True via CUDA emulation but actually has no native
        # bf16 tensor cores — using bf16 there makes inference 3-5x slower than fp16.
        bf16_ok = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability()[0] >= 8
        )
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

        bnb_config = None
        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading base model %s (4bit=%s)", base_model_name, load_in_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if adapter_path is not None:
            logger.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

        self.model = model
        self.has_adapter = adapter_path is not None

    # ---------- Prompt construction ----------

    def _build_prompt(self, question: str, system: str | None = None) -> str:
        """Apply the chat template, leaving the assistant turn open for generation."""
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ---------- Synchronous generation ----------

    @torch.inference_mode()
    def answer(
        self,
        question: str,
        max_new_tokens: int = GEN.max_new_tokens,
        temperature: float = GEN.temperature,
        top_p: float = GEN.top_p,
        system: str | None = None,
    ) -> str:
        """Generate a single answer. Returns just the assistant's reply text."""
        prompt = self._build_prompt(question, system=system)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            repetition_penalty=GEN.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens, not the prompt.
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ---------- Streaming generation ----------

    def stream(
        self,
        question: str,
        max_new_tokens: int = GEN.max_new_tokens,
        temperature: float = GEN.temperature,
        top_p: float = GEN.top_p,
        system: str | None = None,
    ) -> Iterator[str]:
        """Yield tokens as they're generated. Used by the FastAPI SSE endpoint.

        We run `.generate()` in a background thread and consume the streamer
        from the main thread, because HF generate is blocking.
        """
        prompt = self._build_prompt(question, system=system)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            repetition_penalty=GEN.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()

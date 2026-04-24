"""
Medical LoRA Distillation Training Pipeline
============================================
Production-ready, single-file script for Google Colab.

How to use:
  1. Open a new Colab notebook with a T4/L4/A100 GPU runtime
  2. Paste each section into separate cells (or run the whole file)
  3. Edit the CONFIG dict to point to your dataset
  4. Run all cells
"""

# ============================================================
# 0. INSTALL DEPENDENCIES  (run this cell FIRST, then restart runtime)
# ============================================================
#
# IMPORTANT: After running this cell, go to Runtime > Restart Runtime
# before running anything else. This avoids stale module caches.
#
# ---- Copy this into its own Colab cell ----
#
# %%capture
# !pip install unsloth
# !pip install --no-deps trl peft accelerate bitsandbytes xformers
#
# -------------------------------------------
#
# WHY this works and pinning versions doesn't:
#   - Colab pre-installs torch + CUDA matched to its GPU driver.
#   - Pinning transformers==4.46.3 or trl==0.11.4 etc. forces pip to
#     downgrade/upgrade torch's transitive deps, breaking CUDA.
#   - `pip install unsloth` pulls the versions it was tested against.
#   - `--no-deps` installs the tools without touching torch.

# ============================================================
# 1. IMPORTS  — unsloth MUST come first (it monkey-patches internals)
# ============================================================

import os
import json
import random
import sys
import time
import gc
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Unsloth must be imported BEFORE transformers/trl/peft
from unsloth import FastLanguageModel

from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

# ============================================================
# 2. LOGGING
# ============================================================

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger("distill")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))
    logger.addHandler(handler)
    return logger


log = setup_logging()

# ============================================================
# 3. CONFIG — edit this dict, everything flows from here
# ============================================================

CONFIG = {
    # ── Paths ──
    "dataset_file": "medical_distilled_dataset.json",
    "output_dir": "medical_lora_adapter",

    # ── Reproducibility ──
    "seed": 42,

    # ── Base model ──
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_seq_length": 1024,
    "load_in_4bit": True,
    "dtype": None,  # None = auto, or torch.float16 / torch.bfloat16

    # ── LoRA ──
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    "lora_bias": "none",
    "use_gradient_checkpointing": "unsloth",

    # ── Training ──
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,

    # Precision — "auto" picks the best for your GPU, or "fp16" / "bf16" / "fp32"
    "precision": "auto",

    # ── Eval & checkpointing ──
    "logging_steps": 10,
    "eval_steps": 25,
    "save_steps": 25,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",

    # ── Early stopping (set patience=0 to disable) ──
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,

    # ── Data ──
    "test_size": 0.1,
    "max_samples": None,  # set to e.g. 50 for quick debug runs

    # ── Reporting: "none", "wandb", "tensorboard" ──
    "report_to": "none",

    # ── Resume from checkpoint (set path string or None) ──
    "resume_from_checkpoint": None,
}

# ============================================================
# 4. REPRODUCIBILITY
# ============================================================


def set_seed(seed: int) -> None:
    """Pin every source of randomness for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Random seed set to %d", seed)


# ============================================================
# 5. GPU DIAGNOSTICS
# ============================================================


def log_gpu_info() -> None:
    """Log GPU memory stats — helpful for debugging OOM."""
    if not torch.cuda.is_available():
        log.info("CUDA not available — CPU-only run.")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_mem / (1024 ** 3)
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        log.info(
            "GPU %d: %s | %.1f GB total | %.2f GB allocated | %.2f GB reserved",
            i, props.name, total, alloc, reserved,
        )


# ============================================================
# 6. PRECISION AUTO-DETECTION
# ============================================================


def resolve_precision(mode: str) -> tuple:
    """
    Return (fp16, bf16) flags.
    mode: "auto", "fp16", "bf16", or "fp32"
    """
    mode = (mode or "auto").lower().strip()

    if mode == "fp16":
        return True, False
    if mode in ("bf16", "bfloat16"):
        return False, True
    if mode == "fp32":
        return False, False

    # Auto-detect
    if not torch.cuda.is_available():
        log.info("No CUDA — training in fp32.")
        return False, False
    if torch.cuda.is_bf16_supported():
        log.info("GPU supports bf16 — enabling bf16 training.")
        return False, True
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 7:
        log.info("GPU capability %d.%d — enabling fp16 training.", *cap)
        return True, False
    log.info("GPU capability %d.%d — training in fp32.", *cap)
    return False, False


# ============================================================
# 7. DATASET VALIDATION & LOADING
# ============================================================

REQUIRED_INPUT_KEYS = ("hr", "rr", "bp", "spo2", "temp", "type", "symptoms", "doctor_notes")


def validate_sample(sample: dict, idx: int) -> bool:
    """Return True if sample has the required schema."""
    if "input" not in sample or "output" not in sample:
        log.warning("Sample %d: missing 'input' or 'output' — skipped.", idx)
        return False
    missing = [k for k in REQUIRED_INPUT_KEYS if k not in sample["input"]]
    if missing:
        log.warning("Sample %d: missing keys %s — skipped.", idx, missing)
        return False
    return True


def load_dataset_file(path: str, max_samples: Optional[int] = None) -> list:
    """Load, validate, and optionally truncate the JSON dataset."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    log.info("Loading dataset from %s ...", p.resolve())
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset JSON must be a top-level array of objects.")

    valid = [s for i, s in enumerate(raw) if validate_sample(s, i)]
    skipped = len(raw) - len(valid)
    if skipped:
        log.warning("Skipped %d / %d malformed samples.", skipped, len(raw))
    if not valid:
        raise ValueError("No valid samples remaining after validation.")

    if max_samples and max_samples > 0:
        valid = valid[:max_samples]
        log.info("Truncated to %d samples (debug mode).", max_samples)

    log.info("Dataset ready: %d valid samples.", len(valid))
    return valid


def format_sample(sample: dict, tokenizer: Any) -> dict:
    """Convert a raw sample into a tokenizer-formatted text string."""
    case = sample["input"]
    out = sample["output"]

    user_msg = (
        "You are a clinical triage assistant.\n\n"
        f"Patient Type: {case['type']}\n"
        f"HR: {case['hr']}  RR: {case['rr']}  BP: {case['bp']}\n"
        f"SpO2: {case['spo2']}  Temp: {case['temp']}\n"
        f"Symptoms: {case['symptoms']}\n"
        f"Notes: {case['doctor_notes']}\n\n"
        "Return JSON only."
    )
    assistant_msg = json.dumps(out, indent=2, ensure_ascii=False)

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


# ============================================================
# 8. CHECKPOINT SAVE HELPER
# ============================================================


def save_adapter(model: Any, tokenizer: Any, path: str) -> None:
    """Save LoRA adapter weights + tokenizer."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    log.info("Saving LoRA adapter to %s ...", p)
    model.save_pretrained(str(p))
    tokenizer.save_pretrained(str(p))
    log.info("Saved (%d files).", len(list(p.iterdir())))


# ============================================================
# 9. MAIN TRAINING PIPELINE
# ============================================================


def train(cfg: dict) -> None:
    """End-to-end LoRA distillation pipeline."""

    log.info("=" * 60)
    log.info("  Medical LoRA Distillation Pipeline")
    log.info("=" * 60)

    # ── Persist effective config ──
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable_cfg = {k: (str(v) if isinstance(v, type) else v) for k, v in cfg.items()}
    config_path = out_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(serializable_cfg, f, indent=2, default=str)
    log.info("Config snapshot → %s", config_path)

    # ── Seed ──
    set_seed(cfg["seed"])

    # ── GPU ──
    log_gpu_info()

    # ──────────────────────────────────────
    # 1. LOAD BASE MODEL
    # ──────────────────────────────────────
    log.info("Loading base model: %s", cfg["model_name"])
    t0 = time.perf_counter()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=cfg["dtype"],
    )
    log.info("Model loaded in %.1fs.", time.perf_counter() - t0)

    # ──────────────────────────────────────
    # 2. ATTACH LoRA ADAPTER
    # ──────────────────────────────────────
    log.info(
        "LoRA config: r=%d  alpha=%d  dropout=%.3f  targets=%s",
        cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["lora_target_modules"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["lora_target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"],
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        random_state=cfg["seed"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s trainable / %s total (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # ──────────────────────────────────────
    # 3. LOAD & VALIDATE DATASET
    # ──────────────────────────────────────
    raw = load_dataset_file(cfg["dataset_file"], cfg.get("max_samples"))

    log.info("Formatting samples with chat template...")
    formatted = [format_sample(s, tokenizer) for s in raw]

    # Warn about truncation
    oversized = sum(
        1 for s in formatted
        if len(tokenizer.encode(s["text"], add_special_tokens=False)) > cfg["max_seq_length"]
    )
    if oversized:
        log.warning(
            "%d / %d samples exceed max_seq_length (%d) — will be truncated.",
            oversized, len(formatted), cfg["max_seq_length"],
        )

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])
    train_ds, val_ds = split["train"], split["test"]
    log.info("Split: %d train | %d val", len(train_ds), len(val_ds))

    # ──────────────────────────────────────
    # 4. PRECISION
    # ──────────────────────────────────────
    fp16, bf16 = resolve_precision(cfg.get("precision", "auto"))

    # ──────────────────────────────────────
    # 5. BUILD TRAINER
    # ──────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg["weight_decay"],
        fp16=fp16,
        bf16=bf16,
        logging_steps=cfg["logging_steps"],

        # NOTE: use eval_strategy, NOT evaluation_strategy
        # evaluation_strategy was removed in transformers >= 4.46
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],

        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=False,
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        dataloader_pin_memory=True,
        run_name=f"distill-{cfg['model_name'].split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}",
    )

    callbacks = []
    patience = cfg.get("early_stopping_patience", 0)
    if patience and patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=cfg.get("early_stopping_threshold", 0.001),
            )
        )
        log.info("Early stopping: patience=%d, threshold=%.4f", patience, cfg.get("early_stopping_threshold", 0.001))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=training_args,
        callbacks=callbacks,
    )

    # ──────────────────────────────────────
    # 6. TRAIN
    # ──────────────────────────────────────
    log.info("Starting LoRA distillation training...")
    train_start = time.perf_counter()

    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt and not Path(resume_ckpt).exists():
        log.warning("Checkpoint %s not found — starting from scratch.", resume_ckpt)
        resume_ckpt = None

    try:
        result = trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        log.warning("Training interrupted — saving emergency checkpoint...")
        save_adapter(model, tokenizer, str(out_dir / "emergency_checkpoint"))
        log.info("Emergency checkpoint saved. Set resume_from_checkpoint to continue.")
        return
    except torch.cuda.OutOfMemoryError:
        log.critical(
            "CUDA OOM! Try reducing: per_device_train_batch_size (now %d), "
            "max_seq_length (now %d), or lora_r (now %d).",
            cfg["per_device_train_batch_size"], cfg["max_seq_length"], cfg["lora_r"],
        )
        raise

    elapsed = time.perf_counter() - train_start
    log.info("Training completed in %.1f minutes.", elapsed / 60)

    if result.metrics:
        log.info("Train metrics:\n%s", json.dumps(result.metrics, indent=2))

    # ──────────────────────────────────────
    # 7. FINAL EVALUATION
    # ──────────────────────────────────────
    log.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    log.info("Val metrics:\n%s", json.dumps(eval_result, indent=2))

    # Persist metrics
    metrics_path = out_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"train": result.metrics, "eval": eval_result, "elapsed_seconds": elapsed},
            f, indent=2,
        )
    log.info("Metrics → %s", metrics_path)

    # ──────────────────────────────────────
    # 8. SAVE LoRA ADAPTER
    # ──────────────────────────────────────
    save_adapter(model, tokenizer, cfg["output_dir"])

    # GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_info()

    log.info("=" * 60)
    log.info("  DONE — adapter saved to: %s", out_dir.resolve())
    log.info("=" * 60)
    log.info(
        "\nNext steps:\n"
        "  1. Merge:   model.merge_and_unload()\n"
        "  2. Export:   model.save_pretrained('merged/')\n"
        "  3. GGUF:    python llama.cpp/convert.py merged/\n"
    )


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":
    train(CONFIG)

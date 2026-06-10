import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_ROOT = Path(r"C:\Pesquisa\tcc\Ic-llmToNetworkConfig")
DEFAULT_DATASET_DIR = DEFAULT_ROOT / "fine-tuning-dataset"
DEFAULT_TRAIN_FILE = DEFAULT_DATASET_DIR / "discretize_chat_split_notes_train.jsonl"
DEFAULT_VAL_FILE = DEFAULT_DATASET_DIR / "discretize_chat_split_notes_val.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs" / "discretize_qlora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for intent discretization."
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model checkpoint name or local path.",
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_TRAIN_FILE),
        help="Path to the training JSONL file.",
    )
    parser.add_argument(
        "--val-file",
        default=str(DEFAULT_VAL_FILE),
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where adapters and checkpoints will be saved.",
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization mode for the base model.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload for 8-bit loading attempts.",
    )
    parser.add_argument(
        "--offload-folder",
        default=str(DEFAULT_ROOT / "outputs" / "offload"),
        help="Folder used by accelerate/transformers when offloading tensors.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def format_messages_as_text(example: dict, tokenizer) -> dict:
    messages = example.get("messages") or []

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        parts = []
        for msg in messages:
            role = (msg.get("role") or "").upper()
            content = msg.get("content") or ""
            parts.append(f"{role}:\n{content}")
        text = "\n\n".join(parts)

    return {"text": text}


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload_dir = Path(args.offload_folder)
    offload_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = resolve_dtype()

    bnb_config = None
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=args.cpu_offload,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_folder"] = str(offload_dir)
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )

    dataset = dataset.map(
        lambda ex: format_messages_as_text(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        optim="paged_adamw_8bit" if args.quantization in {"4bit", "8bit"} else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    metadata = {
        "model_name": args.model_name,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "output_dir": str(output_dir),
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "quantization": args.quantization,
        "cpu_offload": args.cpu_offload,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

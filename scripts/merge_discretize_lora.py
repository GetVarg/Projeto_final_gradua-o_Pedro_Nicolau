import argparse
import shutil
import sys
import traceback
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = r"C:\Pesquisa\models\Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = r"C:\Pesquisa\models\llama-8b-discretize-lora-adapter"
DEFAULT_OUTPUT = r"C:\Pesquisa\models\llama-8b-discretize-merged"


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path.anchor or path)
    return usage.free / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the discretize LoRA adapter into the local Llama base model.")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()

    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    output_path = Path(args.output)

    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    available_gb = free_gb(output_path.parent)
    if available_gb < args.min_free_gb:
        raise RuntimeError(
            f"Not enough free disk space to merge safely. "
            f"Available={available_gb:.2f} GB, required>={args.min_free_gb:.2f} GB. "
            f"Free space or choose another --output drive."
        )

    print(f"[merge] base={base_path}", flush=True)
    print(f"[merge] adapter={adapter_path}", flush=True)
    print(f"[merge] output={output_path}", flush=True)
    print(f"[merge] free_gb={available_gb:.2f}", flush=True)

    print("[merge] loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[merge] loading base model on CPU", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    print("[merge] loading adapter", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )
    print("[merge] merging adapter into base model", flush=True)
    merged = model.merge_and_unload()

    print("[merge] saving merged model", flush=True)
    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    print("[merge] saving tokenizer", flush=True)
    tokenizer.save_pretrained(output_path)

    print("[merge] done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[merge][fatal] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

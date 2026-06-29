from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import grafo_final as g


def _load_intents_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                intent = item.get("intent") or (item.get("input") or {}).get("intent")
                if isinstance(intent, str) and intent.strip():
                    out.append(intent.strip())
        return out

    if isinstance(data, dict):
        cases = data.get("cases") or data.get("items") or []
        if isinstance(cases, list):
            return [
                intent.strip()
                for item in cases
                if isinstance(item, dict)
                for intent in [item.get("intent") or (item.get("input") or {}).get("intent")]
                if isinstance(intent, str) and intent.strip()
            ]

    raise ValueError(f"Unsupported intent file format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the R2 pipeline. By default it uses the current gabriel/10/0 topology "
            "and the golden-intent suite; with --intent it runs ad-hoc intents without "
            "requiring a golden standard."
        )
    )
    parser.add_argument(
        "--intent",
        action="append",
        default=None,
        help="Intent text to run. Can be passed more than once. Quote the text in the terminal.",
    )
    parser.add_argument(
        "--intent-file",
        type=Path,
        default=None,
        help="File containing intents. Accepts plain text lines, a JSON list, or JSON cases/items.",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path(g.TOPOLOGY_PATH),
        help="Topology JSON path. Defaults to the current gabriel/10/0 topology.",
    )
    parser.add_argument(
        "--run-label",
        default="r2",
        help="Label used in output report filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = g.build_graph_debug()
    test_intents = []

    if args.intent:
        test_intents.extend(intent.strip() for intent in args.intent if intent.strip())

    if args.intent_file:
        intent_file = g.resolve_existing_path(args.intent_file)
        test_intents.extend(_load_intents_from_file(intent_file))

    if not test_intents:
        test_intents = [
            case["intent"]
            for case in (g.GOLDEN_COMMANDS_BY_INTENT.values() or g.BENCHMARK_DIAGNOSTIC_CASES)
        ]

    topology_path = g.resolve_existing_path(args.topology)

    print(f"[R2 PIPELINE ONLY] intents={len(test_intents)}")
    print(f"[R2 PIPELINE ONLY] topology={topology_path}")
    print(f"[R2 PIPELINE ONLY] discretize_model={g.LLAMA_8B_MODEL}")
    print(f"[R2 PIPELINE ONLY] downstream_model={g.LLAMA_8B_MODEL}")

    pipeline_report = g.run_intent_suite(
        app,
        test_intents,
        g.LLAMA_8B_MODEL,
        run_label=args.run_label,
        downstream_model_name=g.LLAMA_8B_MODEL,
        topology_path=topology_path,
    )
    pipeline_report["profile"] = "discretize_8b_downstream_8b"

    pipeline_json_path = g.save_benchmark_report(pipeline_report)
    pipeline_txt_path = g.save_text_report(pipeline_report)
    argument_resolver_log_path = g.save_argument_resolver_log(pipeline_report)
    discretize_json_path = g.save_benchmark_report(
        g.build_discretize_only_report(pipeline_report)
    )

    print(f"[PIPELINE R2] json={pipeline_json_path}")
    print(f"[PIPELINE R2] txt={pipeline_txt_path}")
    print(f"[PIPELINE R2] argument_resolver_log={argument_resolver_log_path}")
    print(f"[PIPELINE R2] discretize_only={discretize_json_path}")


if __name__ == "__main__":
    main()

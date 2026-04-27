from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_like(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import json5  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Could not parse {path} as JSON. Install json5 or fix the file."
            ) from exc
        return json5.loads(text)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    pieces = text.split()
    return " ".join(pieces)


def iter_dataset_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        return [item for item in raw["items"] if isinstance(item, dict)]
    raise ValueError("Unsupported dataset format. Expected a list or a dict with an 'items' list.")


def pick_items(
    items: List[Dict[str, Any]],
    ids: Optional[List[int]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    chosen = items

    if ids:
        wanted = set(ids)
        chosen = [item for item in chosen if item.get("id") in wanted]
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def patch_topology_loader(topology_path: Optional[Path]) -> None:
    if topology_path is None:
        return

    topo = load_json_like(topology_path)

    import no_grafo  # local project module

    def _patched_load_topologia() -> Dict[str, Any]:
        return copy.deepcopy(topo)

    no_grafo.load_topologia = _patched_load_topologia

    try:
        import tools  # type: ignore

        tools.load_topologia = _patched_load_topologia
    except Exception:
        pass


def build_expected_view(dataset_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sub in ((dataset_item.get("target") or {}).get("subintents") or []):
        if not isinstance(sub, dict):
            continue
        out.append(
            {
                "id": sub.get("id"),
                "text": sub.get("text", ""),
                "anchors": sub.get("anchors") or [],
                "notes": sub.get("notes") or {},
                "expected_op": ((sub.get("classification") or {}).get("op")),
            }
        )
    return out


def build_actual_view(result_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    work = result_state.get("work") or {}
    for sub in work.get("subintents") or []:
        if not isinstance(sub, dict):
            continue
        cls = sub.get("classification") or {}
        out.append(
            {
                "id": sub.get("id"),
                "original_text": sub.get("original_text", ""),
                "text": sub.get("text", ""),
                "anchors": sub.get("anchors") or [],
                "traits": sub.get("traits") or {},
                "status": sub.get("status"),
                "depth": sub.get("depth"),
                "actual_op": cls.get("name"),
                "confidence": cls.get("confidence"),
                "rationale": cls.get("rationale"),
                "attempts": sub.get("attempts") or [],
            }
        )
    return out


def compare_by_index(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    max_len = max(len(expected), len(actual))
    text_matches = 0
    op_matches = 0

    for idx in range(max_len):
        exp = expected[idx] if idx < len(expected) else {}
        act = actual[idx] if idx < len(actual) else {}

        exp_text = exp.get("text", "")
        act_text = act.get("text", "")
        exp_op = exp.get("expected_op")
        act_op = act.get("actual_op")

        text_equal = norm_text(exp_text) == norm_text(act_text) and bool(exp_text or act_text)
        op_equal = exp_op == act_op and (exp_op is not None or act_op is not None)

        if text_equal:
            text_matches += 1
        if op_equal:
            op_matches += 1

        rows.append(
            {
                "index": idx,
                "expected_id": exp.get("id"),
                "actual_id": act.get("id"),
                "expected_text": exp_text,
                "actual_text": act_text,
                "text_equal": text_equal,
                "expected_op": exp_op,
                "actual_op": act_op,
                "op_equal": op_equal,
                "actual_status": act.get("status"),
                "actual_confidence": act.get("confidence"),
            }
        )

    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "count_equal": len(expected) == len(actual),
        "text_matches": text_matches,
        "op_matches": op_matches,
        "rows": rows,
    }


def summarize_item(
    dataset_item: Dict[str, Any],
    result_state: Dict[str, Any],
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_item.get("id"),
        "task_view": dataset_item.get("task_view"),
        "root_intent": ((dataset_item.get("input") or {}).get("intent") or "").strip(),
        "orchestrator_ok": not bool(result_state.get("error")),
        "needs_human": bool(result_state.get("needs_human", False)),
        "warnings": result_state.get("warnings") or [],
        "root_entities": result_state.get("root_entities") or [],
        "root_entity_selectors": result_state.get("root_entity_selectors") or [],
        "root_context_keys": sorted(list((result_state.get("root_context") or {}).keys())),
        "root_topology_slice_device_count": len(((result_state.get("root_topology_slice") or {}).get("devices") or {})),
        "expected": expected,
        "actual": actual,
        "comparison": comparison,
    }


def print_console_report(items: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print("ORCHESTRATOR VALIDATION REPORT")
    print("=" * 100)

    total = len(items)
    count_equal = sum(1 for item in items if item["comparison"]["count_equal"])
    full_text_match = sum(
        1
        for item in items
        if item["comparison"]["expected_count"] == item["comparison"]["text_matches"]
        and item["comparison"]["expected_count"] == item["comparison"]["actual_count"]
    )
    full_op_match = sum(
        1
        for item in items
        if item["comparison"]["expected_count"] == item["comparison"]["op_matches"]
        and item["comparison"]["expected_count"] == item["comparison"]["actual_count"]
    )

    print(f"items tested      : {total}")
    print(f"same subintent qty: {count_equal}/{total}")
    print(f"full text match   : {full_text_match}/{total}")
    print(f"full op match     : {full_op_match}/{total}")
    print()

    for item in items:
        comp = item["comparison"]
        print("-" * 100)
        print(f"dataset_id   : {item['dataset_id']}")
        print(f"needs_human  : {item['needs_human']}")
        print(f"warnings     : {len(item['warnings'])}")
        print(f"expected/actual subintents: {comp['expected_count']}/{comp['actual_count']}")
        print(f"text matches : {comp['text_matches']}/{max(comp['expected_count'], comp['actual_count'])}")
        print(f"op matches   : {comp['op_matches']}/{max(comp['expected_count'], comp['actual_count'])}")
        print(f"intent       : {item['root_intent']}")

        for row in comp["rows"]:
            print(
                f"  [{row['index']}] exp_op={row['expected_op']} | act_op={row['actual_op']} | "
                f"text_equal={row['text_equal']} | op_equal={row['op_equal']}"
            )
            if not row["text_equal"] or not row["op_equal"]:
                if row["expected_text"]:
                    print(f"      expected: {row['expected_text']}")
                if row["actual_text"]:
                    print(f"      actual  : {row['actual_text']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run node_orchestrator_root_grouding over intents from primeiro_dataset.json "
            "and compare the produced subintents/classifications with the dataset targets."
        )
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset/primeiro_dataset.json"))
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=19)
    parser.add_argument("--ids", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/orchestrator_validation_report.json"))
    args = parser.parse_args()

    import no_grafo

    patch_topology_loader(args.topology)

    raw_dataset = load_json_like(args.dataset)
    items = iter_dataset_items(raw_dataset)
    chosen = pick_items(items, ids=args.ids, limit=args.limit)

    results: List[Dict[str, Any]] = []

    for dataset_item in chosen:
        intent_text = ((dataset_item.get("input") or {}).get("intent") or "").strip()
        if not intent_text:
            continue

        state: Dict[str, Any] = {
            "user_intent_text": intent_text,
            "timestamp": datetime.now().isoformat(),
            "warnings": [],
            "needs_human": False,
            "error": None,
            "controller": {
                "rag_min_score_schedule": [0.35, 0.25, 0.18, 0.12, 0.08],
                "min_intent_confidence": 0.70,
                "max_refine_depth": 2,
            },
        }

        try:
            result_state = no_grafo.node_orchestrator_root_grouding(copy.deepcopy(state))
        except Exception as exc:
            result_state = copy.deepcopy(state)
            result_state["error"] = str(exc)
            result_state["needs_human"] = True
            result_state.setdefault("warnings", []).append("orchestrator exception")

        expected = build_expected_view(dataset_item)
        actual = build_actual_view(result_state)
        comparison = compare_by_index(expected, actual)
        results.append(summarize_item(dataset_item, result_state, expected, actual, comparison))

    report = {
        "dataset_path": str(args.dataset.resolve()),
        "topology_path": str(args.topology.resolve()) if args.topology else None,
        "tested_items": len(results),
        "generated_at": datetime.now().isoformat(),
        "results": results,
    }

    save_json(args.output, report)
    print_console_report(results)
    print()
    print(f"JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()

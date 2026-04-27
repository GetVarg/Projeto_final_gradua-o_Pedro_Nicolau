from __future__ import annotations

import argparse
import re
from pathlib import Path

SECTION_HEADERS = [
    re.compile(r"^={10,}$"),
    re.compile(r"^\[BENCHMARK CONFIG\]"),
    re.compile(r"^\[BENCHMARK\] Running model:"),
    re.compile(r"^\[PIPELINE\]\[INTENT \d+\]"),
    re.compile(r"^\[BASELINE\]\[INTENT \d+\]"),
    re.compile(r"^\[SUBINTENT S\d+\]"),
    re.compile(r"^\[PLAN_ITEMS\]\[RAW_RESPONSE\]"),
    re.compile(r"^\[PLAN_ITEMS\]\[PARSED_KEYS\]"),
    re.compile(r"^\[DISCRETIZE\]\[RAW_RESPONSE\]"),
    re.compile(r"^\[BASELINE\]\[RAW OUTPUT\]"),
    re.compile(r"^\[BASELINE\]\[PARSED OUTPUT\]"),
    re.compile(r"^\[PIPELINE SUMMARY\]"),
    re.compile(r"^\[BASELINE SUMMARY\]"),
]

FIELD_LINES = [
    re.compile(r"^intents="),
    re.compile(r"^repeats="),
    re.compile(r"^temperature="),
    re.compile(r"^models="),
    re.compile(r"^elapsed_sec\s*:"),
    re.compile(r"^valid_json\s*:"),
    re.compile(r"^parse_error\s*:"),
    re.compile(r"^step_count\s*:"),
    re.compile(r"^parsed_ops\s*:"),
    re.compile(r"^total_subintents\s*:"),
    re.compile(r"^needs_human_count\s*:"),
    re.compile(r"^text\s*:"),
    re.compile(r"^classified_op\s*:"),
    re.compile(r"^confidence\s*:"),
    re.compile(r"^plan_items_count\s*:"),
    re.compile(r"^plan_steps_count\s*:"),
    re.compile(r"^needs_human\s*:"),
    re.compile(r"^\[PIPELINE SUMMARY\]"),
    re.compile(r"^\[BASELINE SUMMARY\]"),
]

DROP_LINES = [
    re.compile(r"^operation_score\s*:"),
    re.compile(r"^operation_note\s*:"),
    re.compile(r"^grounding_score\s*:"),
    re.compile(r"^grounding_note\s*:"),
    re.compile(r"^final_score_pct\s*:"),
    re.compile(r"^\[FINAL SCORE\]"),
    re.compile(r"^- operation_total"),
    re.compile(r"^- grounding_total"),
    re.compile(r"^- final_score_pct"),
    re.compile(r"^- total_subintents"),
    re.compile(r"^\[deterministic_plan\]"),
    re.compile(r"^ANTES DO SYSTEM$"),
    re.compile(r"^DEPOIS DO SYSTEM$"),
    re.compile(r"^teste da llm:"),
    re.compile(r"^_+$"),
    re.compile(r"^\{'devices': \d+, 'networks': \d+\}$"),
]

JSON_STARTERS = ("```json", "{", "[")


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            if "\x00" not in text:
                return text
            candidate = text.replace("\x00", "")
            if "[BENCHMARK" in candidate or "[PIPELINE]" in candidate:
                return candidate
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").replace("\x00", "")



def match_any(line: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(line) for p in patterns)



def capture_json_block(lines: list[str], start_index: int) -> tuple[list[str], int]:
    captured: list[str] = []
    i = start_index

    while i < len(lines) and not lines[i].strip():
        i += 1

    if i < len(lines) and lines[i].strip() == "```json":
        captured.append(lines[i])
        i += 1
        while i < len(lines):
            captured.append(lines[i])
            if lines[i].strip() == "```":
                i += 1
                return captured, i
            i += 1
        return captured, i

    brace_balance = 0
    seen_json = False
    while i < len(lines):
        stripped = lines[i].strip()
        if not seen_json and not stripped.startswith(JSON_STARTERS):
            break
        captured.append(lines[i])
        brace_balance += lines[i].count("{") - lines[i].count("}")
        seen_json = True
        i += 1
        if seen_json and brace_balance <= 0 and stripped.endswith(("}", "]")):
            break
    return captured, i



def extract_manual_view(raw_text: str) -> str:
    lines = raw_text.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if match_any(stripped, DROP_LINES):
            i += 1
            continue

        if match_any(stripped, SECTION_HEADERS):
            out.append(stripped)
            i += 1
            if stripped in {
                "[PLAN_ITEMS][RAW_RESPONSE]",
                "[DISCRETIZE][RAW_RESPONSE]",
                "[BASELINE][RAW OUTPUT]",
                "[BASELINE][PARSED OUTPUT]",
            }:
                block, i = capture_json_block(lines, i)
                out.extend([b.rstrip() for b in block])
            continue

        if match_any(stripped, FIELD_LINES):
            out.append(stripped)
            i += 1
            continue

        if not stripped:
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        i += 1

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically extract only the manual-audit-relevant parts of a benchmark log."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("filtered_run_log_manual_only.txt"))
    args = parser.parse_args()

    raw_text = read_text_auto(args.input)
    filtered_text = extract_manual_view(raw_text)
    args.output.write_text(filtered_text, encoding="utf-8")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("Done. Preserved sections are copied verbatim after decoding; the script only drops unrelated lines.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet

from run_topology import (
    FRR_BASE_DIR,
    JsonTopology,
    configure_interfaces,
    load_topology,
    router_name,
    start_frr,
)


EXIT_MARKER = "__TCC_COMMAND_EXIT_CODE__="
ERROR_PATTERNS = (
    "unknown command",
    "command incomplete",
    "ambiguous command",
    "no such file or directory",
    "cannot find device",
    "can't find device",
    "is not running",
    "not found",
    "rtnetlink answers:",
)


def load_model_commands(path: Path, case_ids: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        output = json.load(fp)

    test_outputs = output.get("test_outputs")
    if not isinstance(test_outputs, list):
        raise ValueError("Model output must contain a test_outputs array.")

    selected = []
    for case in test_outputs:
        if not isinstance(case, dict):
            continue

        case_id = str(case.get("case_id") or case.get("intent_id") or "")
        if case_ids and case_id not in case_ids:
            continue

        commands = case.get("actual_commands") or []
        for index, item in enumerate(commands, start=1):
            if not isinstance(item, dict):
                continue

            target = str(item.get("target_device") or "").strip()
            command = item.get("command")
            if not target or not isinstance(command, str) or not command.strip():
                continue

            selected.append(
                {
                    "case_id": case_id,
                    "intent_text": case.get("intent_text") or "",
                    "command_index": index,
                    "target_device": target,
                    "command": command.strip(),
                }
            )

    return selected


def mininet_node_name(target_device: str, topology: dict) -> str:
    device = (topology.get("devices") or {}).get(target_device)
    if not isinstance(device, dict):
        raise ValueError(f"Target device {target_device!r} does not exist in the topology.")
    return router_name(target_device) if device.get("type") == "router" else target_device


def execute_command(node, command: str) -> tuple[int | None, str, list[str]]:
    raw_output = node.cmd(f"{command}; printf '\\n{EXIT_MARKER}%s\\n' $?")
    match = re.search(rf"{re.escape(EXIT_MARKER)}(\d+)", raw_output)
    exit_code = int(match.group(1)) if match else None
    output = re.sub(rf"\n?{re.escape(EXIT_MARKER)}\d+\s*$", "", raw_output).strip()
    detected_errors = [
        pattern for pattern in ERROR_PATTERNS if pattern in output.lower()
    ]
    return exit_code, output, detected_errors


def apply_commands(net: Mininet, topology: dict, commands: list[dict]) -> list[dict]:
    results = []

    for sequence, item in enumerate(commands, start=1):
        result = {**item, "sequence": sequence}
        try:
            node_name = mininet_node_name(item["target_device"], topology)
            node = net.get(node_name)
            info(
                f"\n*** [{sequence}/{len(commands)}] {item['case_id']} "
                f"{node_name}: {item['command']}\n"
            )
            exit_code, output, detected_errors = execute_command(node, item["command"])
            result.update(
                {
                    "node_name": node_name,
                    "exit_code": exit_code,
                    "output": output,
                    "detected_errors": detected_errors,
                    "ok": exit_code == 0 and not detected_errors,
                }
            )
        except Exception as exc:
            result.update(
                {
                    "exit_code": None,
                    "output": "",
                    "detected_errors": [str(exc)],
                    "ok": False,
                }
            )
        results.append(result)

    return results


def write_report(path: Path, model_output: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for result in results if result.get("ok"))
    report = {
        "model_output": str(model_output),
        "summary": {
            "commands": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "results": results,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the Mininet/FRR lab and apply actual_commands from a model output."
    )
    parser.add_argument("--topology", type=Path, default=Path("/lab/topology.json"))
    parser.add_argument("--model-output", type=Path, default=Path("/lab/model_output.json"))
    parser.add_argument("--report", type=Path, default=Path("/results/model_command_report.json"))
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Apply only this case ID. May be passed more than once.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    setLogLevel(args.log_level)
    topology = load_topology(args.topology)
    commands = load_model_commands(args.model_output, set(args.case_id))
    if not commands:
        raise ValueError("No actual_commands were selected from the model output.")

    topo = JsonTopology(topology)
    net = Mininet(topo=topo, controller=None, autoSetMacs=True, link=TCLink)
    results = []

    try:
        info("*** Starting Mininet\n")
        net.start()
        configure_interfaces(net, topology)
        info("*** Starting FRR daemons per router\n")
        start_frr(net, topology)
        results = apply_commands(net, topology, commands)
    finally:
        info("\n*** Stopping Mininet\n")
        net.stop()
        if FRR_BASE_DIR.exists():
            shutil.rmtree(FRR_BASE_DIR)

    write_report(args.report, args.model_output, results)
    passed = sum(1 for result in results if result.get("ok"))
    failed = len(results) - passed
    print(f"Applied {len(results)} command(s): {passed} passed, {failed} failed.")
    print(f"Report: {args.report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import json
import re
import socket
import subprocess
import sys
import telnetlib
import time
from pathlib import Path
from typing import Any


ERROR_PATTERNS = (
    "% invalid input",
    "% incomplete command",
    "% ambiguous command",
    "% unknown command",
    "% bad mask",
    "% duplicate",
    "invalid input detected",
    "command rejected",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def load_model_commands(path: Path, case_ids: set[str]) -> list[dict]:
    output = load_json(path)
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


def merged_device_config(inventory: dict, target_device: str) -> dict:
    devices = inventory.get("devices") or {}
    device = devices.get(target_device)
    if not isinstance(device, dict):
        raise ValueError(f"Target device {target_device!r} does not exist in the Cisco inventory.")

    defaults = inventory.get("defaults") or {}
    merged = {**defaults, **device}
    if not merged.get("host"):
        raise ValueError(f"Target device {target_device!r} has no management host.")
    if not merged.get("port"):
        raise ValueError(f"Target device {target_device!r} has no management port.")
    return merged


def split_ios_command(command: str) -> list[str]:
    return [part.strip() for part in command.split(";") if part.strip()]


def detect_errors(output: str) -> list[str]:
    lowered = output.lower()
    return [pattern for pattern in ERROR_PATTERNS if pattern in lowered]


class CiscoSession:
    def send_lines(self, lines: list[str]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class CiscoTelnetSession(CiscoSession):
    def __init__(self, config: dict):
        self.config = config
        timeout = int(config.get("command_timeout") or 10)
        self.timeout = timeout
        self.conn = telnetlib.Telnet(
            str(config["host"]),
            int(config["port"]),
            timeout=timeout,
        )
        self._login()

    def _read_available(self, wait: float = 0.2) -> str:
        time.sleep(wait)
        chunks = []
        while True:
            try:
                data = self.conn.read_very_eager()
            except EOFError:
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
            time.sleep(0.05)
        return "".join(chunks)

    def _write_line(self, line: str) -> str:
        self.conn.write(line.encode("utf-8") + b"\n")
        return self._read_available()

    def _login(self) -> None:
        banner = self._read_available(0.5)
        username = str(self.config.get("username") or "")
        password = str(self.config.get("password") or "")
        enable_password = str(self.config.get("enable_password") or "")

        if "username" in banner.lower() and username:
            banner += self._write_line(username)
        if "password" in banner.lower() and password:
            banner += self._write_line(password)

        banner += self._write_line("")
        if enable_password:
            banner += self._write_line("enable")
            banner += self._write_line(enable_password)
        self.login_output = banner

    def send_lines(self, lines: list[str]) -> str:
        output = [getattr(self, "login_output", "")]
        for line in lines:
            output.append(self._write_line(line))
        return "".join(output)

    def close(self) -> None:
        try:
            self.conn.write(b"exit\n")
            self.conn.close()
        except OSError:
            pass


class CiscoParamikoSession(CiscoSession):
    def __init__(self, config: dict):
        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError("SSH mode requires the optional 'paramiko' package.") from exc

        self.config = config
        self.timeout = int(config.get("command_timeout") or 10)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=str(config["host"]),
            port=int(config["port"]),
            username=str(config.get("username") or ""),
            password=str(config.get("password") or ""),
            look_for_keys=False,
            allow_agent=False,
            timeout=self.timeout,
        )
        self.shell = self.client.invoke_shell()
        time.sleep(0.5)
        self.login_output = self._read_available()
        enable_password = str(config.get("enable_password") or "")
        if enable_password:
            self._write_line("enable")
            self._write_line(enable_password)

    def _read_available(self, wait: float = 0.2) -> str:
        time.sleep(wait)
        chunks = []
        started = time.monotonic()
        while time.monotonic() - started < self.timeout:
            if not self.shell.recv_ready():
                break
            chunks.append(self.shell.recv(65535).decode("utf-8", errors="replace"))
            time.sleep(0.05)
        return "".join(chunks)

    def _write_line(self, line: str) -> str:
        self.shell.send(line + "\n")
        return self._read_available()

    def send_lines(self, lines: list[str]) -> str:
        output = [getattr(self, "login_output", "")]
        for line in lines:
            output.append(self._write_line(line))
        return "".join(output)

    def close(self) -> None:
        self.client.close()


def open_session(config: dict) -> CiscoSession:
    protocol = str(config.get("protocol") or "telnet").lower()
    if protocol == "telnet":
        return CiscoTelnetSession(config)
    if protocol == "ssh":
        return CiscoParamikoSession(config)
    raise ValueError(f"Unsupported protocol {protocol!r}. Use 'telnet' or 'ssh'.")


def apply_commands(inventory: dict, commands: list[dict], write_memory: bool) -> list[dict]:
    results = []
    session_cache: dict[str, CiscoSession] = {}

    try:
        for sequence, item in enumerate(commands, start=1):
            result = {**item, "sequence": sequence}
            target = item["target_device"]
            try:
                config = merged_device_config(inventory, target)
                session = session_cache.get(target)
                if session is None:
                    session = open_session(config)
                    session_cache[target] = session

                lines = split_ios_command(item["command"])
                output = session.send_lines(lines)
                errors = detect_errors(output)
                result.update(
                    {
                        "host": config.get("host"),
                        "port": config.get("port"),
                        "protocol": config.get("protocol"),
                        "sent_lines": lines,
                        "output": output.strip(),
                        "detected_errors": errors,
                        "ok": not errors,
                    }
                )
            except (OSError, socket.timeout, RuntimeError, ValueError, EOFError) as exc:
                result.update(
                    {
                        "output": "",
                        "detected_errors": [str(exc)],
                        "ok": False,
                    }
                )
            results.append(result)

        if write_memory:
            for target, session in session_cache.items():
                output = session.send_lines(["write memory"])
                errors = detect_errors(output)
                results.append(
                    {
                        "case_id": "__write_memory__",
                        "target_device": target,
                        "command": "write memory",
                        "sent_lines": ["write memory"],
                        "output": output.strip(),
                        "detected_errors": errors,
                        "ok": not errors,
                    }
                )
    finally:
        for session in session_cache.values():
            session.close()

    return results


def write_report(path: Path, model_output: Path, inventory_path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for result in results if result.get("ok"))
    report = {
        "model_output": str(model_output),
        "inventory": str(inventory_path),
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
        description="Apply Cisco IOS actual_commands from a model output to emulated routers over Telnet or SSH."
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("cisco_lab/results/model_command_report_cisco.json"))
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Apply only this case ID. May be passed more than once.",
    )
    parser.add_argument("--write-memory", action="store_true")
    args = parser.parse_args()

    inventory = load_json(args.inventory)
    commands = load_model_commands(args.model_output, set(args.case_id))
    if not commands:
        raise ValueError("No actual_commands were selected from the model output.")

    results = apply_commands(inventory, commands, write_memory=args.write_memory)
    write_report(args.report, args.model_output, args.inventory, results)
    passed = sum(1 for result in results if result.get("ok"))
    failed = len(results) - passed
    print(f"Applied {len(results)} Cisco command(s): {passed} passed, {failed} failed.")
    print(f"Report: {args.report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

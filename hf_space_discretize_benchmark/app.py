import os
import subprocess
from pathlib import Path

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

app = FastAPI()

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "Ic-llmToNetworkConfig"
BENCH_DIR = PROJECT_DIR / "ibn_langgraph"
OUTPUTS_DIR = BENCH_DIR / "outputs"


def latest_report() -> Path | None:
    if not OUTPUTS_DIR.exists():
        return None
    reports = sorted(
        OUTPUTS_DIR.rglob("discretize_unit_comparison_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


@app.get("/", response_class=HTMLResponse)
def home():
    report = latest_report()
    report_link = '<p>No report yet.</p>'
    if report:
        report_link = f'<p><a href="/report">Open latest report</a><br><code>{report}</code></p>'

    return f"""
    <html>
      <body style="font-family: system-ui; max-width: 920px; margin: 40px auto;">
        <h1>Discretize Benchmark Space</h1>
        <p>Use <code>/run</code> to start the benchmark. Keep the browser tab open while it runs.</p>
        <p><a href="/run">Run quick benchmark (3 intents)</a></p>
        <p><a href="/run?limit=15&max_new_tokens=512">Run full benchmark (15 intents)</a></p>
        {report_link}
      </body>
    </html>
    """


@app.get("/run", response_class=PlainTextResponse)
def run_benchmark(
    limit: int = Query(default=3, ge=1, le=15),
    max_new_tokens: int = Query(default=512, ge=128, le=2048),
):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "Missing HF_TOKEN secret. Add it in Space Settings > Variables and secrets.\n"
    if not BENCH_DIR.exists():
        return f"Missing benchmark directory: {BENCH_DIR}\n"

    preflight_lines = []
    headers = {"Authorization": f"Bearer {hf_token}"}
    try:
        whoami = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers,
            timeout=30,
        )
        preflight_lines.append(f"whoami_status={whoami.status_code}")
        if whoami.ok:
            payload = whoami.json()
            preflight_lines.append(f"whoami_user={payload.get('name')}")
            auth = payload.get("auth") or {}
            access_token = auth.get("accessToken") or {}
            preflight_lines.append(f"token_role={access_token.get('role')}")
    except Exception as exc:
        preflight_lines.append(f"whoami_error={exc}")

    try:
        llama_cfg = requests.get(
            "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json",
            headers=headers,
            timeout=30,
        )
        preflight_lines.append(f"llama_config_status={llama_cfg.status_code}")
        if not llama_cfg.ok:
            return "\n".join(preflight_lines) + "\n\nHF_TOKEN cannot access the gated Llama base model from this Space.\n"
    except Exception as exc:
        return "\n".join(preflight_lines) + f"\n\nllama_config_error={exc}\n"

    cmd = [
        "python3",
        "grafo_final.py",
        "--compare-local-discretize-ft",
        "--limit-intents",
        str(limit),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    proc = subprocess.run(
        cmd,
        cwd=str(BENCH_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=7200,
    )
    report = latest_report()
    suffix = f"\n\nLatest report: {report}\n" if report else "\n\nNo report file found.\n"
    return proc.stdout + suffix


@app.get("/report", response_class=PlainTextResponse)
def report():
    path = latest_report()
    if not path:
        return "No report found yet.\n"
    return path.read_text(encoding="utf-8")

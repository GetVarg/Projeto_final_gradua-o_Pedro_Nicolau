import argparse
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Dict, Optional

from langgraph.graph import StateGraph, END

from main import IBNState
import no_grafo_tcc as tcc_nodes
from no_grafo_tcc import HFRouterChat

node_discretize_intent = tcc_nodes.node_discretize_intent
node_extract_subintent_entities = tcc_nodes.node_extract_subintent_entities
node_context = tcc_nodes.node_context
node_planner = tcc_nodes.node_planner
node_argument_resolver = tcc_nodes.node_argument_resolver
node_compile_commands = tcc_nodes.node_compile_commands

DEBUG_TRACE_NODES = False
DEBUG_EVAL_REPORT = True

BENCHMARK_REPEATS = 2
BENCHMARK_TEMPERATURE = 0.0

TOPOLOGY_PATH = "dataset/topologias_convertidas/gabriel/10/0.json"
GOLDEN_COMMANDS_PATH = "golden_intents_topology_gabriel_10_0.json"
CORRECTED_BENCHMARK_PATH = "../testTcc/benchmark_final_corrigido.json"
SBRC_BENCHMARK_PATH = "../testTcc/benchmark_test_sbrc.json"

LLAMA_1B_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_8B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
QWEN_4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_35B_MODEL = "Qwen/Qwen3.5-35B-A3B"

# Backwards-compatible alias for old experiments that referenced this name.
LLAMA_3B_MODEL = LLAMA_1B_MODEL

RUN_PIPELINE_BENCHMARK = True
RUN_DIRECT_BASELINE_BENCHMARK = False

ACTIVE_DOWNSTREAM_MODEL_OVERRIDE: str | None = None


def _load_golden_command_cases(path: str = GOLDEN_COMMANDS_PATH) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if not p.exists():
        return {}

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] failed to load golden command cases from {p}: {exc}")
        return {}

    out = {}
    for case in data.get("cases") or []:
        if not isinstance(case, dict):
            continue
        intent = case.get("intent")
        if isinstance(intent, str) and intent.strip():
            out[intent] = case
    return out


def resolve_existing_path(path: str | Path) -> Path:
    candidates = []
    p = Path(path)
    if p.is_absolute():
        candidates.append(p)
    else:
        script_dir = Path(__file__).resolve().parent
        candidates.extend([
            Path.cwd() / p,
            script_dir / p,
            script_dir.parent / p,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"File not found: {path}. Tried: {tried}")


def load_corrected_benchmark_cases(path: str | Path, limit: int | None = None) -> list[dict]:
    benchmark_path = resolve_existing_path(path)
    with benchmark_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Benchmark file must contain a JSON list: {benchmark_path}")

    cases = []
    for index, raw_case in enumerate(data, start=1):
        if limit is not None and len(cases) >= limit:
            break
        if not isinstance(raw_case, dict):
            raise ValueError(f"Benchmark case #{index} must be a JSON object.")

        intent = raw_case.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError(f"Benchmark case #{index} has no valid intent.")

        expected_commands = raw_case.get("expected_commands", raw_case.get("expected_comands"))
        if not isinstance(expected_commands, list):
            raise ValueError(f"Benchmark case {raw_case.get('id') or index} has no expected_commands list.")

        case = dict(raw_case)
        case["intent"] = intent.strip()
        case["expected_commands"] = expected_commands
        case["_source"] = str(benchmark_path)
        cases.append(case)

    return cases


def filter_benchmark_cases_by_id(cases: list[dict], case_ids: list[str]) -> list[dict]:
    wanted = [case_id.strip().upper() for case_id in case_ids if case_id.strip()]
    by_id = {
        str(case.get("id", "")).strip().upper(): case
        for case in cases
        if str(case.get("id", "")).strip()
    }
    missing = [case_id for case_id in wanted if case_id not in by_id]
    if missing:
        raise ValueError(f"Benchmark case IDs not found: {missing}")
    return [by_id[case_id] for case_id in wanted]


def slice_benchmark_cases_by_range(
    cases: list[dict],
    start_case: int | None = None,
    end_case: int | None = None,
) -> list[dict]:
    if start_case is None and end_case is None:
        return cases

    start = 1 if start_case is None else start_case
    end = len(cases) if end_case is None else end_case

    if start < 1:
        raise ValueError("--start-case must be >= 1.")
    if end < 1:
        raise ValueError("--end-case must be >= 1.")
    if start > end:
        raise ValueError(f"--start-case must be <= --end-case, got {start} > {end}.")
    if start > len(cases):
        raise ValueError(f"--start-case {start} is beyond the number of cases ({len(cases)}).")

    return cases[start - 1:end]


def build_intents_da_ic_cases(path: str | Path, limit: int | None = None) -> list[dict]:
    benchmark_cases = load_corrected_benchmark_cases(path, limit=None)
    by_intent = {
        case["intent"]: case
        for case in benchmark_cases
    }

    cases = []
    missing = []
    for intent in INTENTS_DA_IC:
        if limit is not None and len(cases) >= limit:
            break
        case = by_intent.get(intent)
        if case is None:
            missing.append(intent)
            continue
        cases.append(case)

    if missing:
        preview = missing[:3]
        raise ValueError(
            "Some INTENTS_DA_IC entries were not found in the SBRC benchmark file. "
            f"missing_count={len(missing)} preview={preview}"
        )

    return cases


def register_expected_cases(cases: list[dict]) -> None:
    for case in cases:
        intent = case.get("intent")
        if isinstance(intent, str) and intent.strip():
            EXPECTED_BY_INTENT[intent] = case


def set_ollama_model(model_name: str) -> None:
    tcc_nodes.llm = tcc_nodes.HFRouterChat(
        model=model_name,
        temperature=BENCHMARK_TEMPERATURE,
    )


def get_ollama_model_name() -> str:
    return getattr(tcc_nodes.llm, "model", None) or tcc_nodes.DEFAULT_OLLAMA_MODEL


def node_switch_to_downstream_model(state: IBNState) -> IBNState:
    if ACTIVE_DOWNSTREAM_MODEL_OVERRIDE:
        set_ollama_model(ACTIVE_DOWNSTREAM_MODEL_OVERRIDE)
        state.setdefault("work", {})["downstream_model"] = ACTIVE_DOWNSTREAM_MODEL_OVERRIDE
    return state


def _topo_compact_with_ips(topo: dict) -> dict:
    devices = topo.get("devices") or {}
    networks = topo.get("networks") or {}
    links = topo.get("links") or topo.get("edges") or []
    return {
        "devices": devices,
        "networks": networks,
        "links": links,
    }

BASELINE_SYSTEM_PROMPT = """
You are a baseline network intent translator.

Convert one user intent directly into final CLI commands.
Do not explain your reasoning.
Return JSON only.

This is a baseline, not a full planner.
Try to identify the main network action and emit the concrete Linux/FRRouting command strings.

Prefer a compact answer.
Use the topology to resolve explicit topology references, peer IPs, connected interfaces, and LAN subnets.
If the intent contains more than one action, return commands in execution order.
If a required command value is missing or ambiguous, return no commands and set needs_human=true.

Use a structure like this:
{
  "intent_summary": "short summary",
  "commands": [
    {
      "target_device": "router/device id where the command should run, such as 0",
      "command": "fully rendered command string"
    }
  ],
  "needs_human": false,
  "missing": [],
  "warnings": []
}

Guidelines:
- Use fully rendered commands, not templates.
- Each command must be an object with target_device and command.
- target_device must be the topology device id where the command should run, for example "0" for router 0.
- Do not include markdown or shell prompts.
- Do not invent values that are absent from the intent.
- Do not use a destination prefix as a next-hop IP.
- For static routes with no next-hop, interface, or blackhole target, set needs_human=true and commands=[].
- For MTU and description on FRR interfaces, use vtysh commands.
- For interface up/down, use ip link set.
- For IPv4 forwarding, use sysctl -w net.ipv4.ip_forward=1.
- For IPv6 forwarding, use sysctl -w net.ipv6.conf.all.forwarding=1.
- Keep notes small.
""".strip()


# BENCHMARK_INTENTS = [
#     "On router 0, set BGP MED to 100 for neighbor 10.0.2.2 and allow AS-path multipath",
#     "Configure a static ARP entry on router 0 to map the IP address 10.0.0.2 to its MAC address 02:00:00:00:00:02",
#     "Configure router 7 with a static route to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50",
#     "On router 9, add IPv6 route 3000::/64 via 9-eth0 and enable IPv6 forwarding",
#     "Create a floating static route 0.0.0.0/0 via 10.0.1.2 distance 250 on router 0",
#     "On router 0, establish BGP peer-group IBGP, set remote-as 65000, and associate neighbor 10.0.1.2 with it",
#     "On router 2, add a static route to network 172.16.9.0/24 using interface 2-eth1 as egress",
#     "Configure BGP on router 7 with local AS 65001 and establish a neighbor relationship with 10.0.9.2 using remote AS 65002",
#     "Enable OSPF process 10 on router 0 and advertise the networks 10.0.0.0/30 and 10.0.1.0/30 in area 0",
#     "Set the description of interface 7-eth1 on router 7 to 'BACKBONE_LINK_TO_9' and enable the interface",
#     "Configure an iptables rule on router 0 to drop all ICMP traffic originating from the network 10.0.2.0/30",
#     "On router 9, assign the IPv6 address 2001:db8:1::1/64 to interface 9-eth0",
#     "Find the neighbor of 0-eth2 and configure a static route to 172.16.9.0/24 via that neighbor's IP",
#     "On router 7, enable IP forwarding and set the MTU of interface 7-eth1 to 9000 bytes",
#     "From router 0, perform a traceroute to 10.0.9.2 and log the output to a file named 'path_check.log'",
#     "Configure a BGP peer-group named 'INTERNAL' on router 2 with remote-as 65001 and bind neighbor 10.0.6.2 to it",
#     "On router 7, configure a static route to 172.16.9.0/24 via 10.0.9.2 with administrative distance 20",
#     "On router 9, assign the IPv4 address 172.16.9.254/24 to interface 9-eth2",
#     "Apply a rate limit of 100Mbps on interface 0-eth2 for all outbound TCP traffic",
#     "Remove all static routes on router 7 that point to the 172.16.0.0/24 prefix",
# ]

# BENCHMARK_INTENTS = [
#     "On router 9, add an IPv6 route to 3000::/64 through interface 9-eth0 and enable IPv6 forwarding.",
#     "Find the neighbor of interface 0-eth2 and configure a static route on router 0 to 172.16.9.0/24 via that neighbor's IP.",
#     "On router 7, configure a static route to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
#     "Create a floating default route on router 0 via 10.0.1.2 with distance 250.",
#     "Configure a BGP peer-group named IBGP on router 0, set remote-as 65000, and bind neighbor 10.0.1.2 to it.",
#     "Enable IP forwarding on router 7 and set the MTU of interface 7-eth1 to 9000 bytes.",
#     "On router 9, assign the IPv6 address 2001:db8:1::1/64 to interface 9-eth0.",
#     "Apply a rate limit of 100Mbps on interface 0-eth2 for outbound TCP traffic.",
#     "Configure a static ARP entry on router 0 mapping 10.0.0.2 to MAC address 02:00:00:00:00:02.",
#     "Remove all static routes on router 7 that point to prefix 172.16.0.0/24."
# ]

BENCHMARK_INTENTS_50 = [

# =========================
# STATIC ROUTING (1-10)
# =========================
"On router 9, add an IPv6 route to 3000::/64 through interface 9-eth0 and enable IPv6 forwarding.",
"Find the neighbor of interface 0-eth2 and configure a static route on router 0 to 172.16.9.0/24 via that neighbor's IP.",
"On router 7, configure a static route to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
"Create a floating default route on router 0 via 10.0.1.2 with distance 250.",
"Configure on router 4 a static route to 172.16.8.0/24 via directly connected neighbor 8.",
"On router 1, add a route to 172.16.6.0/24 via router 3.",
"Configure router 5 to reach 172.16.7.0/24 using next-hop 10.0.5.1.",
"On router 2, install a default route via router 5.",
"Remove all static routes on router 7 that point to prefix 172.16.0.0/24.",
"Replace the default route on router 0 so traffic exits through router 9.",

# =========================
# INTERFACE / L3 CONFIG (11-20)
# =========================
"Enable IP forwarding on router 7 and set the MTU of interface 7-eth1 to 9000 bytes.",
"On router 9, assign the IPv6 address 2001:db8:1::1/64 to interface 9-eth0.",
"Set interface 4-eth2 MTU to 1600 bytes.",
"Shutdown interface 3-eth1 and then bring it back up.",
"Enable IPv6 forwarding on router 2.",
"Assign IPv6 address 2001:db8:2::1/64 to interface 2-eth1.",
"Configure interface 0-eth3 description as LAN_USERS.",
"Bring down interface 8-eth0.",
"Bring up interface 8-eth0.",
"Set MTU 9216 on interface 5-eth1 and enable IP forwarding on router 5.",

# =========================
# ARP / NEIGHBOR / L2 (21-25)
# =========================
"Configure a static ARP entry on router 0 mapping 10.0.0.2 to MAC address 02:00:00:00:00:02.",
"On router 7, create a static ARP entry for 10.0.9.2 with MAC 02:00:00:00:09:02.",
"Delete all static ARP entries on router 0.",
"Find the neighbor connected to 4-eth0 and configure a static ARP for its IP.",
"Configure a static route on router 1 to reach the LAN subnet of router 6 using the IP of the directly connected neighbor on the path from router 1 to router 6.",

# =========================
# ACL / SECURITY (26-35)
# =========================
"Block inbound SSH traffic on interface 0-eth3.",
"Allow HTTP traffic outbound on interface 0-eth3 and deny everything else.",
"Block ICMP traffic from 172.16.7.0/24 on router 2.",
"Permit only TCP port 443 inbound on interface 9-eth2.",
"Deny traffic from host 172.16.4.10 reaching router 0 LAN.",
"Create ACL on router 1 allowing SSH from 172.16.1.10 only.",
"Apply an ACL on router 5 blocking telnet traffic outbound.",
"Permit ICMP echo requests on router 8 LAN interface.",
"Block UDP traffic outbound on interface 7-eth2.",
"Allow DNS traffic and block all other UDP on router 3.",

# =========================
# QoS / TRAFFIC CONTROL (36-40)
# =========================
"Apply a rate limit of 100Mbps on interface 0-eth2 for outbound TCP traffic.",
"Limit outbound traffic on 7-eth1 to 50Mbps.",
"Police inbound traffic on 4-eth3 to 20Mbps.",
"Apply priority treatment for SSH traffic on interface 5-eth2.",
"Limit UDP traffic on router 9 LAN interface to 10Mbps.",

# =========================
# BGP / DYNAMIC ROUTING (41-45)
# =========================
"Configure a BGP peer-group named IBGP on router 0, set remote-as 65000, and bind neighbor 10.0.1.2 to it.",
"On router 5, create BGP process 65000 and peer with router 0.",
"Configure router 2 with BGP ASN 65100 and advertise network 172.16.2.0/24.",
"Create peer-group EDGE on router 4 with remote-as 65200 and attach neighbor 10.0.8.2.",
"Remove BGP neighbor 10.0.1.2 from router 0.",

# =========================
# MULTI-STEP / COMPOSITE INTENTS (46-50)
# =========================
"Enable IP forwarding on router 9, assign IPv6 2001:db8:9::1/64 to 9-eth1, and add default IPv6 route via 9-eth0.",
"On router 0, create a backup route to 172.16.8.0/24 via router 4 with distance 200 and primary via router 9 with distance 10.",
"Block SSH from 172.16.7.0/24 on router 0 and rate-limit HTTP traffic on 0-eth3 to 30Mbps.",
"Find the neighbor on interface 2-eth1, then configure a route from router 2 to 172.16.9.0/24 through that neighbor.",
"Configure router 1 to advertise 172.16.1.0/24 in BGP, enable forwarding, and set MTU 2000 on 1-eth1."

]

BENCHMARK_INTENTS_STRATEGIC = [

# TOPOLOGY GROUNDING
"Find the neighbor of interface 0-eth2 and configure a route to 172.16.9.0/24 through it.",
"Use the device connected to 2-eth1 as next-hop for reaching 172.16.7.0/24.",
"Configure a backup static route on router 4 to 172.16.1.0/24 using the peer of interface 4-eth0 as next-hop."
"Discover who is linked to 7-eth1 and configure a route to 172.16.9.0/24 through it.",
"Use the neighbor attached to 1-eth0 as next-hop to reach router 6 LAN.",

# MULTI STEP
"Enable IPv6 forwarding on router 9 and add an IPv6 route to 3000::/64 through 9-eth0.",
"Bring up interface 8-eth0, assign IPv4 address 10.0.8.10/30, then enable forwarding.",
"Create BGP peer-group IBGP on router 0 and bind neighbor 10.0.1.2.",
"Enable forwarding on router 5 and set MTU 9216 on 5-eth1.",
"Bring interface 3-eth1 up and configure description BACKBONE_TO_6.",

# ARGUMENT RICH
"Configure a static route on router 7 to 172.16.0.0/24 via 10.0.6.1 with distance 50.",
"Set MED 100 for BGP neighbor 10.0.2.2 and enable AS-path multipath on router 0.",
"Create static ARP entry on router 0 mapping 10.0.0.2 to 02:00:00:00:00:02 on 0-eth0.",
"Assign IPv6 address 2001:db8:9::1/64 to 9-eth1 and enable forwarding.",
"Configure backup default route on router 0 via 10.0.1.2 with distance 250.",

# OPERATION CHOICE
"Remove the route to 172.16.0.0/24 from router 7.",
"Disable ARP on interface 0-eth2.",
"Enable IPv4 forwarding on router 7.",
"Shut down interface 8-eth0.",
"Bind neighbor 10.0.8.2 to peer-group EDGE on router 4."

]

BENCHMARK_INTENTS_STRATEGIC = [

    # TOPOLOGY GROUNDING
    "Configure a static route on router 0 to 172.16.9.0/24 using the neighbor of interface 0-eth2 as next-hop.",
    "Configure a static route on router 2 to 172.16.7.0/24 using the device connected to 2-eth1 as next-hop.",
    "Configure a backup static route on router 4 to 172.16.1.0/24 using the peer of interface 4-eth0 as next-hop.",
    "Configure a static route on router 7 to 172.16.9.0/24 using the device linked to 7-eth1 as next-hop.",
    "Configure a static route on router 1 to reach router 6 LAN using the neighbor attached to 1-eth0 as next-hop.",

    # MULTI STEP
    "Enable IPv6 forwarding on router 9 and add an IPv6 route to 3000::/64 through 9-eth0.",
    "Bring up interface 8-eth0, assign IPv4 address 10.0.8.10/30, then enable forwarding on router 8.",
    "Create BGP peer-group IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
    "Enable forwarding on router 5 and set MTU 9216 on 5-eth1.",
    "Bring interface 3-eth1 up and configure description BACKBONE_TO_6.",

    # ARGUMENT RICH
    "Configure a static route on router 7 to 172.16.0.0/24 via 10.0.6.1 with distance 50.",
    "Set MED 100 for BGP neighbor 10.0.2.2 and enable AS-path multipath on router 0.",
    "Create static ARP entry on router 0 mapping 10.0.0.2 to 02:00:00:00:00:02 on 0-eth0.",
    "Assign IPv6 address 2001:db8:9::1/64 to 9-eth1 and enable forwarding on router 9.",
    "Configure backup default route on router 0 via 10.0.1.2 with distance 250.",

    # OPERATION CHOICE
    "Remove the route to 172.16.0.0/24 from router 7.",
    "Disable ARP on interface 0-eth2.",
    "Enable IPv4 forwarding on router 7.",
    "Shut down interface 8-eth0.",
    "Bind neighbor 10.0.8.2 to peer-group EDGE on router 4."
]

BENCHMARK_INTENTS_STRATEGIC = [
# --- 30 INTENTS BASE (Refinados para evitar ambiguidades) ---
    "Configure a static route on router 0 to 172.16.9.0/24 using the IP address of the neighbor connected to interface 0-eth2 as next-hop.",
    "Configure a static route on router 2 to 172.16.7.0/24 using the IP of the device connected to 2-eth1 as next-hop.",
    "Configure a backup static route on router 4 to 172.16.1.0/24 using the IP address of the peer of 4-eth0 as next-hop.",
    "Configure a static route on router 7 to 172.16.9.0/24 using the IP of the device linked to 7-eth1 as next-hop.",
    "Configure a static route on router 1 to reach router 6 LAN using the IP of the neighbor attached to 1-eth0 as next-hop.",
    "Enable IPv6 forwarding on router 9 and add an IPv6 route to 3000::/64 through interface 9-eth0.",
    "Bring up interface 8-eth0, assign the IPv4 address 10.0.8.10/30 to it, and then enable IPv4 forwarding on router 8.",
    "Create a BGP peer-group named IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
    "Enable IPv4 forwarding on router 5 and set the MTU of interface 5-eth1 to 9216 bytes.",
    "Bring interface 3-eth1 administratively up and configure its description as BACKBONE_TO_6.",
    "Configure a static route on router 7 to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
    "Set MED to 100 for BGP neighbor 10.0.2.2 and enable AS-path multipath on router 0.",
    "Configure a static route on router 3 to 172.16.4.0/24 via 10.0.4.2.",
    "Configure BGP local-preference to 200 for peer-group EDGE on router 4.",
    "Bring down interface 1-eth1 and remove its IP address.",
    "Remove the static route pointing to 172.16.0.0/24 from router 7.",
    "Disable ARP encapsulation on interface 0-eth2 of router 0.",
    "Enable IPv4 forwarding globally on router 7.",
    "Administratively shut down interface 8-eth0 on router 8.",
    "Bind neighbor 10.0.8.2 to the BGP peer-group EDGE on router 4.",
    "On router 9, add an IPv6 route to 3000::/64 through interface 9-eth0 and enable IPv6 forwarding.",
    "Find the IP of the neighbor connected to interface 0-eth2, and configure a static route on router 0 to 172.16.9.0/24 via that IP.",
    "On router 7, configure a static route to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
    "Create a floating default route (0.0.0.0/0) on router 0 via 10.0.1.2 with administrative distance 250.",
    "Configure a BGP peer-group named IBGP on router 0, set its remote-as to 65000, and bind neighbor 10.0.1.2 to it.",
    "Enable IP forwarding on router 7 and set the MTU of interface 7-eth1 to 9000 bytes.",
    "On router 9, assign the IPv6 address 2001:db8:1::1/64 to interface 9-eth0.",
    "Apply a rate limit of 100Mbps on interface 0-eth2 of router 0 specifically for outbound TCP traffic.",
    "Configure a static ARP entry on router 0 mapping IP 10.0.0.2 to MAC address 02:00:00:00:00:02.",
    "Remove all static routes on router 7 that point to prefix 172.16.0.0/24.",
    
    # --- 20 INTENTS NOVOS (Foco em Pesquisa, Limites de Raciocínio Topológico e Múltiplos Passos) ---
    "Configure a static route on router 1 to reach the LAN subnet of router 5, using the IP of the neighbor connected to 1-eth1 as the gateway.",
    "Find the router connected to interface 0-eth1, and configure a static route on that remote router back to 172.16.0.0/24 via 10.0.1.1.",
    "Apply a 50Mbps rate limit for inbound UDP traffic on the specific interface of router 4 that connects to router 0.",
    "On router 2, create a BGP peer-group named TRANSIT, set remote-as 64512, and set the BGP authentication password to 'SeCrEt' for this group.",
    "Set the local-preference to 150 for peer-group IBGP on router 0, and configure MED to 50 for neighbor 10.0.2.2.",
    "Apply a route-map named BLOCK_DEFAULT in the inbound direction for BGP neighbor 10.0.1.2 on router 0.",
    "Remove the BGP peer-group EDGE from router 4 and delete all associated static routes pointing to 172.16.1.0/24.",
    "Add a static route on router 6 to 172.16.8.0/24 via 10.0.6.2, and create a backup route to the exact same destination via 10.0.7.2 with administrative distance 200.",
    "Disable ARP, shut down interface 7-eth0, and remove its IPv4 address on router 7.",
    "Enable dual-stack forwarding on router 3, then assign IPv4 10.0.3.1/30 and IPv6 2001:db8:3::1/64 to interface 3-eth0.",
    "Change the IP address of 1-eth0 on router 1 to 10.0.10.1/30 and ensure the interface is administratively up.",
    "Configure a blackhole route on router 9 for the prefix 10.100.0.0/16 by pointing it to the Null0 interface.",
    "Limit inbound ICMP traffic to 5Mbps on interface 2-eth0 of router 2.",
    "Delete the static ARP entry for 10.0.0.2 on router 0, and clear the description of interface 0-eth0.",
    "Disable IPv6 forwarding globally on router 4, but ensure IPv4 forwarding remains enabled.",
    "Assign the IP address 10.255.255.1/32 to the loopback interface on router 1.",
    "Set the description of interface 0-eth2 on router 0 to TO_ROUTER_9 and set the description of 9-eth0 on router 9 to TO_ROUTER_0.",
    "Set the OSPF cost of interface 5-eth0 on router 5 to 100 and configure its description as PRIMARY_LINK.",
    "Enable Jumbo frames by setting the MTU to 9216 on interface 8-eth0 of router 8, and restart the interface by bringing it down then up.",
    "Find the IPv4 network CIDR assigned to interface 3-eth1 and configure a static route on router 0 pointing to that network via 10.0.1.2."
]

BENCHMARK_DIAGNOSTIC_CASES = [
    {
        "id": "D01",
        "target_node": "discretize",
        "case_type": "positive",
        "intent": "Configure a static route on router 0 to 172.16.9.0/24 via the IP of the peer connected to 0-eth2.",
        "expected_ops": ["static_route_add_frr"],
        "expected_min_subintents": 1,
        "expected_max_subintents": 1,
    },
    {
        "id": "D02",
        "target_node": "discretize",
        "case_type": "compositional",
        "intent": "Enable IPv4 forwarding on router 3, set MTU 9000 on interface 3-eth1, and shut down interface 3-eth2.",
        "expected_ops": ["ip_forward_enable", "interface_attribute_config", "interface_admin_config"],
        "expected_min_subintents": 3,
    },
    {
        "id": "D03",
        "target_node": "discretize",
        "case_type": "lookup_auxiliary",
        "intent": "Find the IP of the neighbor connected to 1-eth0 and use it as the next-hop for a static route to 172.16.6.0/24 on router 1.",
        "expected_ops": ["static_route_add_frr"],
        "expected_min_subintents": 1,
        "expected_max_subintents": 1,
    },
    {
        "id": "D04",
        "target_node": "extract_entities",
        "case_type": "positive_explicit",
        "intent": "Configure a static route on router 7 to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
        "expected_ops": ["static_route_add_frr"],
        "expected_bound_args": ["dst_cidr", "next_hop_ip", "administrative_distance"],
    },
    {
        "id": "D05",
        "target_node": "extract_entities",
        "case_type": "structured_relation",
        "intent": "Set the MTU to 9216 on the interface of router 4 that connects to router 0.",
        "expected_ops": ["interface_attribute_config"],
        "expected_bound_args": ["interface", "mtu"],
    },
    {
        "id": "D06",
        "target_node": "extract_entities",
        "case_type": "semantic_value",
        "intent": "Assign IPv4 address 10.0.10.1/30 to interface 1-eth0 on router 1.",
        "expected_ops": ["interface_l3_address_config"],
        "expected_bound_args": ["interface", "address_family"],
    },
    {
        "id": "D07",
        "target_node": "context",
        "case_type": "negative_missing_entity",
        "intent": "Set MTU 9000 on interface 4-eth999 of router 4.",
        "expected_ops": ["interface_attribute_config"],
        "expected_needs_human": True,
    },
    {
        "id": "D08",
        "target_node": "context",
        "case_type": "peer_ip_resolution",
        "intent": "Configure a static route on router 1 to 172.16.5.0/24 using the IP of the neighbor connected to 1-eth1.",
        "expected_ops": ["static_route_add_frr"],
        "expected_bound_args": ["dst_cidr", "next_hop_ip"],
    },
    {
        "id": "D09",
        "target_node": "planner",
        "case_type": "multi_operation",
        "intent": "Create a BGP peer-group named IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
        "expected_ops": ["bgp_peer_group_config", "bgp_neighbor_peer_group_bind"],
    },
    {
        "id": "D10",
        "target_node": "planner",
        "case_type": "operation_choice",
        "intent": "Configure a blackhole route on router 9 for prefix 10.100.0.0/16 by pointing it to Null0.",
        "expected_ops": ["static_route_blackhole_add"],
    },
    {
        "id": "D11",
        "target_node": "planner",
        "case_type": "unsupported_catalog_gap",
        "intent": "Configure BGP authentication password SeCrEt for peer-group TRANSIT on router 2.",
        "expected_ops": [],
        "expected_needs_human": True,
    },
    {
        "id": "D12",
        "target_node": "argument_resolver",
        "case_type": "lan_subnet_derived",
        "intent": "Configure a static route on router 1 to reach the LAN subnet of router 5 using the IP of the neighbor connected to 1-eth1 as gateway.",
        "expected_ops": ["static_route_add_frr"],
        "expected_bound_args": ["dst_cidr", "next_hop_ip"],
    },
    {
        "id": "D13",
        "target_node": "argument_resolver",
        "case_type": "missing_required_arg",
        "intent": "Configure a static route on router 2 to 172.16.7.0/24.",
        "expected_ops": ["static_route_add_frr"],
        "expected_needs_human": True,
    },
    {
        "id": "D14",
        "target_node": "planner",
        "case_type": "restart_sequence",
        "intent": "Enable Jumbo frames by setting MTU 9216 on interface 8-eth0 of router 8 and restart the interface by bringing it down then up.",
        "expected_ops": ["interface_attribute_config", "interface_admin_config"],
        "expected_min_subintents": 2,
    },
    {
        "id": "D15",
        "target_node": "discretize",
        "case_type": "negative_ambiguous",
        "intent": "Improve routing performance between router 1 and router 5.",
        "expected_ops": [],
        "expected_needs_human": True,
    },
]

INTENTS_DA_IC = [
    "Configure a static route on router 0 to 172.16.9.0/24 using the IP address of the neighbor connected to interface 0-eth2 as next-hop.",
    "Configure a static route on router 2 to 172.16.7.0/24 using the IP of the device connected to 2-eth1 as next-hop.",
    "Configure a backup static route on router 4 to 172.16.1.0/24 using the IP address of the peer of 4-eth0 as next-hop.",
    "Configure a static route on router 7 to 172.16.9.0/24 using the IP of the device linked to 7-eth1 as next-hop.",
    "Configure a static route on router 1 to reach router 6 LAN using the IP of the neighbor attached to 1-eth0 as next-hop.",
    "Enable IPv6 forwarding on router 9 and add an IPv6 route to 3000::/64 through interface 9-eth0.",
    "Bring up interface 8-eth0, assign the IPv4 address 10.0.8.10/30 to it, and then enable IPv4 forwarding on router 8.",
    "Create a BGP peer-group named IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
    "Enable IPv4 forwarding on router 5 and set the MTU of interface 5-eth1 to 9216 bytes.",
    "Bring interface 3-eth1 administratively up and configure its description as BACKBONE_TO_6.",
    "Configure a static route on router 7 to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
    "Set MED to 100 for BGP neighbor 10.0.2.2 and enable AS-path multipath on router 0.",
    "Configure a static route on router 3 to 172.16.4.0/24 via 10.0.4.2.",
    "Configure BGP local-preference to 200 for peer-group EDGE on router 4.",
    "Bring down interface 1-eth1 and remove its IP address.",
    "Remove the static route pointing to 172.16.0.0/24 from router 7.",
    "Disable ARP encapsulation on interface 0-eth2 of router 0.",
    "Enable IPv4 forwarding globally on router 7.",
    "Administratively shut down interface 8-eth0 on router 8.",
    "Bind neighbor 10.0.8.2 to the BGP peer-group EDGE on router 4.",
    "On router 9, add an IPv6 route to 3000::/64 through interface 9-eth0 and enable IPv6 forwarding.",
    "Find the IP of the neighbor connected to interface 0-eth2, and configure a static route on router 0 to 172.16.9.0/24 via that IP.",
    "On router 7, configure a static route to 172.16.0.0/24 via 10.0.6.1 with administrative distance 50.",
    "Create a floating default route (0.0.0.0/0) on router 0 via 10.0.1.2 with administrative distance 250.",
    "Configure a BGP peer-group named IBGP on router 0, set its remote-as to 65000, and bind neighbor 10.0.1.2 to it.",
    "Enable IP forwarding on router 7 and set the MTU of interface 7-eth1 to 9000 bytes.",
    "On router 9, assign the IPv6 address 2001:db8:1::1/64 to interface 9-eth0.",
    "Apply a rate limit of 100Mbps on interface 0-eth2 of router 0 specifically for outbound TCP traffic.",
    "Configure a static ARP entry on router 0 mapping IP 10.0.0.2 to MAC address 02:00:00:00:00:02.",
    "Remove all static routes on router 7 that point to prefix 172.16.0.0/24.",
    
    # --- 20 INTENTS NOVOS (Foco em Pesquisa, Limites de Raciocínio Topológico e Múltiplos Passos) ---
    "Configure a static route on router 1 to reach the LAN subnet of router 5, using the IP of the neighbor connected to 1-eth1 as the gateway.",
    "Find the router connected to interface 0-eth1, and configure a static route on that remote router back to 172.16.0.0/24 via 10.0.1.1.",
    "Apply a 50Mbps rate limit for inbound UDP traffic on the specific interface of router 4 that connects to router 0.",
    "On router 2, create a BGP peer-group named TRANSIT, set remote-as 64512, and set the BGP authentication password to 'SeCrEt' for this group.",
    "Set the local-preference to 150 for peer-group IBGP on router 0, and configure MED to 50 for neighbor 10.0.2.2.",
    "Apply a route-map named BLOCK_DEFAULT in the inbound direction for BGP neighbor 10.0.1.2 on router 0.",
    "Remove the BGP peer-group EDGE from router 4 and delete all associated static routes pointing to 172.16.1.0/24.",
    "Add a static route on router 6 to 172.16.8.0/24 via 10.0.6.2, and create a backup route to the exact same destination via 10.0.7.2 with administrative distance 200.",
    "Disable ARP, shut down interface 7-eth0, and remove its IPv4 address on router 7.",
    "Enable dual-stack forwarding on router 3, then assign IPv4 10.0.3.1/30 and IPv6 2001:db8:3::1/64 to interface 3-eth0.",
    "Change the IP address of 1-eth0 on router 1 to 10.0.10.1/30 and ensure the interface is administratively up.",
    "Configure a blackhole route on router 9 for the prefix 10.100.0.0/16 by pointing it to the Null0 interface.",
    "Limit inbound ICMP traffic to 5Mbps on interface 2-eth0 of router 2.",
    "Delete the static ARP entry for 10.0.0.2 on router 0, and clear the description of interface 0-eth0.",
    "Disable IPv6 forwarding globally on router 4, but ensure IPv4 forwarding remains enabled.",
    "Assign the IP address 10.255.255.1/32 to the loopback interface on router 1.",
    "Set the description of interface 0-eth2 on router 0 to TO_ROUTER_9 and set the description of 9-eth0 on router 9 to TO_ROUTER_0.",
    "Set the OSPF cost of interface 5-eth0 on router 5 to 100 and configure its description as PRIMARY_LINK.",
    "Enable Jumbo frames by setting the MTU to 9216 on interface 8-eth0 of router 8, and restart the interface by bringing it down then up.",
    "Find the IPv4 network CIDR assigned to interface 3-eth1 and configure a static route on router 0 pointing to that network via 10.0.1.2."
]

BENCHMARK_INTENTS_DIAGNOSTIC = [
    case["intent"]
    for case in BENCHMARK_DIAGNOSTIC_CASES
]

EXPECTED_BY_INTENT = {
    case["intent"]: case
    for case in BENCHMARK_DIAGNOSTIC_CASES
}

GOLDEN_COMMANDS_BY_INTENT = _load_golden_command_cases()
for intent_text, golden_case in GOLDEN_COMMANDS_BY_INTENT.items():
    merged_case = dict(EXPECTED_BY_INTENT.get(intent_text) or {})
    merged_case.update(golden_case)
    EXPECTED_BY_INTENT[intent_text] = merged_case

BENCHMARK_INTENTS_WITH_DIAGNOSTICS = (
    BENCHMARK_INTENTS_STRATEGIC
    + BENCHMARK_INTENTS_DIAGNOSTIC
)

def parse_model_spec(model_name: str) -> dict:
    provider = None
    model_id = model_name
    if ":" in model_name:
        model_id, provider = model_name.rsplit(":", 1)
    return {
        "model_spec": model_name,
        "model_id": model_id,
        "provider": provider,
        "api_backend": "huggingface_router",
    }


def build_model_metadata(model_name: str) -> dict:
    meta = parse_model_spec(model_name)
    meta["resolved_model"] = get_ollama_model_name()
    return meta


def timed_node(node_name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        token_start = tcc_nodes.get_llm_token_event_count()
        out = fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        token_events = tcc_nodes.get_llm_token_events_since(token_start)
        token_usage = tcc_nodes.summarize_llm_token_events(token_events)

        work = out.get("work") or {}
        subintents = work.get("subintents") or []
        cursor = int(work.get("cursor", 0))
        active_id = None

        if 0 <= cursor < len(subintents) and isinstance(subintents[cursor], dict):
            active_id = subintents[cursor].get("id") or f"S{cursor + 1}"

        timing = out.get("_timing") or {"nodes": [], "totals_ms": {}}
        timing["nodes"].append({
            "node": node_name,
            "ms": round(elapsed_ms, 3),
            "subintent_id": active_id,
            "token_usage": token_usage,
        })
        timing["totals_ms"][node_name] = timing["totals_ms"].get(node_name, 0.0) + elapsed_ms
        token_totals = timing.setdefault("token_totals_by_node", {})
        node_tokens = token_totals.setdefault(node_name, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "missing_usage_count": 0,
        })
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "missing_usage_count"):
            node_tokens[key] += int(token_usage.get(key) or 0)
        out["_timing"] = timing
        return out

    return _wrapped


def timed_traced_node(node_name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    return timed_node(node_name, fn)


def aggregate_token_totals_by_node(results: list[dict]) -> dict:
    totals: dict[str, dict[str, int]] = {}
    for result in results:
        timing = result.get("timing") or {}
        by_node = timing.get("token_totals_by_node") or {}
        if not isinstance(by_node, dict):
            continue
        for node_name, usage in by_node.items():
            if not isinstance(usage, dict):
                continue
            node_totals = totals.setdefault(node_name, {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "missing_usage_count": 0,
            })
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "missing_usage_count"):
                node_totals[key] += int(usage.get(key) or 0)
    return totals


def persist_trace(state: Dict[str, Any], base_dir: str = "outputs") -> Optional[str]:
    trace = state.get("_trace") or {}
    events = trace.get("events") or []
    if not events:
        return None

    now = datetime.now()
    day_folder = f"dia {now.strftime('%d-%m')}"
    root = Path(base_dir) / day_folder
    root.mkdir(parents=True, exist_ok=True)

    existing = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("teste_")])
    test_id = len(existing) + 1
    test_dir = root / f"teste_{test_id}"
    test_dir.mkdir(parents=True, exist_ok=True)

    out_path = test_dir / "trace.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    return str(out_path)


def load_experiment_topology(path: str) -> dict:
    topo_path = Path(path)
    if not topo_path.exists():
        raise FileNotFoundError(f"Topology file not found: {path}")

    with topo_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Topology file must contain a JSON object: {path}")

    return data


def _extract_intent_name(sub: dict) -> Optional[str]:
    intent = sub.get("intent") or {}
    if hasattr(intent, "model_dump"):
        intent = intent.model_dump()
    elif hasattr(intent, "dict"):
        intent = intent.dict()

    if isinstance(intent, dict):
        name = intent.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    return None


def _extract_confidence(sub: dict) -> Optional[float]:
    intent = sub.get("intent") or {}
    if hasattr(intent, "model_dump"):
        intent = intent.model_dump()
    elif hasattr(intent, "dict"):
        intent = intent.dict()

    if isinstance(intent, dict):
        conf = intent.get("confidence")
        if isinstance(conf, (int, float)):
            return float(conf)

    return None


def _score_operation(sub: dict) -> tuple[int, str]:
    name = _extract_intent_name(sub)

    if isinstance(name, str) and name and name != "undetermined":
        return 2, "Operation determined."

    plan_items = sub.get("plan_items") or []
    plan_steps = sub.get("plan_steps") or []
    if (isinstance(plan_items, list) and len(plan_items) > 0) or (isinstance(plan_steps, list) and len(plan_steps) > 0):
        return 1, "Operation not explicitly classified, but planning artifacts were produced."

    return 0, "Operation not determined."


def _score_grounding(sub: dict) -> tuple[int, str]:
    plan_items = sub.get("plan_items") or []
    plan_steps = sub.get("plan_steps") or []

    if isinstance(plan_items, list) and len(plan_items) > 0:
        missing_total = 0
        for item in plan_items:
            if isinstance(item, dict):
                missing = item.get("missing") or []
                if isinstance(missing, list):
                    missing_total += len(missing)

        if missing_total == 0:
            return 2, "Arguments grounded with no reported missing fields."
        return 1, "Partial grounding; some fields are still missing."

    if isinstance(plan_steps, list) and len(plan_steps) > 0:
        return 2, "Executable steps were produced."

    return 0, "No grounded plan produced."


def _ops_match_expected(predicted_ops: list[str], expected_ops: list[str]) -> dict:
    predicted_set = {
        op
        for op in (predicted_ops or [])
        if isinstance(op, str) and op.strip()
    }
    expected_set = {
        op
        for op in (expected_ops or [])
        if isinstance(op, str) and op.strip()
    }
    missing = sorted(expected_set - predicted_set)
    extra = sorted(predicted_set - expected_set)
    return {
        "ok": len(missing) == 0,
        "expected_ops": sorted(expected_set),
        "predicted_ops": sorted(predicted_set),
        "missing_ops": missing,
        "extra_ops": extra,
    }


def _collect_bound_arg_names(subintents: list[dict]) -> set[str]:
    names = set()
    for sub in subintents or []:
        resolution = sub.get("argument_resolution") or {}
        for op in resolution.get("resolved_operations") or []:
            if not isinstance(op, dict):
                continue
            bound_args = op.get("bound_args") or {}
            if isinstance(bound_args, dict):
                names.update(str(k) for k in bound_args.keys())
    return names


def _normalize_command_record(command: object) -> dict | None:
    if isinstance(command, dict):
        target_device = command.get("target_device")
        command_text = command.get("command")
        if isinstance(target_device, (int, float)):
            target_device = str(int(target_device))
        if isinstance(target_device, str) and isinstance(command_text, str):
            target_device = target_device.strip()
            command_text = command_text.strip()
            if target_device and command_text:
                return {
                    "target_device": target_device,
                    "command": command_text,
                }
    return None


def _collect_compiled_commands(subintents: list[dict]) -> list[dict]:
    commands = []
    for sub in subintents or []:
        compilation = sub.get("command_compilation") or {}
        for command in compilation.get("commands") or []:
            normalized = _normalize_command_record(command)
            if normalized:
                commands.append(normalized)
    return commands


def _collect_result_warnings(result: dict) -> list[str]:
    warnings = []

    for warning in result.get("warnings") or []:
        if isinstance(warning, str):
            warnings.append(warning)

    for sub in result.get("subintents") or []:
        for warning in sub.get("warnings") or []:
            if isinstance(warning, str):
                warnings.append(warning)

        resolution = sub.get("argument_resolution") or {}
        for warning in resolution.get("warnings") or []:
            if isinstance(warning, str):
                warnings.append(warning)

        compilation = sub.get("command_compilation") or {}
        for warning in compilation.get("warnings") or []:
            if isinstance(warning, str):
                warnings.append(warning)

    return list(dict.fromkeys(warnings))


def _collect_result_errors(result: dict) -> list[str]:
    errors = []
    if result.get("error"):
        errors.append(str(result.get("error")))

    for sub in result.get("subintents") or []:
        compilation = sub.get("command_compilation") or {}
        for error in compilation.get("errors") or []:
            if isinstance(error, str):
                errors.append(error)

    return list(dict.fromkeys(errors))


def _commands_match_expected(actual_commands: list[dict], expected_commands: list[dict]) -> dict:
    actual = [
        normalized for normalized in (
            _normalize_command_record(command)
            for command in (actual_commands or [])
        )
        if normalized
    ]
    expected = [
        normalized for normalized in (
            _normalize_command_record(command)
            for command in (expected_commands or [])
        )
        if normalized
    ]
    first_diff = None
    for index, (expected_cmd, actual_cmd) in enumerate(zip(expected, actual), start=1):
        if expected_cmd != actual_cmd:
            first_diff = {
                "index": index,
                "expected": expected_cmd,
                "actual": actual_cmd,
            }
            break
    if first_diff is None and len(expected) != len(actual):
        first_diff = {
            "index": min(len(expected), len(actual)) + 1,
            "expected": expected[min(len(expected), len(actual))] if len(expected) > len(actual) else None,
            "actual": actual[min(len(expected), len(actual))] if len(actual) > len(expected) else None,
        }

    return {
        "ok": actual == expected,
        "comparison": "exact_ordered_targeted_command_list",
        "expected_commands": expected,
        "actual_commands": actual,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "first_diff": first_diff,
    }


def compare_expected_result(intent_text: str, result: dict) -> dict | None:
    expected = EXPECTED_BY_INTENT.get(intent_text)
    if not expected:
        return None

    checks = {}
    checks["ops"] = _ops_match_expected(
        result.get("predicted_ops") or [],
        expected.get("expected_ops") or [],
    )

    if "expected_needs_human" in expected:
        actual = bool(result.get("needs_human"))
        wanted = bool(expected.get("expected_needs_human"))
        checks["needs_human"] = {
            "ok": actual == wanted,
            "expected": wanted,
            "actual": actual,
        }

    subintent_count = int(result.get("subintent_count") or 0)
    if "expected_min_subintents" in expected:
        minimum = int(expected.get("expected_min_subintents") or 0)
        checks["min_subintents"] = {
            "ok": subintent_count >= minimum,
            "expected_min": minimum,
            "actual": subintent_count,
        }

    if "expected_max_subintents" in expected:
        maximum = int(expected.get("expected_max_subintents") or 0)
        checks["max_subintents"] = {
            "ok": subintent_count <= maximum,
            "expected_max": maximum,
            "actual": subintent_count,
        }

    if expected.get("expected_bound_args"):
        bound_names = _collect_bound_arg_names(result.get("subintents") or [])
        expected_bound = set(expected.get("expected_bound_args") or [])
        missing_bound = sorted(expected_bound - bound_names)
        checks["bound_args"] = {
            "ok": len(missing_bound) == 0,
            "expected_bound_args": sorted(expected_bound),
            "actual_bound_args": sorted(bound_names),
            "missing_bound_args": missing_bound,
        }

    if "expected_commands" in expected:
        checks["commands"] = _commands_match_expected(
            _collect_compiled_commands(result.get("subintents") or []),
            expected.get("expected_commands") or [],
        )

    return {
        "case_id": expected.get("id"),
        "target_node": expected.get("target_node"),
        "case_type": expected.get("case_type"),
        "golden_source": expected.get("_source") or (GOLDEN_COMMANDS_PATH if intent_text in GOLDEN_COMMANDS_BY_INTENT else "inline"),
        "ok": all(check.get("ok", False) for check in checks.values()),
        "checks": checks,
    }


def compare_expected_baseline_result(intent_text: str, result: dict) -> dict | None:
    expected = EXPECTED_BY_INTENT.get(intent_text)
    if not expected:
        return None

    checks = {}
    if "expected_needs_human" in expected:
        actual = bool(result.get("needs_human"))
        wanted = bool(expected.get("expected_needs_human"))
        checks["needs_human"] = {
            "ok": actual == wanted,
            "expected": wanted,
            "actual": actual,
        }

    if "expected_commands" in expected:
        checks["commands"] = _commands_match_expected(
            result.get("commands") or [],
            expected.get("expected_commands") or [],
        )

    return {
        "case_id": expected.get("id"),
        "target_node": expected.get("target_node"),
        "case_type": expected.get("case_type"),
        "golden_source": expected.get("_source") or (GOLDEN_COMMANDS_PATH if intent_text in GOLDEN_COMMANDS_BY_INTENT else "inline"),
        "ok": all(check.get("ok", False) for check in checks.values()),
        "checks": checks,
    }


def build_test_output_summary(result: dict, mode: str) -> dict:
    expected_comparison = result.get("expected_comparison") or {}
    checks = expected_comparison.get("checks") or {}
    command_check = checks.get("commands") or {}
    needs_human_check = checks.get("needs_human") or {}

    if mode == "direct_baseline":
        actual_commands = result.get("commands") or []
    else:
        actual_commands = _collect_compiled_commands(result.get("subintents") or [])

    expected_commands = command_check.get("expected_commands")
    if expected_commands is None:
        expected_commands = (EXPECTED_BY_INTENT.get(result.get("intent_text")) or {}).get("expected_commands") or []

    return {
        "intent_id": result.get("intent_id"),
        "case_id": expected_comparison.get("case_id"),
        "intent_text": result.get("intent_text"),
        "ok": expected_comparison.get("ok") if expected_comparison else False,
        "has_expected": bool(expected_comparison),
        "commands_ok": command_check.get("ok"),
        "needs_human_ok": needs_human_check.get("ok"),
        "actual_needs_human": result.get("needs_human"),
        "expected_needs_human": needs_human_check.get("expected"),
        "actual_commands": actual_commands,
        "expected_commands": expected_commands,
        "first_diff": command_check.get("first_diff"),
        "errors": _collect_result_errors(result),
        "warnings": _collect_result_warnings(result),
    }


def attach_test_output_summary(result: dict, mode: str) -> None:
    result["test_output"] = build_test_output_summary(result, mode)


def node_pick_subintent_tcc(state: IBNState) -> IBNState:
    work = state.setdefault("work", {})
    subs = work.get("subintents") or []
    cursor = int(work.get("cursor", 0) or 0)

    if not isinstance(subs, list) or cursor >= len(subs):
        state["active_subintent_id"] = None
        state["active_subintent_text"] = None
        return state

    sub = subs[cursor] if isinstance(subs[cursor], dict) else {"id": f"S{cursor + 1}", "text": str(subs[cursor])}
    sub.setdefault("id", f"S{cursor + 1}")
    sub.setdefault("text", "")
    work["subintent"] = sub
    state["active_subintent_id"] = sub.get("id")
    state["active_subintent_text"] = sub.get("text")

    context_by_subintent = work.get("context_by_subintent") or {}
    local_context = context_by_subintent.get(sub.get("id")) or {}
    shared_context = work.get("shared_context") or state.get("shared_context") or {}
    combined_context = tcc_nodes._merge_context_records(local_context, shared_context)
    combined_context["subintent_id"] = sub.get("id")
    combined_context["goal"] = local_context.get("goal")
    combined_context["requested_topology_entities"] = local_context.get("requested_topology_entities") or {}
    combined_context["semantic_only_arguments"] = local_context.get("semantic_only_arguments") or {}
    combined_context["derived_arguments"] = local_context.get("derived_arguments") or {}
    combined_context["notes"] = local_context.get("notes") or {}
    combined_context["local_context"] = local_context
    combined_context["shared_context_scope"] = "root_intent"
    state["subintent_context"] = combined_context

    return state


def node_store_subintent_result_tcc(state: IBNState) -> IBNState:
    work = state.setdefault("work", {})
    sid = state.get("active_subintent_id") or "S?"
    sub = work.get("subintent") or {}
    planner_result = state.get("planner_result") or {}
    argument_resolution = state.get("argument_resolution") or {}
    argument_resolver_debug = state.get("argument_resolver_debug") or {}
    command_compilation = state.get("command_compilation") or {}
    context = state.get("subintent_context") or {}
    apply_ops = planner_result.get("apply_operations") or []
    verify_ops = planner_result.get("verify_operations") or []
    first_op = next((op.get("op") for op in apply_ops + verify_ops if isinstance(op, dict) and op.get("op")), "undetermined")

    result = {
        "subintent_text": state.get("active_subintent_text") or sub.get("text"),
        "intent_frame": sub.get("intent_frame") or {},
        "intent": {
            "name": first_op,
            "category": planner_result.get("objective_mode") or "unknown",
            "confidence": float(planner_result.get("confidence", 0.0) or 0.0),
            "rationale": planner_result.get("objective") or "",
        },
        "needs_human": bool(
            planner_result.get("needs_human")
            or argument_resolution.get("needs_human")
            or command_compilation.get("needs_human")
            or context.get("needs_human")
            or state.get("needs_human")
        ),
        "plan_items": apply_ops,
        "plan_steps": verify_ops,
        "planner_result": planner_result,
        "argument_resolution": argument_resolution,
        "argument_resolver_debug": argument_resolver_debug,
        "command_compilation": command_compilation,
        "context": context,
        "warnings": list(state.get("warnings") or []),
    }

    results = work.setdefault("results", {})
    results[sid] = result
    return state


def node_advance_cursor_tcc(state: IBNState) -> IBNState:
    work = state.setdefault("work", {})
    work["cursor"] = int(work.get("cursor", 0) or 0) + 1
    state["needs_human"] = False
    state["planner_result"] = None
    state["argument_resolution"] = None
    state["argument_resolver_debug"] = None
    state["command_compilation"] = None
    state["subintent_context"] = None
    return state


def router_has_more_subintents_tcc(state: IBNState) -> str:
    work = state.get("work") or {}
    subs = work.get("subintents") or []
    cursor = int(work.get("cursor", 0) or 0)
    return "more" if isinstance(subs, list) and cursor < len(subs) else "done"


def build_graph_debug():
    workflow = StateGraph(IBNState)

    workflow.add_node("discretize", timed_traced_node("discretize", node_discretize_intent))
    workflow.add_node("switch_downstream_model", timed_traced_node("switch_downstream_model", node_switch_to_downstream_model))
    workflow.add_node("extract_entities", timed_traced_node("extract_entities", node_extract_subintent_entities))
    workflow.add_node("pick_subintent", timed_traced_node("pick_subintent", node_pick_subintent_tcc))
    workflow.add_node("context", timed_traced_node("context", node_context))
    workflow.add_node("planner", timed_traced_node("planner", node_planner))
    workflow.add_node("argument_resolver", timed_traced_node("argument_resolver", node_argument_resolver))
    workflow.add_node("compiler", timed_traced_node("compiler", node_compile_commands))
    workflow.add_node("store_result", timed_traced_node("store_result", node_store_subintent_result_tcc))
    workflow.add_node("advance", timed_traced_node("advance", node_advance_cursor_tcc))

    workflow.set_entry_point("discretize")

    workflow.add_edge("discretize", "switch_downstream_model")
    workflow.add_edge("switch_downstream_model", "extract_entities")
    workflow.add_edge("extract_entities", "context")
    workflow.add_edge("context", "pick_subintent")
    workflow.add_edge("pick_subintent", "planner")
    workflow.add_edge("planner", "argument_resolver")
    workflow.add_edge("argument_resolver", "compiler")
    workflow.add_edge("compiler", "store_result")
    workflow.add_edge("store_result", "advance")

    workflow.add_conditional_edges(
        "advance",
        router_has_more_subintents_tcc,
        {
            "more": "pick_subintent",
            "done": END,
        },
    )

    return workflow.compile()


def build_graph():
    return build_graph_debug()


def build_baseline_user_prompt(intent: str, compact_topo: dict) -> str:
    topology_json = json.dumps(compact_topo, ensure_ascii=False, indent=2)
    return f"""
Convert this network intent into final targeted CLI commands.

Intent:
{intent}

Topology context:
{topology_json}

Return JSON only.
Use fully rendered Linux or FRRouting commands.
Each item in commands must be an object: {{"target_device": "...", "command": "..."}}.
If a required command value cannot be resolved, return commands=[] and needs_human=true.
""".strip()


def try_parse_json_fragment(raw: str) -> dict:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def print_pipeline_result(intent_index: int, intent_text: str, result: dict) -> None:
    print("\n" + "=" * 100)
    print(f"[PIPELINE][INTENT {intent_index}] {intent_text}")
    print("=" * 100)
    print(f"elapsed_sec        : {result.get('elapsed_sec')}")
    print(f"needs_human        : {result.get('needs_human')}")
    print(f"error              : {result.get('error')}")
    print(f"subintent_count    : {result.get('subintent_count')}")
    print(f"predicted_ops      : {result.get('predicted_ops')}")
    if result.get("expected_comparison"):
        print("expected_comparison:")
        print(json.dumps(result.get("expected_comparison"), ensure_ascii=False, indent=2))

    for sub in result.get("subintents", []):
        print("\n" + "-" * 80)
        print(f"[SUBINTENT {sub['id']}]")
        print(f"text              : {sub['text']}")
        print("extracted_frame   :")
        print(json.dumps(sub.get("intent_frame") or {}, ensure_ascii=False, indent=2))
        print("context           :")
        print(json.dumps(sub.get("context") or {}, ensure_ascii=False, indent=2))
        print(f"classified_op     : {sub['classified_op']}")
        print(f"confidence        : {sub['confidence']}")
        print(f"needs_human       : {sub['needs_human']}")
        print(f"plan_items_count  : {sub['plan_items_count']}")
        print(f"plan_steps_count  : {sub['plan_steps_count']}")
        print("plan_items        :")
        print(json.dumps(sub.get("plan_items") or [], ensure_ascii=False, indent=2))
        print("plan_steps        :")
        print(json.dumps(sub.get("plan_steps") or [], ensure_ascii=False, indent=2))
        print("argument_resolution:")
        print(json.dumps(sub.get("argument_resolution") or {}, ensure_ascii=False, indent=2))
        print("command_compilation:")
        print(json.dumps(sub.get("command_compilation") or {}, ensure_ascii=False, indent=2))
        print("warnings          :")
        print(json.dumps(sub.get("warnings") or [], ensure_ascii=False, indent=2))


def print_baseline_result(intent_index: int, intent_text: str, result: dict) -> None:
    print("\n" + "=" * 100)
    print(f"[BASELINE][INTENT {intent_index}] {intent_text}")
    print("=" * 100)
    print(f"elapsed_sec        : {result.get('elapsed_sec')}")
    print(f"parse_success      : {result.get('parse_success')}")
    print(f"error              : {result.get('error')}")
    print(f"needs_human        : {result.get('needs_human')}")
    print("commands           :")
    print(json.dumps(result.get("commands") or [], ensure_ascii=False, indent=2))
    if result.get("expected_comparison"):
        print("expected_comparison:")
        print(json.dumps(result.get("expected_comparison"), ensure_ascii=False, indent=2))
    print(f"step_count         : {result.get('step_count')}")
    print(f"predicted_ops      : {result.get('predicted_ops')}")
    print("[RAW]")
    print(result.get("raw") or "")


def run_intent_suite(
    app,
    intents: list[str],
    model_name: str,
    run_label: str = "r1",
    downstream_model_name: str | None = None,
) -> dict:
    global ACTIVE_DOWNSTREAM_MODEL_OVERRIDE
    ACTIVE_DOWNSTREAM_MODEL_OVERRIDE = downstream_model_name

    suite_start = time.perf_counter()
    results = []
    topology = load_experiment_topology(TOPOLOGY_PATH)
    partial_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial_model_dir = Path("outputs") / _safe_model_tag(model_name)
    partial_model_dir.mkdir(parents=True, exist_ok=True)
    partial_discretize_path = partial_model_dir / f"discretize_only_fewshots_temp0_{run_label}_partial_{partial_ts}.json"
    partial_pipeline_path = partial_model_dir / f"pipeline_{run_label}_partial_{partial_ts}.json"
    partial_argument_resolver_log_path = partial_model_dir / f"argument_resolver_log_{run_label}_partial_{partial_ts}.jsonl"

    def build_partial_pipeline_report() -> dict:
        partial = {
            "mode": "pipeline_partial",
            "run_label": run_label,
            "temperature": BENCHMARK_TEMPERATURE,
            "model": model_name,
            "discretize_model": model_name,
            "downstream_model": downstream_model_name or model_name,
            "suite_elapsed_sec": round(time.perf_counter() - suite_start, 3),
            "num_intents": len(intents),
            "completed_intents": len(results),
            "token_totals_by_node": aggregate_token_totals_by_node(results),
            "test_outputs": [result.get("test_output") for result in results if result.get("test_output")],
            "results": results,
        }
        partial["test_output_sections"] = build_test_output_sections(partial)
        return partial

    def write_partial_discretize_report() -> None:
        partial_report = {
            "mode": "discretize_only_fewshots_temp0_partial",
            "run_label": run_label,
            "temperature": BENCHMARK_TEMPERATURE,
            "model": model_name,
            "discretize_model": model_name,
            "downstream_model": downstream_model_name or model_name,
            "suite_elapsed_sec": round(time.perf_counter() - suite_start, 3),
            "num_intents": len(intents),
            "completed_intents": len(results),
            "results": [
                {
                    "intent_id": result.get("intent_id"),
                    "intent_text": result.get("intent_text"),
                    "elapsed_sec": result.get("elapsed_sec"),
                    "error": result.get("error"),
                    "discretize_output": result.get("discretize_output"),
                }
                for result in results
            ],
        }
        with partial_discretize_path.open("w", encoding="utf-8") as f:
            json.dump(partial_report, f, ensure_ascii=False, indent=2)

    def write_partial_pipeline_report() -> None:
        with partial_pipeline_path.open("w", encoding="utf-8") as f:
            json.dump(build_partial_pipeline_report(), f, ensure_ascii=False, indent=2)

    def write_partial_argument_resolver_log() -> None:
        partial_report = build_partial_pipeline_report()
        with partial_argument_resolver_log_path.open("w", encoding="utf-8") as f:
            for result in partial_report.get("results", []):
                for sub in result.get("subintents", []):
                    debug = sub.get("argument_resolver_debug") or {}
                    if not debug:
                        continue
                    row = {
                        "run_label": run_label,
                        "intent_id": result.get("intent_id"),
                        "intent_text": result.get("intent_text"),
                        "case_id": (result.get("expected_comparison") or {}).get("case_id"),
                        "subintent_id": sub.get("id"),
                        "subintent_text": sub.get("text"),
                        "classified_op": sub.get("classified_op"),
                        "debug": debug,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for i, intent_text in enumerate(intents, start=1):
        set_ollama_model(model_name)
        state = {
            "user_intent_text": intent_text,
            "topology_full": topology,
            "debug_node_intent": True,

        }

        start = time.perf_counter()
        try:
            out = app.invoke(
                state,
                config={
                    "configurable": {"thread_id": f"{model_name}-{run_label}-debug-{i}"},
                    "recursion_limit": 50,
                },
            )
            elapsed = time.perf_counter() - start
        except Exception as exc:
            elapsed = time.perf_counter() - start
            result = {
                "intent_id": i,
                "intent_text": intent_text,
                "elapsed_sec": round(elapsed, 3),
                "needs_human": True,
                "error": str(exc),
                "subintent_count": 0,
                "predicted_ops": [],
                "subintents": [],
                "discretize_output": None,
            }
            expected_comparison = compare_expected_result(intent_text, result)
            if expected_comparison is not None:
                result["expected_comparison"] = expected_comparison
            attach_test_output_summary(result, "pipeline")
            results.append(result)
            write_partial_discretize_report()
            write_partial_pipeline_report()
            write_partial_argument_resolver_log()
            print_pipeline_result(i, intent_text, result)
            continue
        except KeyboardInterrupt:
            write_partial_discretize_report()
            write_partial_pipeline_report()
            write_partial_argument_resolver_log()
            print(f"\n[PARTIAL SAVE] pipeline_saved_to={partial_pipeline_path}")
            print(f"[PARTIAL SAVE] argument_resolver_log_saved_to={partial_argument_resolver_log_path}")
            print(f"\n[PARTIAL SAVE] discretize_only_saved_to={partial_discretize_path}")
            raise

        work = out.get("work") or {}
        sub_results = work.get("results") or {}
        discretize_subintents = work.get("subintents") or []
        if not isinstance(discretize_subintents, list):
            discretize_subintents = []
        discretize_output = {
            "root_intent": work.get("root_intent") or intent_text,
            "subintent_count": len(discretize_subintents),
            "subintents": discretize_subintents,
            "discretize_debug": work.get("discretize_debug") or {},
            "warnings": [
                w for w in (out.get("warnings") or [])
                if isinstance(w, str) and w.startswith("discretize")
            ],
        }

        total_subs = 0
        needs_human = bool(out.get("needs_human", False))

        detailed_subintents = []
        predicted_ops = []

        for sid, sub in sub_results.items():
            total_subs += 1
            if sub.get("needs_human"):
                needs_human = True

            classified_op = _extract_intent_name(sub)
            if classified_op and classified_op != "undetermined":
                predicted_ops.append(classified_op)

            detailed_subintents.append({
                "id": sid,
                "text": sub.get("subintent_text"),
                "intent_frame": sub.get("intent_frame") or {},
                "context": sub.get("context") or {},
                "classified_op": classified_op,
                "confidence": _extract_confidence(sub),
                "needs_human": bool(sub.get("needs_human", False)),
                "plan_items": sub.get("plan_items") or [],
                "plan_items_count": len(sub.get("plan_items") or []),
                "plan_steps": sub.get("plan_steps") or [],
                "plan_steps_count": len(sub.get("plan_steps") or []),
                "argument_resolution": sub.get("argument_resolution") or {},
                "argument_resolver_debug": sub.get("argument_resolver_debug") or {},
                "command_compilation": sub.get("command_compilation") or {},
                "warnings": sub.get("warnings") or [],
            })

        result = {
            "intent_id": i,
            "intent_text": intent_text,
            "elapsed_sec": round(elapsed, 3),
            "needs_human": needs_human,
            "error": out.get("error"),
            "subintent_count": total_subs,
            "predicted_ops": predicted_ops,
            "subintents": detailed_subintents,
            "discretize_output": discretize_output,
            "timing": out.get("_timing") or {},
        }
        expected_comparison = compare_expected_result(intent_text, result)
        if expected_comparison is not None:
            result["expected_comparison"] = expected_comparison
        attach_test_output_summary(result, "pipeline")

        results.append(result)
        write_partial_discretize_report()
        write_partial_pipeline_report()
        write_partial_argument_resolver_log()
        print(f"[PARTIAL SAVE] pipeline_saved_to={partial_pipeline_path}")
        print(f"[PARTIAL SAVE] argument_resolver_log_saved_to={partial_argument_resolver_log_path}")
        print_pipeline_result(i, intent_text, result)

    suite_elapsed = time.perf_counter() - suite_start

    report = {
        "mode": "pipeline",
            "run_label": run_label,
            "temperature": BENCHMARK_TEMPERATURE,
            "model": model_name,
            "discretize_model": model_name,
            "downstream_model": downstream_model_name or model_name,
            "suite_elapsed_sec": round(suite_elapsed, 3),
            "num_intents": len(intents),
            "token_totals_by_node": aggregate_token_totals_by_node(results),
            "test_outputs": [result.get("test_output") for result in results if result.get("test_output")],
            "results": results,
        }
    report["comparison_summary"] = summarize_expected_comparisons(report)
    ACTIVE_DOWNSTREAM_MODEL_OVERRIDE = None
    return report


def run_direct_baseline_suite(intents: list[str], model_name: str, topology: dict, run_label: str = "r1") -> dict:
    set_ollama_model(model_name)

    suite_start = time.perf_counter()
    results = []

    topo_compact = _topo_compact_with_ips(topology)
    baseline_llm = HFRouterChat(model=model_name)

    for i, intent_text in enumerate(intents, start=1):
        start = time.perf_counter()
        raw = ""
        parsed = None
        parse_error = None

        try:
            user_prompt = build_baseline_user_prompt(intent_text, topo_compact)
            token_start = tcc_nodes.get_llm_token_event_count()
            resp = baseline_llm.invoke([
                ("system", BASELINE_SYSTEM_PROMPT),
                ("user", user_prompt),
            ])
            token_events = tcc_nodes.get_llm_token_events_since(token_start)
            raw = (resp.content or "").strip()
            parsed = try_parse_json_fragment(raw)
        except Exception as exc:
            parse_error = str(exc)
            token_events = []

        elapsed = time.perf_counter() - start
        valid_json = parsed is not None

        steps = parsed.get("steps") if isinstance(parsed, dict) else None
        if not isinstance(steps, list):
            steps = []

        commands = parsed.get("commands") if isinstance(parsed, dict) else None
        if not isinstance(commands, list):
            commands = []
        commands = [
            normalized for normalized in (
                _normalize_command_record(command)
                for command in commands
            )
            if normalized
        ]

        needs_human = parsed.get("needs_human") if isinstance(parsed, dict) else None
        needs_human = bool(needs_human) if isinstance(needs_human, bool) else False

        parsed_ops = []
        for step in steps:
            if isinstance(step, dict):
                op = step.get("op")
                if isinstance(op, str) and op.strip():
                    parsed_ops.append(op.strip())

        result = {
            "intent_id": i,
            "intent_text": intent_text,
            "elapsed_sec": round(elapsed, 3),
            "parse_success": valid_json,
            "error": parse_error,
            "predicted_ops": parsed_ops,
            "commands": commands,
            "needs_human": needs_human,
            "step_count": len(steps),
            "steps": steps,
            "raw": raw,
            "token_usage": tcc_nodes.summarize_llm_token_events(token_events),
        }
        expected_comparison = compare_expected_baseline_result(intent_text, result)
        if expected_comparison is not None:
            result["expected_comparison"] = expected_comparison
        attach_test_output_summary(result, "direct_baseline")

        results.append(result)
        print_baseline_result(i, intent_text, result)

    suite_elapsed = time.perf_counter() - suite_start

    report = {
        "mode": "direct_baseline",
        "run_label": run_label,
        "temperature": BENCHMARK_TEMPERATURE,
        "model": model_name,
        "suite_elapsed_sec": round(suite_elapsed, 3),
        "num_intents": len(intents),
        "token_totals": {
            key: sum(
                int((result.get("token_usage") or {}).get(key) or 0)
                for result in results
            )
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "missing_usage_count")
        },
        "test_outputs": [result.get("test_output") for result in results if result.get("test_output")],
        "results": results,
    }
    report["comparison_summary"] = summarize_expected_comparisons(report)
    return report


def _safe_model_tag(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def build_test_output_sections(report: dict) -> dict:
    test_outputs = [
        item for item in (report.get("test_outputs") or [])
        if isinstance(item, dict)
    ]
    return {
        "correct": [
            item for item in test_outputs
            if item.get("ok") is True
        ],
        "incorrect": [
            item for item in test_outputs
            if item.get("ok") is not True
        ],
    }


def build_status_filtered_report(report: dict, status: str) -> dict:
    sections = report.get("test_output_sections") or build_test_output_sections(report)
    selected_outputs = sections.get(status) or []
    selected_ids = {
        (item.get("intent_id"), item.get("case_id"))
        for item in selected_outputs
    }
    selected_results = []
    for result in report.get("results") or []:
        test_output = result.get("test_output") or {}
        key = (test_output.get("intent_id"), test_output.get("case_id"))
        if key in selected_ids:
            selected_results.append(result)

    filtered = {
        **{k: v for k, v in report.items() if k not in {"results", "test_outputs", "test_output_sections"}},
        "status_filter": status,
        "num_intents": len(selected_outputs),
        "test_outputs": selected_outputs,
        "results": selected_results,
    }
    filtered["comparison_summary"] = summarize_expected_comparisons(filtered)
    return filtered


def save_benchmark_report(report: dict, base_dir: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _safe_model_tag(report["model"])
    mode_tag = report.get("mode", "benchmark")

    model_dir = Path(base_dir) / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_dir / f"{mode_tag}_{ts}.json"
    correct_path = model_dir / f"{mode_tag}_{ts}_correct.json"
    incorrect_path = model_dir / f"{mode_tag}_{ts}_incorrect.json"

    if report.get("test_outputs"):
        report["test_output_sections"] = build_test_output_sections(report)
        report["split_report_paths"] = {
            "correct": str(correct_path),
            "incorrect": str(incorrect_path),
        }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report.get("test_outputs"):
        with correct_path.open("w", encoding="utf-8") as f:
            json.dump(build_status_filtered_report(report, "correct"), f, ensure_ascii=False, indent=2)
        with incorrect_path.open("w", encoding="utf-8") as f:
            json.dump(build_status_filtered_report(report, "incorrect"), f, ensure_ascii=False, indent=2)

    return str(out_path)


def save_argument_resolver_log(report: dict, base_dir: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _safe_model_tag(report["model"])
    model_dir = Path(base_dir) / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_dir / f"argument_resolver_log_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for result in report.get("results", []):
            for sub in result.get("subintents", []):
                debug = sub.get("argument_resolver_debug") or {}
                if not debug:
                    continue
                row = {
                    "run_label": report.get("run_label"),
                    "profile": report.get("profile"),
                    "intent_id": result.get("intent_id"),
                    "intent_text": result.get("intent_text"),
                    "case_id": (result.get("expected_comparison") or {}).get("case_id"),
                    "subintent_id": sub.get("id"),
                    "subintent_text": sub.get("text"),
                    "classified_op": sub.get("classified_op"),
                    "debug": debug,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(out_path)


def build_discretize_only_report(report: dict) -> dict:
    return {
        "mode": "discretize_only_fewshots_temp0",
        "run_label": report.get("run_label"),
        "temperature": report.get("temperature"),
        "model": report.get("model"),
        "suite_elapsed_sec": report.get("suite_elapsed_sec"),
        "num_intents": report.get("num_intents"),
        "results": [
            {
                "intent_id": result.get("intent_id"),
                "intent_text": result.get("intent_text"),
                "elapsed_sec": result.get("elapsed_sec"),
                "error": result.get("error"),
                "discretize_output": result.get("discretize_output"),
            }
            for result in report.get("results", [])
        ],
    }


def summarize_expected_comparisons(report: dict) -> dict:
    results = report.get("results") or []
    comparable = [
        result.get("expected_comparison")
        for result in results
        if isinstance(result.get("expected_comparison"), dict)
    ]

    def count_check(check_name: str) -> tuple[int, int]:
        checks = [
            comparison.get("checks", {}).get(check_name)
            for comparison in comparable
            if isinstance(comparison.get("checks", {}).get(check_name), dict)
        ]
        ok = sum(1 for check in checks if check.get("ok") is True)
        return ok, len(checks)

    all_ok = sum(1 for comparison in comparable if comparison.get("ok") is True)
    commands_ok, commands_total = count_check("commands")
    needs_human_ok, needs_human_total = count_check("needs_human")
    ops_ok, ops_total = count_check("ops")

    return {
        "cases": len(results),
        "comparable_cases": len(comparable),
        "all_ok": all_ok,
        "commands_ok": commands_ok,
        "commands_total": commands_total,
        "needs_human_ok": needs_human_ok,
        "needs_human_total": needs_human_total,
        "ops_ok": ops_ok,
        "ops_total": ops_total,
    }


def print_comparison_summary(report: dict) -> None:
    summary = summarize_expected_comparisons(report)
    print(
        "[COMPARISON SUMMARY] "
        f"all_ok={summary['all_ok']}/{summary['comparable_cases']} | "
        f"commands_ok={summary['commands_ok']}/{summary['commands_total']} | "
        f"needs_human_ok={summary['needs_human_ok']}/{summary['needs_human_total']} | "
        f"ops_ok={summary['ops_ok']}/{summary['ops_total']}"
    )


def save_text_report(report: dict, base_dir: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _safe_model_tag(report["model"])
    mode_tag = report.get("mode", "benchmark")

    model_dir = Path(base_dir) / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_dir / f"{mode_tag}_{ts}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"mode={mode_tag}\n")
        f.write(f"run_label={report.get('run_label')}\n")
        f.write(f"temperature={report.get('temperature')}\n")
        f.write(f"model={report.get('model')}\n")
        f.write(f"suite_elapsed_sec={report.get('suite_elapsed_sec')}\n")
        f.write(f"num_intents={report.get('num_intents')}\n")

        if mode_tag == "pipeline":
            f.write("\n")
            for result in report.get("results", []):
                f.write("=" * 100 + "\n")
                f.write(f"[PIPELINE][INTENT {result.get('intent_id')}] {result.get('intent_text')}\n")
                f.write("=" * 100 + "\n")
                f.write(f"elapsed_sec        : {result.get('elapsed_sec')}\n")
                f.write(f"needs_human        : {result.get('needs_human')}\n")
                f.write(f"error              : {result.get('error')}\n")
                f.write(f"subintent_count    : {result.get('subintent_count')}\n")
                f.write(f"predicted_ops      : {result.get('predicted_ops')}\n")
                f.write("[NODE_TIMING_AND_TOKENS]\n")
                f.write(json.dumps(result.get("timing") or {}, ensure_ascii=False, indent=2) + "\n")
                if result.get("expected_comparison"):
                    f.write("[EXPECTED_COMPARISON]\n")
                    f.write(json.dumps(result.get("expected_comparison"), ensure_ascii=False, indent=2) + "\n")

                for sub in result.get("subintents", []):
                    f.write("\n" + "-" * 80 + "\n")
                    f.write(f"[SUBINTENT {sub.get('id')}]\n")
                    f.write(f"text              : {sub.get('text')}\n")
                    f.write("[EXTRACTED_FRAME]\n")
                    f.write(json.dumps(sub.get("intent_frame") or {}, ensure_ascii=False, indent=2) + "\n")
                    f.write("[CONTEXT]\n")
                    f.write(json.dumps(sub.get("context") or {}, ensure_ascii=False, indent=2) + "\n")
                    f.write(f"classified_op     : {sub.get('classified_op')}\n")
                    f.write(f"confidence        : {sub.get('confidence')}\n")
                    f.write(f"needs_human       : {sub.get('needs_human')}\n")
                    f.write(f"plan_items_count  : {sub.get('plan_items_count')}\n")
                    f.write(f"plan_steps_count  : {sub.get('plan_steps_count')}\n")
                    f.write("[NODE_PLAN_ITEMS_OUTPUT]\n")
                    f.write(json.dumps(sub.get("plan_items") or [], ensure_ascii=False, indent=2) + "\n")
                    f.write("[PLAN_STEPS]\n")
                    f.write(json.dumps(sub.get("plan_steps") or [], ensure_ascii=False, indent=2) + "\n")
                    f.write("[ARGUMENT_RESOLUTION]\n")
                    f.write(json.dumps(sub.get("argument_resolution") or {}, ensure_ascii=False, indent=2) + "\n")
                    f.write("[ARGUMENT_RESOLVER_DEBUG]\n")
                    f.write(json.dumps(sub.get("argument_resolver_debug") or {}, ensure_ascii=False, indent=2) + "\n")
                    f.write("[COMMAND_COMPILATION]\n")
                    f.write(json.dumps(sub.get("command_compilation") or {}, ensure_ascii=False, indent=2) + "\n")
                    f.write("[WARNINGS]\n")
                    f.write(json.dumps(sub.get("warnings") or [], ensure_ascii=False, indent=2) + "\n")
                f.write("\n")

        elif mode_tag == "direct_baseline":
            f.write("\n")

            for result in report.get("results", []):
                f.write("=" * 100 + "\n")
                f.write(f"[BASELINE][INTENT {result.get('intent_id')}] {result.get('intent_text')}\n")
                f.write("=" * 100 + "\n")
                f.write(f"elapsed_sec        : {result.get('elapsed_sec')}\n")
                f.write(f"parse_success      : {result.get('parse_success')}\n")
                f.write(f"error              : {result.get('error')}\n")
                f.write(f"needs_human        : {result.get('needs_human')}\n")
                f.write("[TOKEN_USAGE]\n")
                f.write(json.dumps(result.get("token_usage") or {}, ensure_ascii=False, indent=2) + "\n")
                f.write("[COMMANDS]\n")
                f.write(json.dumps(result.get("commands") or [], ensure_ascii=False, indent=2) + "\n")
                if result.get("expected_comparison"):
                    f.write("[EXPECTED_COMPARISON]\n")
                    f.write(json.dumps(result.get("expected_comparison"), ensure_ascii=False, indent=2) + "\n")
                f.write(f"predicted_ops      : {result.get('predicted_ops')}\n")
                f.write(f"step_count         : {result.get('step_count')}\n")
                f.write("[STEPS]\n")
                f.write(json.dumps(result.get("steps") or [], ensure_ascii=False, indent=2) + "\n")
                f.write("[RAW]\n")
                f.write((result.get("raw") or "") + "\n")
                f.write("\n")

    return str(out_path)

def build_llama_hf_benchmark_profiles(selected_models: list[str] | None = None) -> list[dict]:
    model_specs = {
        "8b": {
            "run_label": "llama_8b",
            "model": LLAMA_8B_MODEL,
            "description": "all_modules_llama_8b_hf",
        },
        "70b": {
            "run_label": "llama_70b",
            "model": LLAMA_70B_MODEL,
            "description": "all_modules_llama_70b_hf",
        },
        "qwen4b": {
            "run_label": "qwen_4b",
            "model": QWEN_4B_MODEL,
            "description": "all_modules_qwen_4b_hf",
        },
        "qwen35b": {
            "run_label": "qwen_35b",
            "model": QWEN_35B_MODEL,
            "description": "all_modules_qwen_35b_hf",
        },
    }

    selected = selected_models or ["8b", "70b", "qwen4b", "qwen35b"]
    profiles = []
    for key in selected:
        spec = model_specs[key]
        profiles.append(
            {
                "run_label": spec["run_label"],
                "discretize_model": spec["model"],
                "downstream_model": spec["model"],
                "description": spec["description"],
            }
        )
    return profiles


def parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected benchmark cases with selected models through Hugging Face Router."
    )
    parser.add_argument(
        "--suite",
        choices=["corrected", "intents_da_ic"],
        default="corrected",
        help="Benchmark suite to run. 'intents_da_ic' uses INTENTS_DA_IC and compares against benchmark_test_sbrc.json.",
    )
    parser.add_argument(
        "--benchmark-file",
        default=CORRECTED_BENCHMARK_PATH,
        help="JSON benchmark file containing intent and expected_commands fields.",
    )
    parser.add_argument(
        "--sbrc-benchmark-file",
        default=SBRC_BENCHMARK_PATH,
        help="JSON benchmark file used as expected-command source for --suite intents_da_ic.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of benchmark cases to run from the beginning of the file. Default: 5.",
    )
    parser.add_argument(
        "--start-case",
        type=int,
        default=None,
        help="1-based inclusive start position in the selected suite/file.",
    )
    parser.add_argument(
        "--end-case",
        type=int,
        default=None,
        help="1-based inclusive end position in the selected suite/file.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run every case in --benchmark-file, ignoring --limit.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Run only the listed benchmark case IDs, preserving the provided order.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["8b", "70b", "qwen4b", "qwen35b"],
        default=["8b", "70b", "qwen4b", "qwen35b"],
        help="Subset of HF models to run. Default: 8b 70b qwen4b qwen35b.",
    )
    parser.add_argument(
        "--direct-baseline",
        action="store_true",
        help="Also run the direct baseline for each selected model.",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Run only the graph pipeline, even if --direct-baseline is passed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_benchmark_args()
    app = build_graph_debug()
    topology = load_experiment_topology(TOPOLOGY_PATH)

    using_range = args.start_case is not None or args.end_case is not None
    benchmark_limit = None if (args.all_cases or args.case_ids or using_range) else args.limit
    if args.suite == "intents_da_ic":
        benchmark_cases = build_intents_da_ic_cases(args.sbrc_benchmark_file, limit=benchmark_limit)
        if args.case_ids:
            benchmark_cases = filter_benchmark_cases_by_id(
                load_corrected_benchmark_cases(args.sbrc_benchmark_file, limit=None),
                args.case_ids,
            )
        elif using_range:
            benchmark_cases = slice_benchmark_cases_by_range(
                build_intents_da_ic_cases(args.sbrc_benchmark_file, limit=None),
                start_case=args.start_case,
                end_case=args.end_case,
            )
    else:
        benchmark_cases = load_corrected_benchmark_cases(args.benchmark_file, limit=benchmark_limit)
        if args.case_ids:
            benchmark_cases = filter_benchmark_cases_by_id(
                load_corrected_benchmark_cases(args.benchmark_file, limit=None),
                args.case_ids,
            )
        elif using_range:
            benchmark_cases = slice_benchmark_cases_by_range(
                load_corrected_benchmark_cases(args.benchmark_file, limit=None),
                start_case=args.start_case,
                end_case=args.end_case,
            )
    register_expected_cases(benchmark_cases)
    test_intents = [case["intent"] for case in benchmark_cases]
    benchmark_profiles = build_llama_hf_benchmark_profiles(args.models)
    run_direct_baseline = bool(args.direct_baseline and not args.pipeline_only)

    all_reports = []

    print("\n" + "=" * 100)
    print("[BENCHMARK CONFIG]")
    print("=" * 100)
    print(f"suite={args.suite}")
    print(f"benchmark_file={resolve_existing_path(args.sbrc_benchmark_file if args.suite == 'intents_da_ic' else args.benchmark_file)}")
    print(f"start_case={args.start_case}")
    print(f"end_case={args.end_case}")
    print(f"intents={len(test_intents)}")
    print(f"case_ids={[case.get('id') for case in benchmark_cases]}")
    print(f"profiles={len(benchmark_profiles)}")
    print(f"temperature={BENCHMARK_TEMPERATURE}")
    print("hf_models=" + json.dumps(
        {
            "8b": LLAMA_8B_MODEL,
            "70b": LLAMA_70B_MODEL,
            "qwen4b": QWEN_4B_MODEL,
            "qwen35b": QWEN_35B_MODEL,
        },
        ensure_ascii=False,
    ))
    print(f"run_pipeline={RUN_PIPELINE_BENCHMARK}")
    print(f"run_direct_baseline={run_direct_baseline}")

    for profile in benchmark_profiles:
        run_label = profile["run_label"]
        discretize_model = profile["discretize_model"]
        downstream_model = profile["downstream_model"]
        profile_description = profile["description"]

        print("\n" + "=" * 100)
        print(
            f"[BENCHMARK] {profile_description} | run={run_label} | "
            f"discretize={discretize_model} | downstream={downstream_model}"
        )
        print("=" * 100)

        if RUN_PIPELINE_BENCHMARK:
            pipeline_report = run_intent_suite(
                app,
                test_intents,
                discretize_model,
                run_label=run_label,
                downstream_model_name=downstream_model,
            )
            pipeline_report["profile"] = profile_description
            pipeline_json_path = save_benchmark_report(pipeline_report)
            pipeline_txt_path = save_text_report(pipeline_report)
            argument_resolver_log_path = save_argument_resolver_log(pipeline_report)
            discretize_only_report = build_discretize_only_report(pipeline_report)
            discretize_json_path = save_benchmark_report(discretize_only_report)
            all_reports.append(pipeline_report)

            print("\n" + "#" * 100)
            print(
                f"[PIPELINE SUMMARY] profile={profile_description} | run={pipeline_report['run_label']} | "
                f"discretize={pipeline_report['discretize_model']} | downstream={pipeline_report['downstream_model']}"
            )
            print(f"[PIPELINE SUMMARY] suite_elapsed_sec={pipeline_report['suite_elapsed_sec']}")
            print(f"[PIPELINE SUMMARY] json_saved_to={pipeline_json_path}")
            if pipeline_report.get("split_report_paths"):
                print(f"[PIPELINE SUMMARY] correct_json_saved_to={pipeline_report['split_report_paths']['correct']}")
                print(f"[PIPELINE SUMMARY] incorrect_json_saved_to={pipeline_report['split_report_paths']['incorrect']}")
            print(f"[PIPELINE SUMMARY] txt_saved_to={pipeline_txt_path}")
            print(f"[PIPELINE SUMMARY] argument_resolver_log_saved_to={argument_resolver_log_path}")
            print(f"[PIPELINE SUMMARY] discretize_only_saved_to={discretize_json_path}")
            print_comparison_summary(pipeline_report)
            print("#" * 100)

        if run_direct_baseline:
            direct_report = run_direct_baseline_suite(test_intents, downstream_model, topology, run_label=run_label)
            direct_report["profile"] = profile_description
            baseline_json_path = save_benchmark_report(direct_report)
            baseline_txt_path = save_text_report(direct_report)
            all_reports.append(direct_report)

            print("\n" + "#" * 100)
            print(
                f"[BASELINE SUMMARY] profile={profile_description} | model={direct_report['model']} | "
                f"run={direct_report['run_label']}"
            )
            print(f"[BASELINE SUMMARY] suite_elapsed_sec={direct_report['suite_elapsed_sec']}")
            print(f"[BASELINE SUMMARY] json_saved_to={baseline_json_path}")
            if direct_report.get("split_report_paths"):
                print(f"[BASELINE SUMMARY] correct_json_saved_to={direct_report['split_report_paths']['correct']}")
                print(f"[BASELINE SUMMARY] incorrect_json_saved_to={direct_report['split_report_paths']['incorrect']}")
            print(f"[BASELINE SUMMARY] txt_saved_to={baseline_txt_path}")
            print_comparison_summary(direct_report)
            print("#" * 100)

    print("\n" + "=" * 100)
    print("[BENCHMARK] FINAL SUMMARY")
    print("=" * 100)
    for report in all_reports:
        comparison_summary = report.get("comparison_summary") or summarize_expected_comparisons(report)
        comparison_text = (
            f"all_ok={comparison_summary['all_ok']}/{comparison_summary['comparable_cases']} | "
            f"commands_ok={comparison_summary['commands_ok']}/{comparison_summary['commands_total']}"
        )
        if report.get("mode") == "pipeline":
            print(
                f"- mode=pipeline | profile={report.get('profile')} | run={report.get('run_label')} | "
                f"discretize={report.get('discretize_model')} | downstream={report.get('downstream_model')} | "
                f"suite_elapsed_sec={report['suite_elapsed_sec']} | {comparison_text}"
            )
        else:
            print(
                f"- mode=direct_baseline | model={report['model']} | run={report.get('run_label')} | "
                f"suite_elapsed_sec={report['suite_elapsed_sec']} | {comparison_text}"
            )

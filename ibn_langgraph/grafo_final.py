import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from main import IBNState
from no_grafo import (
    node_intent,
    node_entities,
    node_requirements,
    node_anonymize,
    node_context,
    node_plan,
    node_decide_exec,
    node_execute,
    node_verify,
    node_generate_cli,
    node_step_contract,
    node_discretize_intent,
    node_pick_subintent,
    node_plan_items,
    node_store_subintent_result,
    node_advance_cursor,
    router_has_more_subintents,
    node_debug_print,
    router_refined_or_continue,
    node_orchestrator_root_grouding,
    set_ollama_model,
    get_ollama_model_name,
    _topo_compact_with_ips,
    HFRouterChat,
)

DEBUG_TRACE_NODES = False
DEBUG_EVAL_REPORT = True

BENCHMARK_REPEATS = 3
BENCHMARK_TEMPERATURE = 0.0

TOPOLOGY_PATH = "dataset/topologias_convertidas/gabriel/10/0.json"

RUN_PIPELINE_BENCHMARK = True
RUN_DIRECT_BASELINE_BENCHMARK = True

BASELINE_SYSTEM_PROMPT = """
You are a baseline network intent translator.

Convert one user intent into a small JSON draft that looks like an execution plan.
Do not explain your reasoning.
Return JSON only.

This is a baseline, not a full planner.
Try to identify the main network action, the target device or interface, and the main CLI template that would likely be used.

Prefer a compact answer.
Do not try to fully validate the topology.
Use the topology only as light context for names that already appear in the intent.
If the intent contains more than one action, you may return multiple steps.

Use a structure like this:
{
  "intent_summary": "short summary",
  "steps": [
    {
      "device": "... or null",
      "interface": "... or null",
      "op": "normalized_operation_name",
      "command_family": "linux" | "frr" | "unknown",
      "template": "cli template string",
      "args": {"key": "value"},
      "notes": {}
    }
  ],
  "missing": [],
  "warnings": []
}

Guidelines:
- Use template strings, not fully rendered commands.
- Keep operation names short and normalized.
- Do not invent values that are absent from the intent.
- If you are uncertain, still choose the most plausible interpretation.
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
"Show neighbors of router 1 and configure route to router 6 LAN using discovered peer.",

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
        out = fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

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
        })
        timing["totals_ms"][node_name] = timing["totals_ms"].get(node_name, 0.0) + elapsed_ms
        out["_timing"] = timing
        return out

    return _wrapped


def timed_traced_node(node_name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    return timed_node(node_name, fn)


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


def build_graph_debug():
    workflow = StateGraph(IBNState)

    workflow.add_node("debug_print", timed_traced_node("debug_print", node_debug_print))
    workflow.add_node("orchestrator_root", timed_traced_node("orchestrator_root", node_orchestrator_root_grouding))
    workflow.add_node("pick_subintent", timed_traced_node("pick_subintent", node_pick_subintent))
    workflow.add_node("plan_items", timed_traced_node("plan_items", node_plan_items))
    workflow.add_node("store_result", timed_traced_node("store_result", node_store_subintent_result))
    workflow.add_node("advance", timed_traced_node("advance", node_advance_cursor))
    workflow.add_node("plan", timed_traced_node("plan", node_plan))
    workflow.add_node("generate_cli", timed_traced_node("generate_cli", node_generate_cli))

    workflow.set_entry_point("orchestrator_root")

    workflow.add_edge("orchestrator_root", "pick_subintent")
    workflow.add_edge("pick_subintent", "plan_items")
    workflow.add_edge("plan_items", "store_result")
    workflow.add_edge("store_result", "advance")

    workflow.add_conditional_edges(
        "advance",
        router_has_more_subintents,
        {
            "more": "pick_subintent",
            "done": "plan",
        },
    )

    workflow.add_edge("plan", "generate_cli")
    workflow.add_edge("generate_cli", "debug_print")
    workflow.add_edge("debug_print", END)

    return workflow.compile()


def build_graph():
    workflow = StateGraph(IBNState)

    workflow.add_node("intent", node_intent)
    workflow.add_node("context", node_context)
    workflow.add_node("entities", node_entities)
    workflow.add_node("requirements", node_requirements)
    workflow.add_node("anonymize", node_anonymize)
    workflow.add_node("plan", node_plan)
    workflow.add_node("generate_cli", node_generate_cli)
    workflow.add_node("decide_exec", node_decide_exec)
    workflow.add_node("execute", node_execute)
    workflow.add_node("verify", node_verify)
    workflow.add_node("contract", node_step_contract)

    workflow.set_entry_point("intent")

    workflow.add_edge("intent", "entities")
    workflow.add_edge("entities", "anonymize")
    workflow.add_edge("anonymize", "context")
    workflow.add_edge("context", "requirements")

    def router_needs_human(state: IBNState):
        if state.get("needs_human"):
            return "end"

        topo = state.get("topology") or {}
        devices = topo.get("devices") or {}
        if isinstance(devices, dict) and len(devices) == 0:
            return "end"

        return "continue"

    workflow.add_conditional_edges(
        "requirements",
        router_needs_human,
        {
            "end": END,
            "continue": "plan",
        },
    )

    workflow.add_edge("plan", "contract")
    workflow.add_edge("contract", "generate_cli")
    workflow.add_edge("generate_cli", "decide_exec")
    workflow.add_edge("decide_exec", "execute")
    workflow.add_edge("execute", "verify")
    workflow.add_edge("verify", END)

    conn = sqlite3.connect("ibn_checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def build_baseline_user_prompt(intent: str, compact_topo: dict) -> str:
    topology_json = json.dumps(compact_topo, ensure_ascii=False, indent=2)
    return f"""
Convert this network intent into the JSON draft.

Intent:
{intent}

Topology context:
{topology_json}

Return JSON only.
Use a CLI template that is plausible for Linux or FRRouting.
Do not overthink validation.
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

    for sub in result.get("subintents", []):
        print("\n" + "-" * 80)
        print(f"[SUBINTENT {sub['id']}]")
        print(f"text              : {sub['text']}")
        print(f"classified_op     : {sub['classified_op']}")
        print(f"confidence        : {sub['confidence']}")
        print(f"needs_human       : {sub['needs_human']}")
        print(f"plan_items_count  : {sub['plan_items_count']}")
        print(f"plan_steps_count  : {sub['plan_steps_count']}")
        print("plan_items        :")
        print(json.dumps(sub.get("plan_items") or [], ensure_ascii=False, indent=2))
        print("plan_steps        :")
        print(json.dumps(sub.get("plan_steps") or [], ensure_ascii=False, indent=2))
        print("warnings          :")
        print(json.dumps(sub.get("warnings") or [], ensure_ascii=False, indent=2))


def print_baseline_result(intent_index: int, intent_text: str, result: dict) -> None:
    print("\n" + "=" * 100)
    print(f"[BASELINE][INTENT {intent_index}] {intent_text}")
    print("=" * 100)
    print(f"elapsed_sec        : {result.get('elapsed_sec')}")
    print(f"parse_success      : {result.get('parse_success')}")
    print(f"error              : {result.get('error')}")
    print(f"step_count         : {result.get('step_count')}")
    print(f"predicted_ops      : {result.get('predicted_ops')}")
    print("[RAW]")
    print(result.get("raw") or "")


def run_intent_suite(app, intents: list[str], model_name: str, run_label: str = "r1") -> dict:
    set_ollama_model(model_name)

    suite_start = time.perf_counter()
    results = []

    for i, intent_text in enumerate(intents, start=1):
        state = {
            "user_intent_text": intent_text,
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
            }
            results.append(result)
            print_pipeline_result(i, intent_text, result)
            continue

        work = out.get("work") or {}
        sub_results = work.get("results") or {}

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
                "classified_op": classified_op,
                "confidence": _extract_confidence(sub),
                "needs_human": bool(sub.get("needs_human", False)),
                "plan_items": sub.get("plan_items") or [],
                "plan_items_count": len(sub.get("plan_items") or []),
                "plan_steps": sub.get("plan_steps") or [],
                "plan_steps_count": len(sub.get("plan_steps") or []),
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
        }

        results.append(result)
        print_pipeline_result(i, intent_text, result)

    suite_elapsed = time.perf_counter() - suite_start

    report = {
        "mode": "pipeline",
        "run_label": run_label,
        "temperature": BENCHMARK_TEMPERATURE,
        "model": model_name,
        "suite_elapsed_sec": round(suite_elapsed, 3),
        "num_intents": len(intents),
        "results": results,
    }
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
            resp = baseline_llm.invoke([
                ("system", BASELINE_SYSTEM_PROMPT),
                ("user", user_prompt),
            ])
            raw = (resp.content or "").strip()
            parsed = try_parse_json_fragment(raw)
        except Exception as exc:
            parse_error = str(exc)

        elapsed = time.perf_counter() - start
        valid_json = parsed is not None

        steps = parsed.get("steps") if isinstance(parsed, dict) else None
        if not isinstance(steps, list):
            steps = []

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
            "step_count": len(steps),
            "steps": steps,
            "raw": raw,
        }

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
        "results": results,
    }
    return report


def _safe_model_tag(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")

def save_benchmark_report(report: dict, base_dir: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _safe_model_tag(report["model"])
    mode_tag = report.get("mode", "benchmark")

    model_dir = Path(base_dir) / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_dir / f"{mode_tag}_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return str(out_path)

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

                for sub in result.get("subintents", []):
                    f.write("\n" + "-" * 80 + "\n")
                    f.write(f"[SUBINTENT {sub.get('id')}]\n")
                    f.write(f"text              : {sub.get('text')}\n")
                    f.write(f"classified_op     : {sub.get('classified_op')}\n")
                    f.write(f"confidence        : {sub.get('confidence')}\n")
                    f.write(f"needs_human       : {sub.get('needs_human')}\n")
                    f.write(f"plan_items_count  : {sub.get('plan_items_count')}\n")
                    f.write(f"plan_steps_count  : {sub.get('plan_steps_count')}\n")
                    f.write("[NODE_PLAN_ITEMS_OUTPUT]\n")
                    f.write(json.dumps(sub.get("plan_items") or [], ensure_ascii=False, indent=2) + "\n")
                    f.write("[PLAN_STEPS]\n")
                    f.write(json.dumps(sub.get("plan_steps") or [], ensure_ascii=False, indent=2) + "\n")
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
                f.write(f"predicted_ops      : {result.get('predicted_ops')}\n")
                f.write(f"step_count         : {result.get('step_count')}\n")
                f.write("[STEPS]\n")
                f.write(json.dumps(result.get("steps") or [], ensure_ascii=False, indent=2) + "\n")
                f.write("[RAW]\n")
                f.write((result.get("raw") or "") + "\n")
                f.write("\n")

    return str(out_path)

if __name__ == "__main__":
    app = build_graph_debug()
    topology = load_experiment_topology(TOPOLOGY_PATH)

    test_intents = BENCHMARK_INTENTS_STRATEGIC

    # test_models = [
    #     "meta-llama/Llama-3.2-1B-Instruct:novita",
    #     "meta-llama/Llama-3.1-8B-Instruct:novita",
    #     "meta-llama/Llama-3.3-70B-Instruct:groq",

        # "meta-llama/Llama-3.2-1B-Instruct:novita",
        # "meta-llama/Llama-3.1-8B-Instruct:novita",
        # "meta-llama/Llama-3.3-70B-Instruct:groq",
        # "Qwen/Qwen3-4B-Instruct-2507:nscale",
        # "Qwen/Qwen3.5-35B-A3B:novita",
    # ]
    test_models = [
        "Qwen/Qwen3.5-9B:together",
    ]

    all_reports = []

    print("\n" + "=" * 100)
    print("[BENCHMARK CONFIG]")
    print("=" * 100)
    print(f"intents={len(test_intents)}")
    print(f"repeats={BENCHMARK_REPEATS}")
    print(f"temperature={BENCHMARK_TEMPERATURE}")
    print(f"models={len(test_models)}")

    for model_name in test_models:
        for repeat_idx in range(BENCHMARK_REPEATS):
            run_label = f"r{repeat_idx + 1}"

            print("\n" + "=" * 100)
            print(f"[BENCHMARK] Running model: {model_name} | run={run_label}")
            print("=" * 100)

            if RUN_PIPELINE_BENCHMARK:
                pipeline_report = run_intent_suite(app, test_intents, model_name, run_label=run_label)
                pipeline_json_path = save_benchmark_report(pipeline_report)
                pipeline_txt_path = save_text_report(pipeline_report)
                all_reports.append(pipeline_report)

                print("\n" + "#" * 100)
                print(f"[PIPELINE SUMMARY] model={pipeline_report['model']} | run={pipeline_report['run_label']}")
                print(f"[PIPELINE SUMMARY] suite_elapsed_sec={pipeline_report['suite_elapsed_sec']}")
                print(f"[PIPELINE SUMMARY] json_saved_to={pipeline_json_path}")
                print(f"[PIPELINE SUMMARY] txt_saved_to={pipeline_txt_path}")
                print("#" * 100)

            if RUN_DIRECT_BASELINE_BENCHMARK:
                direct_report = run_direct_baseline_suite(test_intents, model_name, topology, run_label=run_label)
                baseline_json_path = save_benchmark_report(direct_report)
                baseline_txt_path = save_text_report(direct_report)
                all_reports.append(direct_report)

                print("\n" + "#" * 100)
                print(f"[BASELINE SUMMARY] model={direct_report['model']} | run={direct_report['run_label']}")
                print(f"[BASELINE SUMMARY] suite_elapsed_sec={direct_report['suite_elapsed_sec']}")
                print(f"[BASELINE SUMMARY] json_saved_to={baseline_json_path}")
                print(f"[BASELINE SUMMARY] txt_saved_to={baseline_txt_path}")
                print("#" * 100)

    print("\n" + "=" * 100)
    print("[BENCHMARK] FINAL SUMMARY")
    print("=" * 100)
    for report in all_reports:
        if report.get("mode") == "pipeline":
            print(
                f"- mode=pipeline | model={report['model']} | run={report.get('run_label')} | "
                f"suite_elapsed_sec={report['suite_elapsed_sec']}"
            )
        else:
            print(
                f"- mode=direct_baseline | model={report['model']} | run={report.get('run_label')} | "
                f"suite_elapsed_sec={report['suite_elapsed_sec']}"
            )

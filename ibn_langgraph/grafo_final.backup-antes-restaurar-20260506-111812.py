import json
import gc
import argparse
import time
import sqlite3
import os
import requests
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from main import IBNState
import no_grafo_tcc as tcc
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
from no_grafo_tcc import (
    node_discretize_intent as node_discretize_intent_tcc,
    node_context as node_context_tcc,
    node_planner as node_planner_tcc,
)
from tools import _normalize_arguments

DEBUG_TRACE_NODES = False
DEBUG_EVAL_REPORT = True
DISCRETIZE_UNIT_INCLUDE_DIAGNOSTICS = False
DISCRETIZE_UNIT_INCLUDE_DEBUG = False

BENCHMARK_REPEATS = 1
BENCHMARK_TEMPERATURE = tcc.temperature

TOPOLOGY_PATH = "dataset/topologias_convertidas/gabriel/10/0.json"

LOCAL_DISCRETIZE_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_DISCRETIZE_FINETUNED_ADAPTER = "Nicolau3002/autotrain-oqr6i-c75w5"
LOCAL_DISCRETIZE_LOAD_IN_4BIT = True
LOCAL_DISCRETIZE_MAX_NEW_TOKENS = 1024
LOCAL_DISCRETIZE_DEVICE_MAP = "auto"
HF_ENDPOINT_MAX_NEW_TOKENS = 1024
HF_ENDPOINT_BASE_ROUTER_MODEL = "meta-llama/Llama-3.2-3B-Instruct:novita"

RUN_PIPELINE_BENCHMARK = False
RUN_DIRECT_BASELINE_BENCHMARK = False
RUN_DISCRETIZE_UNIT_BENCHMARK = True

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

BENCHMARK_INTENTS_DISCRETIZE_UNIT = [
    # nivel_1
    "Configure a static route on router 3 to 172.16.4.0/24 via 10.0.4.2.",
    "Enable IPv4 forwarding globally on router 7.",
    "Administratively shut down interface 8-eth0 on router 8.",
    "Configure a static ARP entry on router 0 mapping IP 10.0.0.2 to MAC address 02:00:00:00:00:02.",

    # nivel_2
    "Configure a static route on router 0 to 172.16.9.0/24 using the IP address of the neighbor connected to interface 0-eth2 as next-hop.",
    "Configure a static route on router 2 to 172.16.7.0/24 using the IP of the device connected to 2-eth1 as next-hop.",
    "Configure a static route on router 1 to reach the LAN subnet of router 5, using the IP of the neighbor connected to 1-eth1 as the gateway.",
    "Apply a 50Mbps rate limit for inbound UDP traffic on the specific interface of router 4 that connects to router 0.",
    "Find the IPv4 network CIDR assigned to interface 3-eth1 and configure a static route on router 0 pointing to that network via 10.0.1.2.",

    # nivel_3
    "Create a BGP peer-group named IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
    "Configure a BGP peer-group named IBGP on router 0, set its remote-as to 65000, and bind neighbor 10.0.1.2 to it.",
    "Set MED to 100 for BGP neighbor 10.0.2.2 and enable AS-path multipath on router 0.",
    "Enable Jumbo frames by setting the MTU to 9216 on interface 8-eth0 of router 8, and restart the interface by bringing it down then up.",
    "Find the router connected to interface 0-eth1, and configure a static route on that remote router back to 172.16.0.0/24 via 10.0.1.1.",
    "Disable ARP, shut down interface 7-eth0, and remove its IPv4 address on router 7.",
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


class LocalTransformersChat:
    """Minimal chat-compatible wrapper for local base/PEFT discretize benchmarks."""

    def __init__(
        self,
        model: str,
        adapter: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = LOCAL_DISCRETIZE_MAX_NEW_TOKENS,
        load_in_4bit: bool = LOCAL_DISCRETIZE_LOAD_IN_4BIT,
        device_map: str | None = LOCAL_DISCRETIZE_DEVICE_MAP,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._load()

    @property
    def label(self) -> str:
        if self.adapter:
            return f"local-peft:{self.model}+{self.adapter}"
        return f"local-base:{self.model}"

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token = os.environ.get("HF_TOKEN")
        cuda_available = torch.cuda.is_available()
        load_in_4bit = bool(self.load_in_4bit and cuda_available)
        if self.load_in_4bit and not cuda_available:
            print("[LOCAL DISCRETIZE] CUDA not available; loading without 4-bit quantization on CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_kwargs = {
            "torch_dtype": torch.float16 if cuda_available else torch.float32,
            "token": token,
        }
        if cuda_available:
            model_kwargs["device_map"] = self.device_map or "auto"
        else:
            model_kwargs["low_cpu_mem_usage"] = True

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        try:
            self.model_obj = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        except ValueError as exc:
            if load_in_4bit and "Some modules are dispatched on the CPU or the disk" in str(exc):
                print(
                    "[LOCAL DISCRETIZE] 4-bit GPU loading failed because the model did not fit cleanly. "
                    "Retry with --no-4bit, or run this benchmark in the T4 Space."
                )
            raise

        if self.adapter:
            from peft import PeftModel

            self.model_obj = PeftModel.from_pretrained(
                self.model_obj,
                self.adapter,
                token=token,
            )

        self.model_obj.eval()

    def unload(self) -> None:
        model_obj = getattr(self, "model_obj", None)
        tokenizer = getattr(self, "tokenizer", None)
        if model_obj is not None:
            del model_obj
        if tokenizer is not None:
            del tokenizer
        self.model_obj = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def bind(self, **kwargs):
        options = kwargs.get("options") or {}
        clone = object.__new__(LocalTransformersChat)
        clone.model = self.model
        clone.adapter = self.adapter
        clone.temperature = options.get("temperature", self.temperature)
        clone.max_new_tokens = options.get("max_tokens", self.max_new_tokens)
        clone.load_in_4bit = self.load_in_4bit
        clone.device_map = self.device_map
        clone.tokenizer = self.tokenizer
        clone.model_obj = self.model_obj
        return clone

    def _format_prompt(self, messages) -> str:
        system_parts = []
        user_parts = []
        assistant_parts = []

        for item in messages:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            elif isinstance(item, dict):
                role, content = item.get("role"), item.get("content")
            else:
                raise ValueError(f"Unsupported message format: {item!r}")

            role = str(role or "").lower()
            content = str(content or "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)

        prompt_parts = []
        if system_parts:
            prompt_parts.append("\n\n".join(system_parts).strip())
        for user_text in user_parts:
            prompt_parts.append(f"User:\n{user_text.strip()}")
        for assistant_text in assistant_parts:
            prompt_parts.append(f"Assistant:\n{assistant_text.strip()}")

        prompt = "\n\n".join(part for part in prompt_parts if part).strip()
        return f"{prompt}\n\nAssistant:\n"

    def invoke(self, messages):
        import torch

        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = next(self.model_obj.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model_obj.generate(**generate_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return SimpleNamespace(content=content)


class HFEndpointTextGenerationChat:
    """Chat-compatible wrapper for a Hugging Face Inference Endpoint or TGI URL."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        label: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = HF_ENDPOINT_MAX_NEW_TOKENS,
        timeout: int = 300,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key or os.environ["HF_TOKEN"]
        self.model = label or self.endpoint_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def bind(self, **kwargs):
        options = kwargs.get("options") or {}
        return HFEndpointTextGenerationChat(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            label=self.model,
            temperature=options.get("temperature", self.temperature),
            max_new_tokens=options.get("max_tokens", self.max_new_tokens),
            timeout=self.timeout,
        )

    def _normalize_messages(self, messages) -> list[dict]:
        normalized = []
        for item in messages:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            elif isinstance(item, dict):
                role, content = item.get("role"), item.get("content")
            else:
                raise ValueError(f"Unsupported message format: {item!r}")
            normalized.append({"role": str(role or "user"), "content": str(content or "")})
        return normalized

    def _format_prompt(self, messages) -> str:
        system_parts = []
        user_parts = []
        assistant_parts = []

        for msg in self._normalize_messages(messages):
            role = msg["role"].lower()
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)
            else:
                user_parts.append(content)

        prompt_parts = []
        if system_parts:
            prompt_parts.append("\n\n".join(system_parts).strip())
        for user_text in user_parts:
            prompt_parts.append(f"User:\n{user_text.strip()}")
        for assistant_text in assistant_parts:
            prompt_parts.append(f"Assistant:\n{assistant_text.strip()}")
        prompt = "\n\n".join(part for part in prompt_parts if part).strip()
        return f"{prompt}\n\nAssistant:\n"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_generated_text(self, data: Any, prompt: str) -> str:
        generated = ""
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                generated = str(first.get("generated_text") or first.get("text") or "")
        elif isinstance(data, dict):
            if "generated_text" in data:
                generated = str(data.get("generated_text") or "")
            elif "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message") or {}
                    generated = str(message.get("content") or choice.get("text") or "")
            elif "outputs" in data and data["outputs"]:
                output = data["outputs"][0]
                generated = str(output.get("generated_text") if isinstance(output, dict) else output)

        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    def invoke(self, messages):
        normalized_messages = self._normalize_messages(messages)
        use_chat_completion = self.endpoint_url.endswith("/v1/chat/completions")

        if use_chat_completion:
            payload = {
                "messages": normalized_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
            response = requests.post(
                self.endpoint_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return SimpleNamespace(content=self._parse_generated_text(data, ""))

        prompt = self._format_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.temperature > 0,
                "return_full_text": False,
            },
        }
        response = requests.post(
            self.endpoint_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return SimpleNamespace(content=self._parse_generated_text(data, prompt))


def set_tcc_router_model(model_name: str) -> None:
    tcc.temperature = BENCHMARK_TEMPERATURE
    tcc.llm = tcc.HFRouterChat(model=model_name, temperature=BENCHMARK_TEMPERATURE)


def set_tcc_local_model(base_model: str, adapter: str | None = None) -> LocalTransformersChat:
    tcc.temperature = BENCHMARK_TEMPERATURE
    llm = LocalTransformersChat(
        model=base_model,
        adapter=adapter,
        temperature=BENCHMARK_TEMPERATURE,
        max_new_tokens=LOCAL_DISCRETIZE_MAX_NEW_TOKENS,
        load_in_4bit=LOCAL_DISCRETIZE_LOAD_IN_4BIT,
        device_map=LOCAL_DISCRETIZE_DEVICE_MAP,
    )
    tcc.llm = llm
    return llm


def set_tcc_hf_endpoint_model(endpoint_url: str, label: str | None = None) -> HFEndpointTextGenerationChat:
    tcc.temperature = BENCHMARK_TEMPERATURE
    llm = HFEndpointTextGenerationChat(
        endpoint_url=endpoint_url,
        label=label,
        temperature=BENCHMARK_TEMPERATURE,
        max_new_tokens=HF_ENDPOINT_MAX_NEW_TOKENS,
    )
    tcc.llm = llm
    return llm


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


def node_capture_context_snapshot(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    sid = state.get("active_subintent_id")
    raw_context = state.get("subintent_context")

    if not sid:
        return state

    capture_debug = work.get("context_capture_debug")
    if not isinstance(capture_debug, dict):
        capture_debug = {}

    debug_entry = {
        "state_context_present": isinstance(raw_context, dict),
        "state_context_nonempty": bool(raw_context) if isinstance(raw_context, dict) else False,
        "state_context_keys": sorted(raw_context.keys()) if isinstance(raw_context, dict) else [],
    }

    if not isinstance(raw_context, dict):
        capture_debug[sid] = debug_entry
        work["context_capture_debug"] = capture_debug
        state["work"] = work
        return state

    context_results = work.get("context_results")
    if not isinstance(context_results, dict):
        context_results = {}
    context_results[sid] = raw_context
    work["context_results"] = context_results

    subs = work.get("subintents") or []
    cursor = int(work.get("cursor", 0))
    if 0 <= cursor < len(subs) and isinstance(subs[cursor], dict):
        subs[cursor]["context"] = raw_context
    work["subintents"] = subs

    debug_entry["stored_in_context_results"] = sid in context_results
    debug_entry["stored_under_subintent"] = bool(
        0 <= cursor < len(subs)
        and isinstance(subs[cursor], dict)
        and isinstance(subs[cursor].get("context"), dict)
    )
    capture_debug[sid] = debug_entry
    work["context_capture_debug"] = capture_debug

    state["work"] = work
    return state


def node_capture_planner_snapshot(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    sid = state.get("active_subintent_id")
    planner_result = state.get("planner_result")

    if not sid or not isinstance(planner_result, dict):
        return state

    planner_results = work.get("planner_results")
    if not isinstance(planner_results, dict):
        planner_results = {}
    planner_results[sid] = planner_result
    work["planner_results"] = planner_results

    subs = work.get("subintents") or []
    cursor = int(work.get("cursor", 0))
    if 0 <= cursor < len(subs) and isinstance(subs[cursor], dict):
        subs[cursor]["planner"] = planner_result
    work["subintents"] = subs

    state["work"] = work
    return state


def build_graph_discretize_unit():
    workflow = StateGraph(IBNState)

    workflow.add_node("discretize", timed_traced_node("discretize", node_discretize_intent_tcc))
    workflow.add_node("pick_subintent", timed_traced_node("pick_subintent", node_pick_subintent))
    workflow.add_node("context", timed_traced_node("context", node_context_tcc))
    workflow.add_node("capture_context", timed_traced_node("capture_context", node_capture_context_snapshot))
    workflow.add_node("planner", timed_traced_node("planner", node_planner_tcc))
    workflow.add_node("capture_planner", timed_traced_node("capture_planner", node_capture_planner_snapshot))
    workflow.add_node("advance", timed_traced_node("advance", node_advance_cursor))

    workflow.set_entry_point("discretize")
    workflow.add_edge("discretize", "pick_subintent")
    workflow.add_edge("pick_subintent", "context")
    workflow.add_edge("context", "capture_context")
    workflow.add_edge("capture_context", "planner")
    workflow.add_edge("planner", "capture_planner")
    workflow.add_edge("capture_planner", "advance")
    workflow.add_conditional_edges(
        "advance",
        router_has_more_subintents,
        {
            "more": "pick_subintent",
            "done": END,
        },
    )

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


def print_discretize_result(intent_index: int, intent_text: str, result: dict) -> None:
    print("\n" + "=" * 100)
    print(f"[DISCRETIZE][INTENT {intent_index}] {intent_text}")
    print("=" * 100)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _requires_topology_context(sub: dict) -> bool:
    frame = sub.get("intent_frame") or {}
    arguments = _normalize_arguments(frame.get("arguments") or {})
    topology_arguments = arguments.get("topology_arguments") or {}
    return bool(topology_arguments)


def _format_discretize_subintent(sub: dict) -> dict:
    return {
        "id": sub.get("id"),
        "text": sub.get("text"),
        "atomic": True,
        "depends_on": [],
        "requires_topology_context": _requires_topology_context(sub),
    }


def _format_node_context_output(sub: dict, context: dict, capture_debug: dict, final_state_context: dict) -> dict:
    frame = sub.get("intent_frame") or {}
    arguments = _normalize_arguments(frame.get("arguments") or {})
    topology_arguments = arguments.get("topology_arguments") or {}
    context_present = isinstance(context, dict) and len(context) > 0
    slice_topology = context.get("slice_topology") or {}
    slice_devices = slice_topology.get("devices") or {}
    slice_networks = slice_topology.get("networks") or {}
    interface_names = []
    for dev_meta in slice_devices.values():
        for if_name in ((dev_meta or {}).get("interfaces") or {}).keys():
            if if_name not in interface_names:
                interface_names.append(if_name)
    formatted = {
        "subintent_id": context.get("subintent_id") or sub.get("id"),
        "requested_topology_entities": context.get("requested_topology_entities") or {},
        "matched_topology_entities": context.get("matched_topology_entities") or [],
        "unmatched_topology_arguments": context.get("unmatched_topology_arguments") or [],
        "slice_topology": {
            "devices": sorted(slice_devices.keys()),
            "interfaces": interface_names,
            "networks": sorted(slice_networks.keys()),
        } if slice_devices or slice_networks else {
            "devices": [],
            "interfaces": [],
            "networks": [],
        },
        "confidence": float(context.get("confidence", 0.0) or 0.0),
        "needs_human": bool(context.get("needs_human", False)),
    }

    if DISCRETIZE_UNIT_INCLUDE_DIAGNOSTICS:
        formatted["diagnostics"] = context.get("diagnostics") or {
            "captured_context_missing": not context_present,
            "subintent_topology_argument_count": len(topology_arguments) if isinstance(topology_arguments, dict) else 0,
            "subintent_topology_argument_names": sorted(topology_arguments.keys()) if isinstance(topology_arguments, dict) else [],
        }

    if DISCRETIZE_UNIT_INCLUDE_DEBUG:
        formatted["debug_source"] = {
            "subintent_has_intent_frame": bool(frame),
            "subintent_topology_argument_count": len(topology_arguments) if isinstance(topology_arguments, dict) else 0,
            "subintent_topology_argument_names": sorted(topology_arguments.keys()) if isinstance(topology_arguments, dict) else [],
            "captured_context_present": context_present,
            "captured_context_keys": sorted(context.keys()) if isinstance(context, dict) else [],
            "capture_node_debug": capture_debug if isinstance(capture_debug, dict) else {},
            "final_state_context_present": isinstance(final_state_context, dict),
            "final_state_context_keys": sorted(final_state_context.keys()) if isinstance(final_state_context, dict) else [],
        }

    return formatted


def _format_node_planner_output(planner_result: dict) -> dict:
    if not isinstance(planner_result, dict):
        return {
            "subintent_id": None,
            "objective": "unspecified_goal",
            "apply_operations": [],
            "verify_operations": [],
            "confidence": 0.0,
            "needs_human": True,
        }

    return {
        "subintent_id": planner_result.get("subintent_id"),
        "objective": planner_result.get("objective") or "unspecified_goal",
        "apply_operations": planner_result.get("apply_operations") or [],
        "verify_operations": planner_result.get("verify_operations") or [],
        "confidence": float(planner_result.get("confidence", 0.0) or 0.0),
        "objective_mode": planner_result.get("objective_mode") or "apply",
        "needs_human": bool(planner_result.get("needs_human", False)),
    }


def _format_global_context_output(context: dict) -> dict:
    slice_topology = context.get("slice_topology") or {}
    slice_devices = slice_topology.get("devices") or {}
    slice_networks = slice_topology.get("networks") or {}
    interface_names = []
    for dev_meta in slice_devices.values():
        for if_name in ((dev_meta or {}).get("interfaces") or {}).keys():
            if if_name not in interface_names:
                interface_names.append(if_name)
    return {
        "subintent_ids": context.get("subintent_ids") or [],
        "matched_topology_entities": context.get("matched_topology_entities") or [],
        "slice_topology": {
            "devices": sorted(slice_devices.keys()),
            "interfaces": interface_names,
            "networks": sorted(slice_networks.keys()),
        } if slice_devices or slice_networks else {
            "devices": [],
            "interfaces": [],
            "networks": [],
        },
        "confidence": float(context.get("confidence", 0.0) or 0.0),
        "needs_human": bool(context.get("needs_human", False)),
    }


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


def run_discretize_unit_suite(
    app,
    intents: list[str],
    model_name: str,
    topology: dict,
    run_label: str = "r1",
    configure_router_model: bool = True,
) -> dict:
    if configure_router_model:
        set_tcc_router_model(model_name)

    suite_start = time.perf_counter()
    results = []

    for i, intent_text in enumerate(intents, start=1):
        state = {
            "user_intent_text": intent_text,
            "topology_full": topology,
        }

        start = time.perf_counter()
        try:
            out = app.invoke(
                state,
                config={
                    "configurable": {"thread_id": f"{model_name}-{run_label}-discretize-{i}"},
                    "recursion_limit": 20,
                },
            )
            elapsed = time.perf_counter() - start
        except Exception as exc:
            elapsed = time.perf_counter() - start
            result = {
                "id": i,
                "intent": intent_text,
                "node_discretize_intent": {
                    "subintents": [],
                    "confidence": 0.0,
                    "needs_human": True,
                },
                "node_context": [],
                "node_planner": [],
                "global_context": {},
                "error": str(exc),
            }
            results.append(result)
            print_discretize_result(i, intent_text, result)
            continue

        work = out.get("work") or {}
        subintents = work.get("subintents") or []
        context_results = work.get("context_results") or {}
        planner_results = work.get("planner_results") or {}
        capture_debug_map = work.get("context_capture_debug") or {}
        final_state_context = out.get("context") or {}

        formatted_subintents = []
        formatted_contexts = []
        formatted_planners = []
        for sub in subintents:
            if not isinstance(sub, dict):
                continue
            sid = sub.get("id")
            context = context_results.get(sid) or sub.get("context") or {}
            planner_result = planner_results.get(sid) or sub.get("planner") or {}
            capture_debug = capture_debug_map.get(sid) or {}
            formatted_subintents.append(_format_discretize_subintent(sub))
            formatted_contexts.append(_format_node_context_output(sub, context, capture_debug, final_state_context))
            formatted_planners.append(_format_node_planner_output(planner_result))

        result = {
            "id": i,
            "intent": intent_text,
            "node_discretize_intent": {
                "subintents": formatted_subintents,
                "confidence": 1.0 if formatted_subintents else 0.0,
                "needs_human": bool(out.get("needs_human", False)),
            },
            "node_context": formatted_contexts[0] if len(formatted_contexts) == 1 else formatted_contexts,
            "node_planner": formatted_planners[0] if len(formatted_planners) == 1 else formatted_planners,
            "global_context": _format_global_context_output(final_state_context),
        }

        if DISCRETIZE_UNIT_INCLUDE_DEBUG:
            result["graph_debug"] = {
                "work_has_context_results": isinstance(context_results, dict) and len(context_results) > 0,
                "captured_context_result_ids": sorted(context_results.keys()) if isinstance(context_results, dict) else [],
                "capture_debug_ids": sorted(capture_debug_map.keys()) if isinstance(capture_debug_map, dict) else [],
                "final_state_has_context": isinstance(final_state_context, dict) and len(final_state_context) > 0,
                "final_state_context_keys": sorted(final_state_context.keys()) if isinstance(final_state_context, dict) else [],
            }
        if out.get("error"):
            result["error"] = out.get("error")

        results.append(result)
        print_discretize_result(i, intent_text, result)

    suite_elapsed = time.perf_counter() - suite_start

    report = {
        "mode": "discretize_unit",
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


def _discretize_subintent_texts(result: dict) -> list[str]:
    node = result.get("node_discretize_intent") or {}
    subintents = node.get("subintents") or []
    if not isinstance(subintents, list):
        return []
    return [
        str(sub.get("text") or "").strip()
        for sub in subintents
        if isinstance(sub, dict) and str(sub.get("text") or "").strip()
    ]


def build_discretize_comparison_report(base_report: dict, finetuned_report: dict) -> dict:
    base_results = base_report.get("results") or []
    ft_results = finetuned_report.get("results") or []
    rows = []

    for base_result, ft_result in zip(base_results, ft_results):
        base_texts = _discretize_subintent_texts(base_result)
        ft_texts = _discretize_subintent_texts(ft_result)
        rows.append({
            "id": base_result.get("id"),
            "intent": base_result.get("intent"),
            "base": {
                "subintent_count": len(base_texts),
                "subintents": base_texts,
                "needs_human": (base_result.get("node_discretize_intent") or {}).get("needs_human"),
                "error": base_result.get("error"),
            },
            "finetuned": {
                "subintent_count": len(ft_texts),
                "subintents": ft_texts,
                "needs_human": (ft_result.get("node_discretize_intent") or {}).get("needs_human"),
                "error": ft_result.get("error"),
            },
            "changed": base_texts != ft_texts,
        })

    return {
        "mode": "discretize_unit_comparison",
        "run_label": finetuned_report.get("run_label"),
        "temperature": BENCHMARK_TEMPERATURE,
        "model": f"{base_report.get('model')}__vs__{finetuned_report.get('model')}",
        "base_model": base_report.get("model"),
        "finetuned_model": finetuned_report.get("model"),
        "suite_elapsed_sec": round(
            float(base_report.get("suite_elapsed_sec", 0.0) or 0.0)
            + float(finetuned_report.get("suite_elapsed_sec", 0.0) or 0.0),
            3,
        ),
        "num_intents": min(len(base_results), len(ft_results)),
        "changed_count": sum(1 for row in rows if row["changed"]),
        "results": rows,
    }


def run_local_discretize_ft_comparison(app, intents: list[str], topology: dict, run_label: str = "r1") -> dict:
    print("\n" + "=" * 100)
    print("[LOCAL DISCRETIZE] Loading base model")
    print("=" * 100)
    base_llm = set_tcc_local_model(LOCAL_DISCRETIZE_BASE_MODEL, adapter=None)
    try:
        base_report = run_discretize_unit_suite(
            app,
            intents,
            base_llm.label,
            topology,
            run_label=f"{run_label}-base",
            configure_router_model=False,
        )
    finally:
        base_llm.unload()

    print("\n" + "=" * 100)
    print("[LOCAL DISCRETIZE] Loading fine-tuned adapter")
    print("=" * 100)
    ft_llm = set_tcc_local_model(
        LOCAL_DISCRETIZE_BASE_MODEL,
        adapter=LOCAL_DISCRETIZE_FINETUNED_ADAPTER,
    )
    try:
        finetuned_report = run_discretize_unit_suite(
            app,
            intents,
            ft_llm.label,
            topology,
            run_label=f"{run_label}-finetuned",
            configure_router_model=False,
        )
    finally:
        ft_llm.unload()

    comparison = build_discretize_comparison_report(base_report, finetuned_report)

    print("\n" + "=" * 100)
    print("[BASE VS FINE-TUNED DISCRETIZE COMPARISON]")
    print("=" * 100)
    print(f"BASE      : {comparison.get('base_model')}")
    print(f"FINETUNED : {comparison.get('finetuned_model')}")
    print(f"CHANGED   : {comparison.get('changed_count')}/{comparison.get('num_intents')}")

    for row in comparison.get("results", []):
        print("\n" + "-" * 100)
        print(f"[INTENT {row.get('id')}] {row.get('intent')}")
        print("-" * 100)
        base = row.get("base") or {}
        finetuned = row.get("finetuned") or {}
        print(f"BASE elapsed={base.get('elapsed_sec')}s subintent_count={base.get('subintent_count')} error={base.get('error')}")
        for idx, text in enumerate(base.get("subintents") or [], start=1):
            print(f"  BASE S{idx}: {text}")
        print(f"FINETUNED elapsed={finetuned.get('elapsed_sec')}s subintent_count={finetuned.get('subintent_count')} error={finetuned.get('error')}")
        for idx, text in enumerate(finetuned.get("subintents") or [], start=1):
            print(f"  FT   S{idx}: {text}")
        print(f"changed={row.get('changed')}")

    return comparison


def run_hf_endpoint_discretize_ft_comparison(
    app,
    intents: list[str],
    topology: dict,
    finetuned_endpoint_url: str,
    base_router_model: str = HF_ENDPOINT_BASE_ROUTER_MODEL,
    base_endpoint_url: str | None = None,
    finetuned_label: str = LOCAL_DISCRETIZE_FINETUNED_ADAPTER,
    run_label: str = "r1",
) -> dict:
    if base_endpoint_url:
        print("\n" + "=" * 100)
        print("[HF ENDPOINT DISCRETIZE] Running base endpoint")
        print("=" * 100)
        base_llm = set_tcc_hf_endpoint_model(base_endpoint_url, label=f"hf-endpoint-base:{base_endpoint_url}")
        base_model_label = base_llm.model
        configure_base_router = False
    else:
        print("\n" + "=" * 100)
        print("[HF ROUTER DISCRETIZE] Running base model")
        print("=" * 100)
        set_tcc_router_model(base_router_model)
        base_model_label = base_router_model
        configure_base_router = False

    base_report = run_discretize_unit_suite(
        app,
        intents,
        base_model_label,
        topology,
        run_label=f"{run_label}-base",
        configure_router_model=configure_base_router,
    )

    print("\n" + "=" * 100)
    print("[HF ENDPOINT DISCRETIZE] Running fine-tuned endpoint")
    print("=" * 100)
    ft_llm = set_tcc_hf_endpoint_model(
        finetuned_endpoint_url,
        label=f"hf-endpoint-ft:{finetuned_label}",
    )
    finetuned_report = run_discretize_unit_suite(
        app,
        intents,
        ft_llm.model,
        topology,
        run_label=f"{run_label}-finetuned",
        configure_router_model=False,
    )

    return build_discretize_comparison_report(base_report, finetuned_report)


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

        elif mode_tag == "discretize_unit":
            f.write("\n")
            for result in report.get("results", []):
                f.write("=" * 100 + "\n")
                f.write(f"[DISCRETIZE][INTENT {result.get('id')}] {result.get('intent')}\n")
                f.write("=" * 100 + "\n")
                f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                f.write("\n")

        elif mode_tag == "discretize_unit_comparison":
            f.write(f"base_model={report.get('base_model')}\n")
            f.write(f"finetuned_model={report.get('finetuned_model')}\n")
            f.write(f"changed_count={report.get('changed_count')}\n")
            f.write("\n")
            for result in report.get("results", []):
                f.write("=" * 100 + "\n")
                f.write(f"[COMPARE][INTENT {result.get('id')}] {result.get('intent')}\n")
                f.write("=" * 100 + "\n")
                f.write(f"changed={result.get('changed')}\n")
                f.write("[BASE]\n")
                f.write(json.dumps(result.get("base") or {}, ensure_ascii=False, indent=2) + "\n")
                f.write("[FINETUNED]\n")
                f.write(json.dumps(result.get("finetuned") or {}, ensure_ascii=False, indent=2) + "\n")
                f.write("\n")

    return str(out_path)

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IBN benchmark suites.")
    parser.add_argument(
        "--compare-local-discretize-ft",
        action="store_true",
        help=(
            "Compare node_discretize_intent with the local Llama 3.2 3B base model "
            "against the local PEFT adapter trained with AutoTrain."
        ),
    )
    parser.add_argument(
        "--compare-hf-endpoint-discretize-ft",
        action="store_true",
        help=(
            "Compare node_discretize_intent using HF Router for the base model and a "
            "Hugging Face Inference Endpoint/TGI URL for the fine-tuned model."
        ),
    )
    parser.add_argument(
        "--finetuned-endpoint-url",
        default=os.environ.get("HF_FINETUNED_ENDPOINT_URL"),
        help="Hugging Face endpoint URL for the fine-tuned model. Can also use HF_FINETUNED_ENDPOINT_URL.",
    )
    parser.add_argument(
        "--base-endpoint-url",
        default=os.environ.get("HF_BASE_ENDPOINT_URL"),
        help="Optional Hugging Face endpoint URL for the base model. If omitted, HF Router is used.",
    )
    parser.add_argument(
        "--base-router-model",
        default=HF_ENDPOINT_BASE_ROUTER_MODEL,
        help="HF Router model spec for the base model when --base-endpoint-url is omitted.",
    )
    parser.add_argument(
        "--finetuned-label",
        default=LOCAL_DISCRETIZE_FINETUNED_ADAPTER,
        help="Label used in reports for the fine-tuned endpoint.",
    )
    parser.add_argument(
        "--base-model",
        default=LOCAL_DISCRETIZE_BASE_MODEL,
        help="Base model used by --compare-local-discretize-ft.",
    )
    parser.add_argument(
        "--adapter-model",
        default=LOCAL_DISCRETIZE_FINETUNED_ADAPTER,
        help="PEFT adapter repo used by --compare-local-discretize-ft.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=LOCAL_DISCRETIZE_MAX_NEW_TOKENS,
        help="Generation limit for local and HF endpoint benchmarks.",
    )
    parser.add_argument(
        "--limit-intents",
        type=int,
        default=None,
        help="Limit the number of benchmark intents. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantized loading for the local transformers benchmark.",
    )
    parser.add_argument(
        "--device-map",
        default=LOCAL_DISCRETIZE_DEVICE_MAP,
        help="Device map passed to transformers when CUDA is available. Default: auto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    LOCAL_DISCRETIZE_BASE_MODEL = args.base_model
    LOCAL_DISCRETIZE_FINETUNED_ADAPTER = args.adapter_model
    LOCAL_DISCRETIZE_MAX_NEW_TOKENS = args.max_new_tokens
    HF_ENDPOINT_MAX_NEW_TOKENS = args.max_new_tokens
    LOCAL_DISCRETIZE_LOAD_IN_4BIT = not args.no_4bit
    LOCAL_DISCRETIZE_DEVICE_MAP = args.device_map

    app = build_graph_discretize_unit()
    topology = load_experiment_topology(TOPOLOGY_PATH)

    test_intents = BENCHMARK_INTENTS_DISCRETIZE_UNIT
    if args.limit_intents is not None:
        test_intents = test_intents[: max(0, args.limit_intents)]

    # test_models = [
    #     "meta-llama/Llama-3.2-1B-Instruct:novita",
    #     "meta-llama/Llama-3.3-70B-Instruct:groq",

        # "meta-llama/Llama-3.2-1B-Instruct:novita",
        # "meta-llama/Llama-3.1-8B-Instruct:novita",
        # "meta-llama/Llama-3.3-70B-Instruct:groq",
        # "Qwen/Qwen3-4B-Instruct-2507:nscale",
        # "Qwen/Qwen3.5-35B-A3B:novita",
    # ]
    test_models = [
        "meta-llama/Llama-3.1-8B-Instruct:novita",
    ]

    all_reports = []

    print("\n" + "=" * 100)
    print("[BENCHMARK CONFIG]")
    print("=" * 100)
    print(f"intents={len(test_intents)}")
    print(f"repeats={BENCHMARK_REPEATS}")
    print(f"temperature={BENCHMARK_TEMPERATURE}")
    print(f"models={len(test_models)}")

    if args.compare_local_discretize_ft:
        comparison_report = run_local_discretize_ft_comparison(
            app,
            test_intents,
            topology,
            run_label="r1",
        )
        comparison_json_path = save_benchmark_report(comparison_report)
        comparison_txt_path = save_text_report(comparison_report)
        all_reports.append(comparison_report)

        print("\n" + "#" * 100)
        print("[DISCRETIZE FT COMPARISON SUMMARY]")
        print(f"base_model={comparison_report['base_model']}")
        print(f"finetuned_model={comparison_report['finetuned_model']}")
        print(f"changed_count={comparison_report['changed_count']}/{comparison_report['num_intents']}")
        print(f"suite_elapsed_sec={comparison_report['suite_elapsed_sec']}")
        print(f"json_saved_to={comparison_json_path}")
        print(f"txt_saved_to={comparison_txt_path}")
        print("#" * 100)
        raise SystemExit(0)

    if args.compare_hf_endpoint_discretize_ft:
        if not args.finetuned_endpoint_url:
            raise ValueError(
                "Missing --finetuned-endpoint-url. You can also set HF_FINETUNED_ENDPOINT_URL."
            )
        comparison_report = run_hf_endpoint_discretize_ft_comparison(
            app,
            test_intents,
            topology,
            finetuned_endpoint_url=args.finetuned_endpoint_url,
            base_router_model=args.base_router_model,
            base_endpoint_url=args.base_endpoint_url,
            finetuned_label=args.finetuned_label,
            run_label="r1",
        )
        comparison_json_path = save_benchmark_report(comparison_report)
        comparison_txt_path = save_text_report(comparison_report)
        all_reports.append(comparison_report)

        print("\n" + "#" * 100)
        print("[DISCRETIZE HF ENDPOINT COMPARISON SUMMARY]")
        print(f"base_model={comparison_report['base_model']}")
        print(f"finetuned_model={comparison_report['finetuned_model']}")
        print(f"changed_count={comparison_report['changed_count']}/{comparison_report['num_intents']}")
        print(f"suite_elapsed_sec={comparison_report['suite_elapsed_sec']}")
        print(f"json_saved_to={comparison_json_path}")
        print(f"txt_saved_to={comparison_txt_path}")
        print("#" * 100)
        raise SystemExit(0)

    for model_name in test_models:
        for repeat_idx in range(BENCHMARK_REPEATS):
            run_label = f"r{repeat_idx + 1}"

            print("\n" + "=" * 100)
            print(f"[BENCHMARK] Running model: {model_name} | run={run_label}")
            print("=" * 100)

            if RUN_DISCRETIZE_UNIT_BENCHMARK:
                discretize_report = run_discretize_unit_suite(
                    app,
                    test_intents,
                    model_name,
                    topology,
                    run_label=run_label,
                )
                discretize_json_path = save_benchmark_report(discretize_report)
                discretize_txt_path = save_text_report(discretize_report)
                all_reports.append(discretize_report)

                print("\n" + "#" * 100)
                print(f"[DISCRETIZE SUMMARY] model={discretize_report['model']} | run={discretize_report['run_label']}")
                print(f"[DISCRETIZE SUMMARY] suite_elapsed_sec={discretize_report['suite_elapsed_sec']}")
                print(f"[DISCRETIZE SUMMARY] json_saved_to={discretize_json_path}")
                print(f"[DISCRETIZE SUMMARY] txt_saved_to={discretize_txt_path}")
                print("#" * 100)

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
        elif report.get("mode") == "discretize_unit":
            print(
                f"- mode=discretize_unit | model={report['model']} | run={report.get('run_label')} | "
                f"suite_elapsed_sec={report['suite_elapsed_sec']}"
            )
        else:
            print(
                f"- mode=direct_baseline | model={report['model']} | run={report.get('run_label')} | "
                f"suite_elapsed_sec={report['suite_elapsed_sec']}"
            )

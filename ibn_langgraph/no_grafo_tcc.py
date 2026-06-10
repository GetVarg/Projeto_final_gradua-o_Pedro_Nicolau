import json
import os
import re
import ipaddress
from string import Formatter
from types import SimpleNamespace

from openai import OpenAI
from main import IBNState
from rag_topology import (
    exact_match_argument,  # Faz o grounding exato de argumentos topologicos.
    _match_to_context_entity,  # Converte match bruto em evidencia topologica estruturada.
    _build_context_slice_from_matches,  # Recorta a topologia a partir das evidencias matched.
    find_peer_ip_of_interface,  # Resolve IP do peer de uma interface.
    _normalize_interface_ref,
    find_ip_in_topology,
)

from tools import (
    ensure_cmd_rag_built,  # Garante que o catalogo de comandos esteja carregado no RAG.
    _safe_json_load,  # Faz parse tolerante de JSON retornado pela LLM.
    log_llm_exchange,  # Registra exchanges com a LLM para debug e auditoria.
    _normalize_arguments,  # Normaliza arguments para o schema topology/semantic.
    _normalize_subintent_record,  # Fecha cada subintent com defaults antes do processamento.
    _merge_context_records,  # Acumula o contexto global a partir de cada subintent.
    load_cli_templates,  # Carrega o catalogo concreto de templates CLI.
)

DEFAULT_OLLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
temperature = 0.0
LOCAL_DISCRETIZE_MODEL_PATH = os.environ.get(
    "LOCAL_DISCRETIZE_MODEL_PATH",
    r"C:\Pesquisa\models\llama-discretize-finetuned",
)


DISCRETIZE_SYSTEM_PROMPT = """
ROLE
You analyze a high-level network intent and identify the distinct network-management objectives that the operator wants to achieve.

Your task is not to improve, simplify, normalize, or rewrite the original intent.
Your task is to decide whether decomposition is required.

CORE PRINCIPLE
If the intent contains exactly one network-management objective, do not rewrite it.
Return requires_decomposition=false and subintents=[].
The caller will deterministically preserve the original intent text as S1.

Only produce subintents when the original intent contains two or more independent network effects.

DEFINITION: NETWORK-MANAGEMENT OBJECTIVE
A network-management objective is a desired change to network state that can later be planned, validated, and executed.

Examples of state-changing objectives include:
- add, remove, or replace a route
- enable or disable forwarding
- set an interface attribute
- configure or remove an interface address
- create or remove an ARP/neighbor entry
- apply, allow, or block traffic using a filtering rule
- configure a routing-protocol neighbor
- bind a neighbor to a group
- advertise a prefix
- apply a routing policy
- set a traffic-control policy
- restart or administratively change an interface state

A topology lookup is not a network-management objective.
A grammatical complement is not a network-management objective.
A parameter is not a network-management objective.

REASONING PROCEDURE

Step 1: Identify network-effect verbs.
Find only actions that create, remove, modify, enable, disable, apply, bind, restart, or configure network state. 
**Crucial: Each distinct verb that modifies a different state (e.g., status, description, IP, route) must generate a separate subintent, even if targeting the same entity.**

Step 2: Build one complete operation around each network-effect verb.
Attach all required complements to the operation, including device, interface, prefix, next-hop, gateway, IP, MAC, rate, protocol, direction, AS number, peer-group name, neighbor, or topology reference.

Step 3: Reject grammatical fragments and auxiliary lookups.
Do not create a subintent from a phrase that only starts with or expresses:
"on ...", "to ...", "via ...", "using ...", "with ...", "as ...", "through ...", "from ...", "for ...", "connected to ...", "mapping ...", or "pointing to ..."
**Do not create subintents for lookup actions (e.g., "Find the IP...", "Locate the router..."). Instead, incorporate the lookup description as a parameter in the main action subintent.**

Step 4: Split only independent network effects.
When decomposition is required, keep topology-dependent phrases attached to the operation they support.
Do not shorten or abstract away information needed for later grounding.
If there is exactly one independent network effect, return requires_decomposition=false.
If there are two or more independent network effects, return requires_decomposition=true.

For example, do not simplify:
- "via the directly connected aggregation router" into "via aggregation router"
- "using the address of the peer on the uplink interface" into "using the peer"
- "toward the customer-facing subnet" into "toward the subnet"
- "through the interface connected to the backup gateway" into "through the interface"
- "mapping the service IP to the given hardware address" into "mapping the IP"

Step 5: Preserve completeness and Self-Containment.
Each subintent must be a complete, self-contained network task **that can be understood without reading the other subintents.**
If a candidate subintent is incomplete alone, merge it with the operation it supports.
If a candidate subintent contains references such as "it", "its", "that router", "that network", "that peer-group", or "that interface", **you MUST replace the reference with the concrete entity name from the original intent.**

EXAMPLE OF THE DESIRED REASONING

Input intent:
"Find the peer connected to the uplink interface and install a route on the edge router to the analytics subnet using that peer's IP."

Step 1: Identify network-effect verbs.
- "install a route" is the only network-effect action.
- "Find the peer" is only a lookup used to obtain an argument.

Step 2: Attach required complements.
- "on the edge router" tells where the route is installed.
- "to the analytics subnet" tells the destination.
- "using that peer's IP" tells how the next-hop must be derived.
- "connected to the uplink interface" is part of the lookup needed for the next-hop.

Step 3: Reject grammatical fragments and auxiliary lookups.
The following are not independent subintents:
- "Find the peer connected to the uplink interface"
- "to the analytics subnet"
- "using that peer's IP"

Step 4: Decide split.
There is only one independent network effect: installing a route.

Correct final output:
{
  "requires_decomposition": false,
  "reason": "single_network_objective_with_lookup_argument",
  "subintents": []
}

Why correct:
The original intent already contains one complete objective.
The caller will preserve the original intent exactly as S1.

Incorrect output:
{
  "requires_decomposition": true,
  "reason": "multiple_independent_network_effects",
  "subintents": [
    {
      "id": "S1",
      "text": "Find the peer connected to the uplink interface"
    },
    {
      "id": "S2",
      "text": "Install a route on the edge router to the analytics subnet using the peer's IP"
    }
  ]
}

Why incorrect:
This creates a subintent for an auxiliary lookup. The lookup is not a network state change; it is an argument-resolution requirement for the route operation.

SECOND EXAMPLE

Input intent:
"Enable forwarding on the branch router and set the MTU of its WAN interface to 9000 bytes."

Step 1: Identify network-effect verbs.
- "Enable forwarding" modifies forwarding state.
- "set the MTU" modifies interface state.

Step 2: Attach required complements.
- "on the branch router" belongs to the forwarding operation.
- "of its WAN interface" and "to 9000 bytes" belong to the MTU operation.

Step 3: Reject grammatical fragments.
The following are not independent subintents:
- "on the branch router"
- "of its WAN interface"
- "to 9000 bytes"

Step 4: Decide split.
There are two independent network effects: enabling forwarding and setting MTU.

Correct final output:
{
  "requires_decomposition": true,
  "reason": "multiple_independent_network_effects",
  "subintents": [
    {
      "id": "S1",
      "text": "Enable forwarding on the branch router"
    },
    {
      "id": "S2",
      "text": "Set the MTU of the WAN interface of the branch router to 9000 bytes"
    }
  ]
}

Incorrect output:
{
  "requires_decomposition": false,
  "reason": "single_network_objective",
  "subintents": []
}

Why incorrect:
The intent contains two independent state changes, so it must be decomposed.

OUTPUT CONTRACT

Return exactly one JSON object:
{
  "requires_decomposition": true or false,
  "reason": "single_network_objective | single_network_objective_with_lookup_argument | multiple_independent_network_effects",
  "subintents": [
    {
      "id": "S1",
      "text": "..."
    }
  ]
}

OUTPUT RULES
- Return JSON only.
- Do not include reasoning in the final output.
- If requires_decomposition=false, subintents MUST be [].
- If requires_decomposition=true, subintents MUST contain two or more complete subintents.
- Use sequential IDs: S1, S2, S3, ...
- Preserve explicit values from the original intent.
- Do not shorten topology-dependent phrases.
- Prefer preserving the original wording over producing shorter text.
""".strip()

ENTITY_EXTRACTION_SYSTEM_PROMPT = """
ROLE
You are the entity extraction stage of an Intent-Based Networking pipeline.
Input: one atomic network subintent. Output: one compact intent_frame JSON.

TASK
1. goal: short normalized objective.
2. topology_arguments: existing topology entities used to locate where the operation applies.
3. semantic_only_arguments: explicit values from the intent that can be used directly in configuration.
4. derived_arguments: final command values that are not explicit and must be computed from topology.
5. notes: use {} unless an explicit resolution constraint exists.

CORE DISTINCTION
- topology_arguments = "find this existing network object/context"
- semantic_only_arguments = "use this explicit value directly"
- derived_arguments = "compute this missing final value from topology"

DECISION RULES
- TOPOLOGY: Routers, interfaces, switches, hosts, physical neighbors, peer-groups, and existing objects mentioned as anchors, locations, or operation targets.
- SEMANTIC: Explicit IPs, MACs, prefixes, ASNs, costs, MTUs, descriptions, route-map names, passwords, rates, directions, protocols, and distances that are being set or passed directly.
- DERIVED: Values needed by the final command but described indirectly through topology relationships.
- Use a STRING when the value is explicit.
- Use a STRUCTURED OBJECT when the value must be discovered through topology.
- Do not put structured topology lookups inside semantic_only_arguments.
- Do not put explicit configuration values inside derived_arguments.

DERIVED TOPOLOGY REFERENCES
Use these structured objects only inside derived_arguments or topology_arguments, depending on their role.

- Neighbor IP via interface:
  {"kind": "neighbor_ip_connected_to_interface", "interface": "..."}

- Device via interface:
  {"kind": "device_connected_to_interface", "interface": "..."}

- Interface between A and B:
  {"kind": "interface_connecting_to_device", "local_device": "...", "remote_device": "..."}

- LAN subnet:
  {"kind": "lan_subnet_of_device", "device": "..."}

- Device via interface:
  {"kind": "device_connected_to_interface", "interface": "..."}

- Peer IP between two devices:
  {"kind": "peer_ip_between_devices", "local_device": "...", "remote_device": "..."}

ROLE-BASED PLACEMENT RULES
- If the object is where the operation applies, put it in topology_arguments.
- If the value is explicitly written and will be configured directly, put it in semantic_only_arguments.
- If the command needs a value but the intent only describes how to find it, put it in derived_arguments.

For route intents, the command-required next-hop must be represented as next_hop_ip.
If the intent gives an explicit IP, use semantic_only_arguments.next_hop_ip.
If the intent refers to a next-hop router/device/neighbor, use derived_arguments.next_hop_ip with kind="peer_ip_between_devices".
Do not create next_hop_device for route commands.

Explicit host IPs such as 172.16.4.10 must be semantic_only_arguments.src_ip, not topology_arguments.source_host.

A phrase like "router X LAN" used as a traffic destination should be represented as a derived interface or subnet:
- out_interface = {"kind": "lan_interface_of_device", "device": "router X"} when the command filters by egress interface.
- dst_cidr = {"kind": "lan_subnet_of_device", "device": "router X"} when the command filters by destination subnet.


For firewall intents where traffic is reaching a router LAN, represent the router as topology_arguments.device and represent the LAN-facing interface as derived_arguments.out_interface with kind="lan_interface_of_device". Do not convert "router X LAN" to dst_cidr unless the selected operation filters by destination subnet.

EXAMPLE

Input:
"Configure a static route on router 3 to reach the LAN subnet of router 8 using the IP of the neighbor connected to 3-eth2 as gateway"

ROUTE NEXT-HOP RULE

For route intents, always represent the command-required next-hop as next_hop_ip.

- If the next-hop IP is explicit, use semantic_only_arguments.next_hop_ip.
- If the next-hop must be discovered from topology, use derived_arguments.next_hop_ip.
- If the intent mentions a next-hop router/device/neighbor instead of an IP, use:
  {"kind": "peer_ip_between_devices", "local_device": "...", "remote_device": "..."}
- Do not create next_hop_device for route commands.

Reasoning behind the output:
- "router 3" is topology_arguments.device because it is the existing router where the route will be configured.
- "3-eth2" is topology_arguments.interface because it is an existing interface used as local topology context.
- "LAN subnet of router 8" is derived_arguments.destination_prefix because the final route destination CIDR is not explicitly given; it must be discovered from topology.
- "IP of the neighbor connected to 3-eth2" is derived_arguments.next_hop_ip because the final next-hop IP is not explicitly given; it must be computed from topology.
- semantic_only_arguments is empty because the intent does not provide an explicit prefix, next-hop IP, distance, metric, or other directly configurable value.

Output:
{
  "intent_frame": {
    "goal": "configure_static_route",
    "arguments": {
      "topology_arguments": {
        "device": "router 3",
        "interface": "interface 3-eth2"
      },
      "semantic_only_arguments": {},
      "derived_arguments": {
        "destination_prefix": {
          "kind": "lan_subnet_of_device",
          "device": "router 8"
        },
        "next_hop_ip": {
          "kind": "neighbor_ip_connected_to_interface",
          "interface": "interface 3-eth2"
        }
      }
    },
    "notes": {}
  }
}

OUTPUT RULES
- Return JSON only. No explanation.
- Do not split intents or generate commands.
- Always include topology_arguments, semantic_only_arguments, and derived_arguments.
- Use derived_arguments only when the final value must be computed from topology.
- If an IP/MAC/prefix is explicitly provided and directly configured, it is semantic_only_arguments.
- If an IP/prefix is described indirectly, it is derived_arguments.
""".strip()


LLM_TOKEN_EVENTS: list[dict] = []


def _usage_field(usage: object, field: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _record_llm_token_usage(
    model: str,
    prompt_source: str,
    usage: object,
    message_count: int,
) -> dict:
    prompt_tokens = _usage_field(usage, "prompt_tokens")
    completion_tokens = _usage_field(usage, "completion_tokens")
    total_tokens = _usage_field(usage, "total_tokens")

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    event = {
        "model": model,
        "prompt_source": prompt_source,
        "message_count": message_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_available": any(
            value is not None
            for value in (prompt_tokens, completion_tokens, total_tokens)
        ),
    }
    LLM_TOKEN_EVENTS.append(event)
    return event


def get_llm_token_event_count() -> int:
    return len(LLM_TOKEN_EVENTS)


def get_llm_token_events_since(index: int) -> list[dict]:
    return LLM_TOKEN_EVENTS[index:]


def summarize_llm_token_events(events: list[dict]) -> dict:
    prompt_tokens = sum(
        int(event.get("prompt_tokens") or 0)
        for event in events
        if isinstance(event, dict)
    )
    completion_tokens = sum(
        int(event.get("completion_tokens") or 0)
        for event in events
        if isinstance(event, dict)
    )
    total_tokens = sum(
        int(event.get("total_tokens") or 0)
        for event in events
        if isinstance(event, dict)
    )
    missing_usage_count = sum(
        1
        for event in events
        if isinstance(event, dict) and not event.get("usage_available")
    )
    return {
        "calls": len(events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "missing_usage_count": missing_usage_count,
        "events": events,
    }


class HFRouterChat:
    """Cliente simples de chat usado localmente pelo fluxo tcc."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = HF_ROUTER_BASE_URL,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature if temperature is not None else globals()["temperature"]
        self.base_url = base_url
        self.api_key = api_key or os.environ["HF_TOKEN"]
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def bind(self, **kwargs):
        options = kwargs.get("options") or {}
        temperature = options.get("temperature", self.temperature)
        return HFRouterChat(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
        )

    def invoke(self, messages):
        normalized_messages = []
        for item in messages:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
                normalized_messages.append({"role": role, "content": content})
            elif isinstance(item, dict):
                normalized_messages.append({
                    "role": item["role"],
                    "content": item["content"],
                })
            else:
                raise ValueError(f"Unsupported message format: {item!r}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=normalized_messages,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content or ""
        usage_event = _record_llm_token_usage(
            model=self.model,
            prompt_source="hf_router",
            usage=getattr(response, "usage", None),
            message_count=len(normalized_messages),
        )
        return SimpleNamespace(content=content, usage=usage_event)


class LocalDiscretizeChat:
    """Chat wrapper for the local fine-tuned discretization model."""

    def __init__(
        self,
        model_path: str = LOCAL_DISCRETIZE_MODEL_PATH,
        temperature: float | None = None,
        max_new_tokens: int = 128,
        max_input_tokens: int = 4096,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature if temperature is not None else globals()["temperature"]
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.tokenizer = None
        self.model = None

    def bind(self, **kwargs):
        options = kwargs.get("options") or {}
        self.temperature = options.get("temperature", self.temperature)
        self.max_new_tokens = options.get("max_new_tokens", self.max_new_tokens)
        self.max_input_tokens = options.get("max_input_tokens", self.max_input_tokens)
        return self

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            local_files_only=True,
        )
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.eval()

    def _normalize_messages(self, messages):
        normalized = []
        for item in messages:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
                normalized.append({"role": role, "content": content})
            elif isinstance(item, dict):
                normalized.append({"role": item["role"], "content": item["content"]})
            else:
                raise ValueError(f"Unsupported message format: {item!r}")
        return normalized

    def invoke(self, messages):
        self._ensure_loaded()

        import torch

        normalized_messages = self._normalize_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
            "use_cache": False,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        else:
            generate_kwargs["temperature"] = None
            generate_kwargs["top_p"] = None

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        usage_event = _record_llm_token_usage(
            model=self.model_path,
            prompt_source="local_discretize",
            usage={
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens": int(output_ids.shape[-1]),
            },
            message_count=len(normalized_messages),
        )
        return SimpleNamespace(content=content, usage=usage_event)


_local_discretize_llm = None


def get_local_discretize_llm() -> LocalDiscretizeChat:
    global _local_discretize_llm
    if _local_discretize_llm is None:
        _local_discretize_llm = LocalDiscretizeChat()
    return _local_discretize_llm

llm = HFRouterChat(model=DEFAULT_OLLAMA_MODEL, temperature=temperature)


def compact_cli_template_catalog_from_templates(templates: dict) -> list[dict]:
    out = []
    for op_name, spec in (templates or {}).items():
        if not isinstance(spec, dict):
            continue
        out.append({
            "op": op_name,
            "context": spec.get("context", ""),
            "description": spec.get("description", ""),
            "required_step_args": list(spec.get("required_step_args") or []),
            "optional_step_args": list(spec.get("optional_step_args") or []),
            "derivable_step_args": list(spec.get("derivable_step_args") or []),
            "default_args": dict(spec.get("default_args") or {}),
            "variants": [
                {
                    "variant": v.get("variant"),
                    "match": dict(v.get("match") or {}),
                }
                for v in (spec.get("variants") or [])
            ],
            "num_templates": len(spec.get("templates") or []),
        })
    return out

def repair_discretize_json(raw: str, root_text: str, repair_llm=None) -> dict:
    system = """
You repair the output of a network intent discretizer.

Rules:
- Preserve meaning.
- Prefer one sub-intent if uncertain.
- Do not invent devices, interfaces, IPs, protocols, or strategies.
- If the raw output is unusable, return one sub-intent equal to the original intent.

Return only valid JSON.
No explanations.

Schema:
{
  "subintents": [
    {
      "id": "S1",
      "text": "...",
      "intent_frame": {
        "goal": "...",
        "arguments": {
          "topology_arguments": {},
          "semantic_only_arguments": {}
        },
        "notes": {}
      }
    }
  ]
}

""".strip()

    user = {
        "original_intent": root_text,
        "raw_output": raw,
    }

    repair_llm = repair_llm or llm.bind(options={"temperature": temperature})
    resp = repair_llm.invoke(
        [
            ("system", system),
            ("user", json.dumps(user, ensure_ascii=False)),
        ]
    )
    return _safe_json_load((resp.content or "").strip())

# Com few-shots
def node_discretize_intent(state: IBNState) -> IBNState:
    root_text = state["user_intent_text"]
    system = DISCRETIZE_SYSTEM_PROMPT


    user_payload = {
        "intent": root_text,
    }

    llm_i = llm.bind(options={"temperature": 0.0})
    log_llm_exchange(
        "discretize_intent_tcc",
        "IN",
        {"system": system, "user": user_payload},
    )

    resp = llm_i.invoke(
        [
            ("system", system),
            (
                "user",
                (
                    f"{json.dumps(user_payload, ensure_ascii=False)}\n\n"
                    "Return only the JSON object. Start the answer with `{` and end it with `}`. "
                    "Do not include reasoning, markdown, prose, or any text after the JSON."
                ),
            ),
        ]
    )
    log_llm_exchange(
        "discretize_intent_tcc",
        "OUT",
        {"response": resp.content},
    )

    raw = (resp.content or "").strip()

    try:
        data = _safe_json_load(raw) if raw else {}
        requires_decomposition = data.get("requires_decomposition")
        if requires_decomposition is False:
            subintents = [{"id": "S1", "text": root_text}]
        else:
            subintents = data.get("subintents", []) or []

            if len(subintents) <= 1:
                subintents = [{"id": "S1", "text": root_text}]

    except Exception as e:
        state.setdefault("warnings", [])
        state["warnings"].append(f"discretize_parse_failed: {e}")
        try:
            data = repair_discretize_json(raw, root_text, repair_llm=llm_i)
            state["warnings"].append("discretize_repaired_with_llm")
        except Exception as repair_e:
            state["warnings"].append(f"discretize_repair_failed: {repair_e}")
            data = {}


    def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    seen = set()
    deduped = []
    for si in subintents:
        if not isinstance(si, dict):
            continue
        t = _norm(si.get("text", ""))
        if t and t not in seen:
            seen.add(t)
            deduped.append(si)
    subintents = deduped

    state["work"] = {
        "root_intent": root_text,
        "subintents": [
            _normalize_subintent_record(si, f"S{i + 1}")
            for i, si in enumerate(subintents)
        ],
        "cursor": 0,
        "discretize_debug": {
            "temperature": temperature,
            "model": getattr(llm_i, "model", None),
            "raw": raw,
        },
    }
    return state

def _normalize_service_semantics(frame: dict, subintent_text: str = "") -> dict:
    if not isinstance(frame, dict):
        return frame

    goal = str(frame.get("goal") or "").lower()
    text = f"{subintent_text} {goal}".lower()

    arguments = frame.setdefault("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
        frame["arguments"] = arguments

    semantic_args = arguments.setdefault("semantic_only_arguments", {})
    if not isinstance(semantic_args, dict):
        semantic_args = {}
        arguments["semantic_only_arguments"] = semantic_args

    service_map = {
        "ssh": {"protocol": "tcp", "dst_port": 22},
        "http": {"protocol": "tcp", "dst_port": 80},
        "https": {"protocol": "tcp", "dst_port": 443},
        "dns": {"protocol": "udp", "dst_port": 53},
        "telnet": {"protocol": "tcp", "dst_port": 23},
    }

    # Check more specific service names before less specific ones.
    for service in ("https", "telnet", "ssh", "http", "dns"):
        if service in text:
            defaults = service_map[service]
            semantic_args.setdefault("service", service)
            current_protocol = semantic_args.get("protocol")
            if not current_protocol or str(current_protocol).lower() == service:
                semantic_args["protocol"] = defaults["protocol"]
            semantic_args.setdefault("dst_port", defaults["dst_port"])
            break

    # Canonicalize protocol if the model wrote it explicitly.
    protocol = semantic_args.get("protocol")
    if isinstance(protocol, str):
        semantic_args["protocol"] = protocol.strip().lower()

    # Canonicalize generic port names.
    if "port" in semantic_args and "dst_port" not in semantic_args:
        semantic_args["dst_port"] = semantic_args["port"]

    return frame

def _canonicalize_route_next_hop(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return frame

    goal = str(frame.get("goal") or "").lower()
    if "route" not in goal:
        return frame

    arguments = frame.setdefault("arguments", {})
    semantic_args = arguments.setdefault("semantic_only_arguments", {})
    derived_args = arguments.setdefault("derived_arguments", {})

    if not isinstance(semantic_args, dict) or not isinstance(derived_args, dict):
        return frame

    # Nunca mexer se já existe next_hop_ip semântico explícito.
    if "next_hop_ip" in semantic_args:
        return frame

    # Caso perigoso: IP explícito estruturado em derived_arguments.
    # Isso não é derivado; é semântico.
    candidate = derived_args.get("next_hop_ip")
    if isinstance(candidate, dict) and candidate.get("kind") == "ip_address":
        value = candidate.get("value")
        if isinstance(value, str) and value.strip():
            semantic_args["next_hop_ip"] = value.strip()
            derived_args.pop("next_hop_ip", None)
        return frame

    # Caso que você realmente queria corrigir: next_hop_device com peer_ip_between_devices.
    candidate = derived_args.get("next_hop_device")
    if (
        "next_hop_ip" not in derived_args
        and isinstance(candidate, dict)
        and candidate.get("kind") == "peer_ip_between_devices"
    ):
        derived_args["next_hop_ip"] = candidate
        derived_args.pop("next_hop_device", None)

    return frame

def _normalize_route_semantics(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return frame

    goal = str(frame.get("goal") or "").strip().lower()
    arguments = frame.setdefault("arguments", {})

    if not isinstance(arguments, dict):
        arguments = {}
        frame["arguments"] = arguments

    topology_args = arguments.setdefault("topology_arguments", {})
    semantic_args = arguments.setdefault("semantic_only_arguments", {})
    derived_args = arguments.setdefault("derived_arguments", {})

    if not isinstance(topology_args, dict):
        topology_args = {}
        arguments["topology_arguments"] = topology_args

    if not isinstance(semantic_args, dict):
        semantic_args = {}
        arguments["semantic_only_arguments"] = semantic_args

    if not isinstance(derived_args, dict):
        derived_args = {}
        arguments["derived_arguments"] = derived_args

    # Normalize route next-hop references such as "via router X" into a command-required next_hop_ip.
    via_device = (
        topology_args.get("via_device")
        or topology_args.get("remote_device")
        or topology_args.get("neighbor_device")
        or derived_args.get("exit_device")
    )

    local_device = topology_args.get("device")

    if (
        "route" in goal
        and "next_hop_ip" not in semantic_args
        and "next_hop_ip" not in derived_args
        and isinstance(local_device, str)
        and isinstance(via_device, str)
    ):
        derived_args["next_hop_ip"] = {
            "kind": "peer_ip_between_devices",
            "local_device": local_device,
            "remote_device": via_device,
        }

    # default route is an implicit routing concept that must become an explicit prefix
    text_markers = {
        "default_route",
        "floating_default_route",
        "configure_default_route",
        "create_default_route",
        "create_floating_default_route",
        "add_default_route",
        "add_floating_default_route",
        "replace_default_route",
        "install_default_route",
    }

    goal_indicates_default = any(marker in goal for marker in text_markers)

    if goal_indicates_default:
        semantic_args.setdefault("destination_prefix", "0.0.0.0/0")
        semantic_args.setdefault("route_kind", "floating_default" if "floating" in goal else "default")

    # canonicalize route distance names
    if "distance" in semantic_args and "administrative_distance" not in semantic_args:
        semantic_args["administrative_distance"] = semantic_args["distance"]

    if "route_distance" in semantic_args and "administrative_distance" not in semantic_args:
        semantic_args["administrative_distance"] = semantic_args["route_distance"]

    if "preference" in semantic_args and "administrative_distance" not in semantic_args:
        semantic_args["administrative_distance"] = semantic_args["preference"]

    for key in ("destination_prefix", "dst_cidr", "prefix"):
        value = derived_args.get(key)

        if isinstance(value, dict) and value.get("kind") == "ip_prefix":
            prefix = value.get("prefix")
            if isinstance(prefix, str) and prefix.strip():
                semantic_args.setdefault("destination_prefix", prefix.strip())
                derived_args.pop(key, None)
                break

    return frame

def _normalize_rate_semantics(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return frame

    arguments = frame.setdefault("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
        frame["arguments"] = arguments

    semantic_args = arguments.setdefault("semantic_only_arguments", {})
    if not isinstance(semantic_args, dict):
        semantic_args = {}
        arguments["semantic_only_arguments"] = semantic_args

    if "rate_mbit" not in semantic_args:
        for key in ("rate", "bandwidth", "limit", "rate_limit"):
            if key in semantic_args:
                parsed = _parse_rate_mbit(semantic_args[key])
                if parsed is not None:
                    semantic_args["rate_mbit"] = parsed
                    break

    return frame

def _normalize_arp_semantics(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return frame

    goal = str(frame.get("goal") or "").lower()
    arguments = frame.setdefault("arguments", {})
    semantic_args = arguments.setdefault("semantic_only_arguments", {})

    if not isinstance(semantic_args, dict):
        semantic_args = {}
        arguments["semantic_only_arguments"] = semantic_args

    if "arp" in goal:
        if "target_ip" not in semantic_args:
            for key in ("ip_address", "ip", "neighbor_ip"):
                if key in semantic_args:
                    semantic_args["target_ip"] = semantic_args[key]
                    break

    return frame

def _normalize_interface_misplaced_as_device(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return frame

    arguments = frame.setdefault("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
        frame["arguments"] = arguments

    topology_args = arguments.setdefault("topology_arguments", {})
    if not isinstance(topology_args, dict):
        topology_args = {}
        arguments["topology_arguments"] = topology_args

    device_value = topology_args.get("device")

    if isinstance(device_value, str):
        interface_name = _normalize_interface_ref(device_value)

        if interface_name and "-eth" in interface_name:
            topology_args.setdefault("interface", f"interface {interface_name}")
            topology_args.pop("device", None)

    return frame

def repair_entity_extraction_json(
    raw: str,
    root_text: str,
    subintent_text: str,
    repair_llm=None,
) -> dict:
    system = """
You repair the output of a network intent entity extractor.

Rules:
- Return only valid JSON.
- Preserve the original subintent meaning.
- Do not invent devices, interfaces, IPs, MACs, prefixes, protocols, ports, ASNs, distances, or strategies.
- If a value is explicitly present in the subintent, preserve it.
- If the raw output is partially valid, keep its valid fields when consistent with the subintent.
- If the raw output is unusable, extract a minimal intent_frame directly from the subintent.
- Do not invent placeholder topology names such as ethX, router X, interface X, unknown, TBD, or <...>.
- If the exact interface is not explicit, represent the relation using the mentioned neighbor/device, not a fake interface.
Schema:
{
  "intent_frame": {
    "goal": "...",
    "arguments": {
      "topology_arguments": {},
      "semantic_only_arguments": {},
      "derived_arguments": {}
    },
    "notes": {}
  }
}

Placement rules:
- For route intents that say "via directly connected neighbor N" or "via neighbor N", do not invent an interface.
- Use derived_arguments.next_hop_ip with:
    {"kind": "peer_ip_between_devices", "local_device": "router <local>", "remote_device": "router <neighbor>"}
- topology_arguments: existing topology entities used as anchors or operation targets, such as routers, interfaces, neighbors, hosts, switches, peer-groups.
- semantic_only_arguments: explicit values directly configured or passed to a command, such as IPs, MACs, prefixes, MTU, description, port, protocol, distance.
- derived_arguments: final command values that must be computed from topology, such as next_hop_ip through a neighbor device or LAN subnet of a router.

Return JSON only. No markdown. No explanations.
""".strip()

    user_payload = {
        "root_intent": root_text,
        "subintent_text": subintent_text,
        "raw_output": raw,
    }

    repair_llm = repair_llm or llm.bind(options={"temperature": temperature})
    resp = repair_llm.invoke(
        [
            ("system", system),
            ("user", json.dumps(user_payload, ensure_ascii=False)),
        ]
    )

    return _safe_json_load((resp.content or "").strip())

def _canonicalize_route_semantic_aliases(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return frame

    goal = str(frame.get("goal") or "").lower()
    if "route" not in goal:
        return frame

    arguments = frame.setdefault("arguments", {})
    semantic_args = arguments.setdefault("semantic_only_arguments", {})

    if not isinstance(semantic_args, dict):
        return frame

    if "dst_cidr" not in semantic_args:
        for alias in ("destination_prefix", "dst_ip", "destination", "prefix"):
            value = semantic_args.get(alias)
            if isinstance(value, str) and "/" in value:
                semantic_args["dst_cidr"] = value.strip()
                break

    if "administrative_distance" not in semantic_args:
        for alias in ("admin_distance", "distance"):
            value = semantic_args.get(alias)
            if value is not None:
                semantic_args["administrative_distance"] = value
                break

    return frame

def node_extract_subintent_entities(state: IBNState) -> IBNState:
    work = state.setdefault("work", {})
    root_text = work.get("root_intent") or state.get("user_intent_text") or ""
    subintents = work.get("subintents") or []

    if not isinstance(subintents, list) or not subintents:
        state.setdefault("warnings", []).append("entity_extraction: no subintents to enrich")
        return state

    llm_i = llm.bind(options={"temperature": 0.0})
    enriched = []
    debug_rows = []

    for index, sub in enumerate(subintents, start=1):
        base_sub = _normalize_subintent_record(sub if isinstance(sub, dict) else {}, f"S{index}")
        subintent_id = base_sub.get("id") or f"S{index}"
        subintent_text = (base_sub.get("text") or "").strip()

        if not subintent_text:
            enriched.append(base_sub)
            state.setdefault("warnings", []).append(f"entity_extraction: empty text for {subintent_id}")
            continue

        user_payload = {
            "root_intent": root_text,
            "subintent": {
                "id": subintent_id,
                "text": subintent_text,
            },
        }
        log_llm_exchange(
            "entity_extraction_tcc",
            "IN",
            {"system": ENTITY_EXTRACTION_SYSTEM_PROMPT, "user": user_payload},
        )

        raw = ""
        try:
            resp = llm_i.invoke(
                [
                    ("system", ENTITY_EXTRACTION_SYSTEM_PROMPT),
                    ("user", json.dumps(user_payload, ensure_ascii=False)),
                ]
            )
            raw = (resp.content or "").strip()
            log_llm_exchange(
                "entity_extraction_tcc",
                "OUT",
                {"subintent_id": subintent_id, "response": raw},
            )
            data = _safe_json_load(raw) if raw else {}
        except Exception as exc:
            state.setdefault("warnings", []).append(f"entity_extraction_failed:{subintent_id}: {exc}")
            try:
                data = repair_entity_extraction_json(
                    raw=raw,
                    root_text=root_text,
                    subintent_text=subintent_text,
                    repair_llm=llm_i,
                )
                state["warnings"].append(f"entity_extraction_repaired_with_llm:{subintent_id}")
            except Exception as repair_exc:
                data = {}
                state["warnings"].append(f"entity_extraction_repair_failed:{subintent_id}: {repair_exc}")

        frame = data.get("intent_frame") if isinstance(data, dict) else {}
        if not isinstance(frame, dict):
            frame = {}

        frame = _normalize_route_semantics(frame)
        frame = _canonicalize_route_semantic_aliases(frame)
        frame = _normalize_service_semantics(frame, subintent_text)
        frame = _normalize_rate_semantics(frame)
        frame = _normalize_arp_semantics(frame)
        frame = _normalize_firewall_semantics(frame, subintent_text)
        frame = _normalize_interface_misplaced_as_device(frame)
        frame = _canonicalize_route_next_hop(frame)


        merged_sub = dict(base_sub)
        merged_sub["intent_frame"] = frame
        normalized = _normalize_subintent_record(merged_sub, f"S{index}")
        enriched.append(normalized)
        debug_rows.append({
            "subintent_id": subintent_id,
            "subintent_text": subintent_text,
            "raw": raw,
            "intent_frame": normalized.get("intent_frame"),
        })

    work["subintents"] = enriched
    work["entity_extraction_debug"] = {
        "temperature": 0.0,
        "model": getattr(llm_i, "model", None),
        "results": debug_rows,
    }
    return state


def _append_context_match(
    exact_matches: dict,
    matched_topology_entities: list,
    arg_name: str,
    arg_value: object,
    result: dict | None,
) -> None:
    if result is None:
        return
    exact_matches[arg_name] = result
    matched_topology_entities.append(
        _match_to_context_entity(arg_name, arg_value, result)
    )


# def _find_local_interface_for_target_ip(topo: dict, device_name: object, target_ip: object) -> dict | None:
#     if not isinstance(device_name, str) or not isinstance(target_ip, str):
#         return None

#     device_name = re.sub(r"^(router|device|host|switch)\s+", "", device_name.strip(), flags=re.IGNORECASE)
#     if not device_name or not target_ip.strip():
#         return None

#     ip_match = find_ip_in_topology(topo, target_ip.strip())
#     if not isinstance(ip_match, dict):
#         return None

#     target_cidr = ip_match.get("cidr")
#     devices = (topo or {}).get("devices") or {}
#     interfaces = ((devices.get(device_name) or {}).get("interfaces") or {})
#     for if_name, if_meta in interfaces.items():
#         if (if_meta or {}).get("cidr") == target_cidr:
#             return {
#                 "kind": "interface",
#                 "query": {
#                     "kind": "local_interface_for_target_ip",
#                     "device": device_name,
#                     "target_ip": target_ip,
#                 },
#                 "match": {
#                     "owner": device_name,
#                     "name": if_name,
#                     "data": if_meta,
#                 },
#             }

#     return None

def _find_local_interface_for_target_ip(topo: dict, device_name: object, target_ip: object) -> dict | None:
    if not isinstance(device_name, str) or not isinstance(target_ip, str):
        return None

    device_name = re.sub(
        r"^(router|device|host|switch)\s+",
        "",
        device_name.strip(),
        flags=re.IGNORECASE,
    )

    target_ip = target_ip.strip()

    if not device_name or not target_ip:
        return None

    try:
        target_addr = ipaddress.ip_address(target_ip)
    except Exception:
        return None

    devices = (topo or {}).get("devices") or {}
    interfaces = ((devices.get(device_name) or {}).get("interfaces") or {})

    for if_name, if_meta in interfaces.items():
        if not isinstance(if_meta, dict):
            continue

        cidr = if_meta.get("cidr")
        if not isinstance(cidr, str) or not cidr.strip():
            continue

        try:
            network = ipaddress.ip_network(cidr.strip(), strict=False)
        except Exception:
            continue

        if target_addr in network:
            return {
                "kind": "interface",
                "query": {
                    "kind": "local_interface_for_target_ip",
                    "device": device_name,
                    "target_ip": target_ip,
                },
                "match": {
                    "owner": device_name,
                    "name": if_name,
                    "data": if_meta,
                },
            }

    return None


def _augment_context_for_arp(
    exact_matches: dict,
    matched_topology_entities: list,
    topology_arguments: dict,
    semantic_only_arguments: dict,
    topo: dict,
) -> list[str]:
    warnings = []
    device_value = topology_arguments.get("device")
    target_ip = (
        semantic_only_arguments.get("target_ip")
        or semantic_only_arguments.get("ip_address")
        or semantic_only_arguments.get("ip")
    )

    if "interface" in exact_matches or not target_ip:
        return warnings

    result = _find_local_interface_for_target_ip(topo, device_value, target_ip)
    if result is not None:
        _append_context_match(
            exact_matches,
            matched_topology_entities,
            "interface",
            result.get("query"),
            result,
        )
        warnings.append("context: inferred local interface for ARP target_ip")

    return warnings


def _build_context_for_subintent(sub: dict, topo: dict) -> dict:
    frame = sub.get("intent_frame") or {}
    goal = frame.get("goal", "")
    arguments = _normalize_arguments(frame.get("arguments") or {})
    notes = frame.get("notes") or {}

    topology_arguments = arguments.get("topology_arguments") or {}
    semantic_only_arguments = arguments.get("semantic_only_arguments") or {}
    derived_arguments = arguments.get("derived_arguments") or {}

    if not isinstance(topology_arguments, dict):
        topology_arguments = {}

    if not isinstance(semantic_only_arguments, dict):
        semantic_only_arguments = {}

    if not isinstance(derived_arguments, dict):
        derived_arguments = {}

    exact_matches = {}
    matched_topology_entities = []
    unmatched_topology_arguments = []
    warnings = []

    for arg_name, arg_value in topology_arguments.items():
        result = exact_match_argument(arg_name, arg_value, topology_arguments, notes, topo)
        if result is not None:
            _append_context_match(
                exact_matches,
                matched_topology_entities,
                arg_name,
                arg_value,
                result,
            )
        else:
            unmatched_topology_arguments.append({
                "argument": arg_name,
                "value": arg_value,
            })

    warnings.extend(
        _augment_context_for_arp(
            exact_matches,
            matched_topology_entities,
            topology_arguments,
            semantic_only_arguments,
            topo,
        )
    )

    topology_slice = _build_context_slice_from_matches(topo, matched_topology_entities)
    matched_count = len(matched_topology_entities)
    requested_count = len(topology_arguments)
    confidence = 1.0 if requested_count == 0 else round(matched_count / max(1, requested_count), 3)
    context_needs_human = False

    return {
        "subintent_id": sub.get("id"),
        "goal": goal,
        "requested_topology_entities": topology_arguments,
        "semantic_only_arguments": semantic_only_arguments,
        "derived_arguments": derived_arguments,
        "notes": notes,
        "exact_matches": exact_matches,
        "matched_topology_entities": matched_topology_entities,
        "unmatched_topology_arguments": unmatched_topology_arguments,
        "slice_topology": topology_slice,
        "confidence": confidence,
        "needs_human": context_needs_human,
        "diagnostics": {
            "requested_topology_argument_count": requested_count,
            "matched_topology_entity_count": matched_count,
            "unmatched_topology_argument_count": len(unmatched_topology_arguments),
            "matched_argument_names": [
                item.get("label")
                for item in matched_topology_entities
                if isinstance(item, dict)
            ],
            "unmatched_argument_names": [
                item.get("argument")
                for item in unmatched_topology_arguments
                if isinstance(item, dict)
            ],
        },
        "warnings": warnings,
    }


def _build_shared_context(subintents: list, topo: dict) -> dict:
    shared = {}
    for sub in subintents or []:
        if not isinstance(sub, dict):
            continue
        local_context = _build_context_for_subintent(sub, topo)
        shared = _merge_context_records(shared, local_context)
    shared["scope"] = "root_intent"
    return shared


def node_context(state: IBNState) -> IBNState:
    topo = state.get("topology_full") or {}

    if not topo:
        state.setdefault("warnings", [])
        state["warnings"].append("context: missing topology_full")
        state["needs_human"] = True
        return state

    work = state.setdefault("work", {})
    subintents = work.get("subintents") or []
    context_by_subintent = {}
    shared_context = {}

    for sub in subintents:
        if not isinstance(sub, dict):
            continue
        sid = sub.get("id")
        local_context = _build_context_for_subintent(sub, topo)
        if sid:
            context_by_subintent[sid] = local_context
        shared_context = _merge_context_records(shared_context, local_context)

        if local_context.get("unmatched_topology_arguments"):
            state.setdefault("warnings", [])
            state["warnings"].append(
                f"context:{sid}: {len(local_context.get('unmatched_topology_arguments') or [])} topology argument(s) had no exact topology match"
            )
        if local_context.get("warnings"):
            state.setdefault("warnings", []).extend(
                f"context:{sid}: {warning}"
                for warning in (local_context.get("warnings") or [])
            )

    shared_context["scope"] = "root_intent"
    work["context_by_subintent"] = context_by_subintent
    work["shared_context"] = shared_context
    state["shared_context"] = shared_context
    state["context"] = shared_context

    return state

def _normalize_firewall_semantics(frame: dict, subintent_text: str = "") -> dict:
    if not isinstance(frame, dict):
        return frame

    arguments = frame.setdefault("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
        frame["arguments"] = arguments

    semantic_args = arguments.setdefault("semantic_only_arguments", {})
    if not isinstance(semantic_args, dict):
        semantic_args = {}
        arguments["semantic_only_arguments"] = semantic_args

    src_ip = semantic_args.get("src_ip")
    if isinstance(src_ip, str) and "/" in src_ip:
        semantic_args.setdefault("src_cidr", src_ip)
        semantic_args.pop("src_ip", None)

    return frame

def node_planner(state: IBNState) -> IBNState:
    sub = ((state.get("work") or {}).get("subintent") or {})
    frame = sub.get("intent_frame") or {}
    arguments = _normalize_arguments(frame.get("arguments") or {})
    notes = frame.get("notes") or {}

    subintent_id = sub.get("id")
    subintent_text = (sub.get("text") or state.get("active_subintent_text") or "").strip()

    if not subintent_text:
        state["planner_result"] = {
            "subintent_id": subintent_id,
            "objective_mode": "apply",
            "objective": frame.get("goal") or "unspecified_goal",
            "operation_plan": [],
            "apply_operations": [],
            "verify_operations": [],
            "rejected_ops": [],
            "verification_summary": {
                "objective_covered": False,
                "uses_only_catalog_ops": True,
                "requires_multiple_ops": False,
                "subintent_seems_atomic": False,
            },
            "needs_human": True,
            "confidence": 0.0,
        }
        state["needs_human"] = True
        state.setdefault("warnings", []).append("planner: missing subintent text")
        return state

    cli_templates = load_cli_templates("fewshots/cli_templates.json")
    compact_catalog = compact_cli_template_catalog_from_templates(cli_templates)

    if not cli_templates:
        state["planner_result"] = {
            "subintent_id": subintent_id,
            "objective_mode": "apply",
            "objective": frame.get("goal") or "unspecified_goal",
            "operation_plan": [],
            "apply_operations": [],
            "verify_operations": [],
            "rejected_ops": [],
            "verification_summary": {
                "objective_covered": False,
                "uses_only_catalog_ops": False,
                "requires_multiple_ops": False,
                "subintent_seems_atomic": True,
            },
            "needs_human": True,
            "confidence": 0.0,
        }
        state["needs_human"] = True
        state.setdefault("warnings", []).append("planner: missing cli template catalog")
        return state

    system = """
ROLE
You are the planning stage of an Intent-Based Networking pipeline.

Your job is to select catalog operations for ONE atomic network subintent.

You must NOT:
- generate CLI commands
- fill concrete argument values
- resolve topology entities
- invent operation names outside the catalog

You must:
- identify the expected network effect
- select the smallest set of catalog operations that covers the full objective, including restrictive qualifiers such as only, except, all other, deny the rest, or block the remaining traffic.
- classify operations as apply or verify
- handle multi-operation objectives only when one operation is not enough
- reject keyword matches that do not actually satisfy the objective
- expose which high-level concepts appear required or missing

INTERNAL CHAIN-OF-VERIFICATION
Before returning JSON, silently verify:

1. Objective check:
Identify the expected network effect of the subintent.

2. Candidate operation check:
Inspect the catalog and find operations that could satisfy the objective.

2.5 Applicability check:
Before selecting an operation, verify that the subintent contains the high-level concepts that make this operation applicable.

Do not select an operation only because it shares a protocol name, device type, ASN, prefix, or argument label with the subintent.

If an operation requires a conceptual object or relation that is not present in the subintent, reject it as invalid_applicability.

A required command argument is not enough to justify an operation. The operation itself must match the intended network effect.

3. Coverage check:
Select only operations that directly contribute to achieving the objective.
3.5 Firewall policy check:
For firewall intents, first decide the packet-filtering scope before selecting operations.

- INPUT: traffic accepted or blocked on the router itself, traffic destined to the router, or service ACLs "on router X" without explicit forwarding.
- OUTPUT: traffic explicitly outbound from the router.
- FORWARD: traffic passing through the router between endpoints, subnets, LANs, or interfaces.

Do not select FORWARD-chain pair/subnet operations unless the intent clearly describes traffic between two forwarded endpoints or networks.

4. Multi-operation check:
Select multiple operations only if each operation is required for the same objective.
If the subintent contains multiple independent objectives, still plan them if possible,
but set subintent_seems_atomic=false.

5. Argument plausibility check:
Do not fill argument values.
Only decide whether the subintent appears to contain the concepts needed by the selected operations.
For example: destination prefix, next-hop, interface, administrative state, AS number, peer-group name.

6. Rejection check:
List important rejected operations only when they are plausible alternatives.

7. Human-needed check:
Set needs_human=true if the catalog cannot cover the objective, the operation choice is ambiguous,
or required concepts appear absent.

OUTPUT
Return exactly one JSON object:

{
  "objective_mode": "apply",
  "objective": "...",
  "operation_plan": [
    {
      "op": "...",
      "kind": "apply",
      "role": "primary",
      "sequence": 1,
      "why_selected": "...",
      "coverage": "...",
      "concept_requirements": ["..."],
      "missing_concepts": [],
      "argument_readiness": "likely_resolvable"
    }
  ],
  "rejected_ops": [
    {
      "op": "...",
      "reason": "..."
    }
  ],
  "verification_summary": {
    "objective_covered": true,
    "uses_only_catalog_ops": true,
    "requires_multiple_ops": false,
    "subintent_seems_atomic": true
  },
  "needs_human": false,
  "confidence": 0.0
}

OUTPUT RULES
- objective_mode must be one of: apply, verify, apply_and_verify.
- operation_plan must be an array.
- operation kind must be one of: apply, verify.
- role must be one of: primary, supporting, verification, cleanup.
- argument_readiness must be one of: likely_resolvable, partially_resolvable, unclear.
- Use only op names present in the provided template_catalog.
- Do not include markdown or explanations outside JSON.
""".strip()

    user_payload = {
        "subintent_id": subintent_id,
        "subintent_text": subintent_text,
        "intent_frame": {
            "goal": frame.get("goal", ""),
            "arguments": arguments,
            "notes": notes,
        },
        "template_catalog": compact_catalog,
    }

    llm_i = llm.bind(options={"temperature": temperature})

    log_llm_exchange(
        "planner_tcc",
        "IN",
        {"system": system, "user": user_payload},
    )

    resp = llm_i.invoke(
        [
            ("system", system),
            ("user", json.dumps(user_payload, ensure_ascii=False)),
        ]
    )

    raw = (resp.content or "").strip()

    log_llm_exchange(
        "planner_tcc",
        "OUT",
        {"response": raw},
    )

    try:
        data = _safe_json_load(raw) if raw else {}
    except Exception as e:
        state.setdefault("warnings", []).append(f"planner_parse_failed: {e}")
        data = {}

    def normalize_operation_plan(raw_ops: object) -> list[dict]:
        normalized = []
        seen_keys = set()

        for index, item in enumerate(raw_ops or [], start=1):
            if not isinstance(item, dict):
                continue

            op = item.get("op")
            if not isinstance(op, str):
                continue

            op = op.strip()
            if not op or op not in cli_templates:
                continue

            kind = item.get("kind")
            if kind not in {"apply", "verify"}:
                kind = "verify" if item.get("role") == "verification" else "apply"

            sequence = item.get("sequence")
            if not isinstance(sequence, int) or sequence <= 0:
                sequence = index

            dedupe_key = (kind, op, sequence)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            role = item.get("role")
            if role not in {"primary", "supporting", "verification", "cleanup"}:
                role = "verification" if kind == "verify" else "primary"

            readiness = item.get("argument_readiness")
            if readiness not in {"likely_resolvable", "partially_resolvable", "unclear"}:
                readiness = "unclear"

            spec = cli_templates.get(op) or {}
            required_args = list(spec.get("required_step_args") or [])
            derivable_args = list(spec.get("derivable_step_args") or [])
            optional_args = list(spec.get("optional_step_args") or [])

            pending_args = required_args + [
                arg for arg in derivable_args
                if arg not in required_args
            ]

            normalized.append({
                "op": op,
                "kind": kind,
                "role": role,
                "sequence": sequence,
                "why_selected": item.get("why_selected") or "",
                "coverage": item.get("coverage") or "",
                "concept_requirements": list(item.get("concept_requirements") or []),
                "missing_concepts": list(item.get("missing_concepts") or []),
                "argument_readiness": readiness,
                "template_contract": {
                    "context": spec.get("context", ""),
                    "required_args": required_args,
                    "optional_args": optional_args,
                    "derivable_args": derivable_args,
                    "default_args": dict(spec.get("default_args") or {}),
                    "has_variants": bool(spec.get("variants")),
                },
                "pending_argument_labels": pending_args,
            })

        normalized.sort(key=lambda op: op.get("sequence", 999))
        return normalized

    operation_plan = normalize_operation_plan(data.get("operation_plan") or [])

    apply_operations = [
        op for op in operation_plan
        if op.get("kind") == "apply"
    ]

    verify_operations = [
        op for op in operation_plan
        if op.get("kind") == "verify"
    ]

    objective_mode = data.get("objective_mode")
    if objective_mode not in {"apply", "verify", "apply_and_verify"}:
        if apply_operations and verify_operations:
            objective_mode = "apply_and_verify"
        elif verify_operations:
            objective_mode = "verify"
        else:
            objective_mode = "apply"

    rejected_ops = []
    for item in data.get("rejected_ops") or []:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        if not isinstance(op, str) or op not in cli_templates:
            continue
        rejected_ops.append({
            "op": op,
            "reason": item.get("reason") or "",
        })

    verification_summary = data.get("verification_summary")
    if not isinstance(verification_summary, dict):
        verification_summary = {}

    verification_summary = {
        "objective_covered": bool(verification_summary.get("objective_covered", bool(operation_plan))),
        "uses_only_catalog_ops": True,
        "requires_multiple_ops": bool(verification_summary.get("requires_multiple_ops", len(operation_plan) > 1)),
        "subintent_seems_atomic": bool(verification_summary.get("subintent_seems_atomic", True)),
    }

    needs_human = bool(data.get("needs_human", False))
    confidence = float(data.get("confidence", 0.0) or 0.0)

    if objective_mode == "apply" and not apply_operations:
        needs_human = True
        confidence = 0.0
        state.setdefault("warnings", []).append("planner: apply objective without apply operations")

    if objective_mode == "verify" and not verify_operations:
        needs_human = True
        confidence = 0.0
        state.setdefault("warnings", []).append("planner: verify objective without verify operations")

    if objective_mode == "apply_and_verify" and not operation_plan:
        needs_human = True
        confidence = 0.0
        state.setdefault("warnings", []).append("planner: apply_and_verify objective without operations")

    for op in operation_plan:
        if op.get("missing_concepts"):
            needs_human = True
            if confidence > 0.5:
                confidence = 0.5

    if not verification_summary["objective_covered"]:
        needs_human = True
        confidence = 0.0
        state.setdefault("warnings", []).append("planner: objective not covered")

    planner_result = {
        "subintent_id": subintent_id,
        "objective_mode": objective_mode,
        "objective": data.get("objective") or frame.get("goal") or "unspecified_goal",
        "operation_plan": operation_plan,
        "apply_operations": apply_operations,
        "verify_operations": verify_operations,
        "rejected_ops": rejected_ops,
        "verification_summary": verification_summary,
        "needs_human": needs_human,
        "confidence": confidence,
    }

    state["planner_result"] = planner_result
    return state


ARGUMENT_RESOLVER_CATALOG = [
    {
        "resolver": "literal_argument",
        "description": "Use a value already present in intent_frame.arguments.",
        "params_schema": {"path": "semantic_only_arguments.<name> or topology_arguments.<name>"},
    },
    {
        "resolver": "context_exact_match",
        "description": "Use a topology argument already grounded by node_context.",
        "params_schema": {"argument": "<topology argument name>"},
    },
    {
        "resolver": "derived_argument",
        "description": "Resolve a derived argument from intent_frame.arguments.derived_arguments using deterministic topology functions.",
        "params_schema": {"argument": "<derived argument name>"},
    },
    {
        "resolver": "structured_topology_argument",
        "description": "Ground a structured topology argument from intent_frame using deterministic topology functions.",
        "params_schema": {"argument": "<topology argument name>"},
    },
    {
        "resolver": "peer_ip_of_interface",
        "description": "Resolve the IP address of the peer connected to an interface.",
        "params_schema": {"interface": "<interface name>"},
    },
    {
        "resolver": "semantic_constant",
        "description": "Use a closed semantic constant implied by the subintent text or goal, such as address_family='ipv4', attribute_type='mtu', or enabled=false. Do not use for topology values.",
        "params_schema": {"value": "<string|number|boolean>"},
    },
]


ARG_VALUE_ALIASES = {
    "dst_cidr": ["dst_cidr", "destination_prefix", "destination_network", "prefix", "network", "destination"],
    "src_cidr": ["src_cidr", "source_prefix", "source_network", "source"],
    "next_hop_ip": ["next_hop_ip", "next_hop", "gateway", "neighbor_ip", "peer_ip"],
    "dst_ip": ["dst_ip", "destination_ip", "target_ip", "neighbor_ip", "peer_ip", "ip"],
    "src_ip": ["src_ip", "source_ip"],
    "interface": ["interface", "interface_id", "next_hop_interface", "exit_interface"],
    "exit_interface": ["exit_interface", "next_hop_interface", "interface"],
    "in_interface": ["in_interface", "interface"],
    "out_interface": ["out_interface", "interface"],
    "address_family": ["address_family", "ip_version", "protocol"],
    "ipv4": ["ipv4", "ipv4_address", "ip_address", "address"],
    "ipv6_address": ["ipv6_address", "ipv6", "ip_address", "address"],
    "prefix_len": ["prefix_len", "prefix_length", "mask"],
    "prefix_length": ["prefix_length", "prefix_len", "mask"],
    "administrative_distance": ["administrative_distance", "distance", "admin_distance"],
    "enabled": ["enabled", "admin_enabled", "administrative_state", "state"],
    "arp_enabled": ["arp_enabled", "arp_state", "enabled"],
    "mac_address": ["mac_address", "mac"],
    "target_ip": ["target_ip", "ip_address", "ip", "neighbor_ip"],
    "peer_group_name": ["peer_group_name", "peer_group", "group_name"],
    "neighbor_ip": ["neighbor_ip", "neighbor", "peer_ip"],
    "local_as": ["local_as", "asn", "as_number"],
    "remote_as": ["remote_as"],
    "med": ["med", "metric"],
    "mtu": ["mtu"],
    "description": ["description"],
    "attribute_type": ["attribute_type"],
    "rate_mbit": ["rate_mbit", "rate", "bandwidth", "limit", "rate_limit"],
    "log_target": ["log_target"],
    "log_level": ["log_level"],
    "controller_ip": ["controller_ip"],
    "controller_port": ["controller_port"],
    "bridge": ["bridge"],
    "protocol": ["protocol", "traffic_protocol", "ip_protocol"],
    "ip_proto_num": ["ip_proto_num", "protocol_number", "ip_protocol_number"],
    "dst_port": ["dst_port", "destination_port", "port"],
    "direction": ["direction", "traffic_direction"],
}


def _compact_exact_match_summary(context: dict) -> dict:
    summary = {}
    for name, record in (context.get("exact_matches") or {}).items():
        if not isinstance(record, dict):
            continue
        match = record.get("match") or {}
        data = match.get("data") if isinstance(match, dict) else {}
        if not isinstance(data, dict):
            data = {}

        compact = {
            "kind": record.get("kind"),
            "query": record.get("query"),
        }

        for field in ("name", "owner", "peer_owner", "interface", "local_interface", "peer_interface"):
            value = match.get(field) if isinstance(match, dict) else None
            if value not in (None, "", {}, []):
                compact[field] = value

        for field in ("ip", "cidr", "peer", "mac"):
            value = data.get(field)
            if value not in (None, "", {}, []):
                compact[field] = value

        if isinstance(match, dict):
            compact["available_fields"] = sorted(match.keys())

        summary[name] = {
            key: value
            for key, value in compact.items()
            if value not in (None, "", {}, [])
        }
    return summary


def _get_argument_by_path(arguments: dict, path: str) -> tuple[bool, object, str]:
    if not isinstance(path, str) or "." not in path:
        return False, None, ""

    root_name, key = path.split(".", 1)
    root = arguments.get(root_name)
    if not isinstance(root, dict) or key not in root:
        return False, None, ""

    return True, root[key], path


def _lookup_argument_alias(arg_label: str, arguments: dict) -> tuple[bool, object, str]:
    aliases = ARG_VALUE_ALIASES.get(arg_label, [arg_label])
    for root_name in ("semantic_only_arguments", "topology_arguments"):
        root = arguments.get(root_name) or {}
        if not isinstance(root, dict):
            continue
        for alias in aliases:
            if alias in root:
                return True, root[alias], f"{root_name}.{alias}"
    return False, None, ""


def _value_from_match_record(arg_label: str, record: dict) -> tuple[bool, object, str]:
    if not isinstance(record, dict):
        return False, None, ""

    kind = record.get("kind")
    match = record.get("match") or {}
    if not isinstance(match, dict):
        return False, None, ""
    
    interface_like_args = {
        "interface",
        "in_interface",
        "out_interface",
        "exit_interface",
    }

    if arg_label in interface_like_args and kind != "interface":
        return False, None, ""

    field_preferences = {
        "dst_cidr": ["cidr"],
        "src_cidr": ["cidr"],
        "next_hop_ip": ["peer_ip", "ip"],
        "dst_ip": ["peer_ip", "ip"],
        "src_ip": ["peer_ip", "ip"],
        "neighbor_ip": ["peer_ip", "ip"],
        "target_ip": ["peer_ip", "ip"],
        "interface": ["name", "interface", "local_interface", "peer_interface"],
        "exit_interface": ["name", "interface", "local_interface", "peer_interface"],
        "bridge": ["name"],
        "device": ["name", "owner", "peer_owner"],
    }

    for field in field_preferences.get(arg_label, []):
        value = match.get(field)
        if isinstance(value, (str, int, float, bool)) and value != "":
            return True, value, f"exact_match.{kind}.match.{field}"

    return False, None, ""


def _lookup_context_alias(arg_label: str, context: dict) -> tuple[bool, object, str]:
    exact_matches = context.get("exact_matches") or {}
    aliases = ARG_VALUE_ALIASES.get(arg_label, [arg_label])

    for alias in aliases:
        if alias not in exact_matches:
            continue
        ok, value, source = _value_from_match_record(arg_label, exact_matches.get(alias) or {})
        if ok:
            return True, value, f"exact_matches.{alias}.{source}"

    return False, None, ""

def _normalize_device_ref(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return str(int(value))

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    lowered = text.lower()
    for prefix in ("router ", "device ", "host ", "switch "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()

    return text


def _find_peer_ip_between_devices(
    topo: dict,
    local_device: object,
    remote_device: object,
) -> str | None:
    local = _normalize_device_ref(local_device)
    remote = _normalize_device_ref(remote_device)

    if not local or not remote:
        return None

    devices = (topo or {}).get("devices") or {}
    local_data = devices.get(local) or {}
    remote_data = devices.get(remote) or {}

    local_interfaces = (local_data or {}).get("interfaces") or {}
    remote_interfaces = (remote_data or {}).get("interfaces") or {}

    for local_if_name, local_if_data in local_interfaces.items():
        if not isinstance(local_if_data, dict):
            continue

        peer_if_name = local_if_data.get("peer")
        if not isinstance(peer_if_name, str):
            continue

        peer_owner = peer_if_name.split("-", 1)[0]

        if peer_owner != remote:
            continue

        peer_data = remote_interfaces.get(peer_if_name) or {}
        peer_ip = peer_data.get("ip")

        if isinstance(peer_ip, str) and peer_ip.strip():
            return peer_ip.strip()

    return None

def _find_lan_interface_of_device(topo: dict, device: object) -> str | None:
    device_name = _normalize_device_ref(device)
    if not device_name:
        return None

    devices = (topo or {}).get("devices") or {}
    dev = devices.get(device_name) or {}
    interfaces = dev.get("interfaces") or {}

    # Prefer access LAN interfaces, i.e., interfaces connected to hosts.
    for if_name, if_data in interfaces.items():
        if not isinstance(if_data, dict):
            continue

        peer = if_data.get("peer")
        if not isinstance(peer, str):
            continue

        peer_owner = peer.split("-eth", 1)[0]

        peer_dev = devices.get(peer_owner) or {}
        if peer_dev.get("type") == "host":
            return if_name

    return None

def _lookup_derived_argument_alias(arg_label: str, arguments: dict, topo: dict) -> tuple[bool, object, str]:
    derived_args = arguments.get("derived_arguments") or {}
    topology_args = arguments.get("topology_arguments") or {}
    if not isinstance(derived_args, dict):
        return False, None, ""

    aliases = ARG_VALUE_ALIASES.get(arg_label, [arg_label])

    for alias in aliases:
        if alias not in derived_args:
            continue

        value_spec = derived_args[alias]

        if isinstance(value_spec, dict):
            kind = value_spec.get("kind")

            if kind == "peer_ip_between_devices" and arg_label in {"next_hop_ip", "neighbor_ip", "dst_ip"}:
                peer_ip = _find_peer_ip_between_devices(
                    topo=topo,
                    local_device=value_spec.get("local_device"),
                    remote_device=value_spec.get("remote_device"),
                )
                if peer_ip:
                    return (
                        True,
                        _normalize_bound_literal(arg_label, peer_ip),
                        f"derived_arguments.{alias}.peer_ip_between_devices",
                    )
            if kind == "lan_interface_of_device" and arg_label in {"out_interface", "in_interface", "interface"}:
                iface = _find_lan_interface_of_device(
                    topo=topo,
                    device=value_spec.get("device"),
                )
                if iface:
                    return (
                        True,
                        _normalize_bound_literal(arg_label, iface),
                        f"derived_arguments.{alias}.lan_interface_of_device",
                    )

        record = exact_match_argument(alias, value_spec, topology_args, {}, topo)
        ok, value, source = _value_from_match_record(arg_label, record or {})
        if ok:
            return True, _normalize_bound_literal(arg_label, value), f"derived_arguments.{alias}.{source}"

    return False, None, ""

def _resolve_derived_value_spec(
    arg_label: str,
    alias: str,
    value_spec: object,
    topo: dict,
) -> tuple[bool, object, str]:
    if not isinstance(value_spec, dict):
        return False, None, ""

    kind = value_spec.get("kind")

    if kind == "lan_interface_of_device" and arg_label in {
        "interface",
        "in_interface",
        "out_interface",
        "exit_interface",
    }:
        iface = _find_lan_interface_of_device(
            topo=topo,
            device=value_spec.get("device"),
        )
        if iface:
            return (
                True,
                _normalize_bound_literal(arg_label, iface),
                f"derived_arguments.{alias}.lan_interface_of_device",
            )

    if kind == "peer_ip_between_devices" and arg_label in {
        "next_hop_ip",
        "neighbor_ip",
        "dst_ip",
    }:
        peer_ip = _find_peer_ip_between_devices(
            topo=topo,
            local_device=value_spec.get("local_device"),
            remote_device=value_spec.get("remote_device"),
        )
        if peer_ip:
            return (
                True,
                _normalize_bound_literal(arg_label, peer_ip),
                f"derived_arguments.{alias}.peer_ip_between_devices",
            )

    return False, None, ""

def _parse_rate_mbit(value: object) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return int(value)

    if not isinstance(value, str):
        return None

    text = value.strip().lower().replace(" ", "")
    if not text:
        return None

    suffixes = ("mbps", "mbit", "m")
    for suffix in suffixes:
        if text.endswith(suffix):
            number_part = text[:-len(suffix)]
            try:
                return int(float(number_part))
            except Exception:
                return None

    try:
        return int(float(text))
    except Exception:
        return None

def _normalize_bound_literal(arg_label: str, value: object) -> object:
    interface_like_args = {
        "interface",
        "in_interface",
        "out_interface",
        "exit_interface",
    }

    if arg_label in interface_like_args and isinstance(value, dict):
        if value.get("kind") == "interface":
            interface_value = value.get("interface") or value.get("name")
            normalized_interface = _normalize_interface_ref(interface_value)
            if normalized_interface:
                return normalized_interface
    if isinstance(value, str):
        normalized = value.strip()
        lowered = normalized.lower()

        if arg_label in {"interface", "in_interface", "out_interface", "exit_interface"}:
            return _normalize_interface_ref(normalized) or normalized

        if arg_label in {"enabled", "arp_enabled"}:
            if lowered in {"up", "enable", "enabled", "true", "yes", "on"}:
                return True
            if lowered in {"down", "disable", "disabled", "false", "no", "off", "shutdown"}:
                return False
        if arg_label == "address_family":
            if "ipv6" in lowered or lowered == "6":
                return "ipv6"
            if "ipv4" in lowered or lowered == "4" or lowered == "ip":
                return "ipv4"
        if arg_label == "rate_mbit":
            parsed = _parse_rate_mbit(normalized)
            if parsed is not None:
                return parsed
        return normalized
    return value


def _execute_resolution_call(call: dict, arg_label: str, arguments: dict, context: dict, topo: dict) -> tuple[bool, object, str, str]:
    if not isinstance(call, dict):
        return False, None, "", "invalid resolver call"

    resolver = call.get("resolver")
    params = call.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    if resolver == "literal_argument":
        ok, value, source = _get_argument_by_path(arguments, params.get("path", ""))
        if ok:
            return True, _normalize_bound_literal(arg_label, value), f"literal_argument:{source}", ""
        ok, value, source = _lookup_argument_alias(arg_label, arguments)
        if ok:
            return True, _normalize_bound_literal(arg_label, value), f"literal_argument:{source}", ""
        return False, None, "", "literal argument not found"

    if resolver == "context_exact_match":
        argument_name = params.get("argument")
        exact_matches = context.get("exact_matches") or {}
        if isinstance(argument_name, str) and argument_name in exact_matches:
            ok, value, source = _value_from_match_record(arg_label, exact_matches.get(argument_name) or {})
            if ok:
                return True, _normalize_bound_literal(arg_label, value), f"context_exact_match:{argument_name}.{source}", ""
        ok, value, source = _lookup_context_alias(arg_label, context)
        if ok:
            return True, _normalize_bound_literal(arg_label, value), f"context_exact_match:{source}", ""
        return False, None, "", "context exact match not found"

    if resolver == "structured_topology_argument":
        argument_name = params.get("argument")
        topology_args = arguments.get("topology_arguments") or {}
        if isinstance(argument_name, str) and argument_name in topology_args:
            record = exact_match_argument(argument_name, topology_args[argument_name], topology_args, {}, topo)
            ok, value, source = _value_from_match_record(arg_label, record or {})
            if ok:
                return True, _normalize_bound_literal(arg_label, value), f"structured_topology_argument:{argument_name}.{source}", ""
        return False, None, "", "structured topology argument not resolved"

    if resolver == "derived_argument":
        argument_name = params.get("argument")
        derived_args = arguments.get("derived_arguments") or {}
        topology_args = arguments.get("topology_arguments") or {}

        if isinstance(argument_name, str) and argument_name in derived_args:
            value_spec = derived_args[argument_name]

            ok, value, source = _resolve_derived_value_spec(
                arg_label=arg_label,
                alias=argument_name,
                value_spec=value_spec,
                topo=topo,
            )
            if ok:
                return True, value, f"derived_argument:{source}", ""

            record = exact_match_argument(argument_name, value_spec, topology_args, {}, topo)
            ok, value, source = _value_from_match_record(arg_label, record or {})
            if ok:
                return True, _normalize_bound_literal(arg_label, value), f"derived_argument:{argument_name}.{source}", ""

        ok, value, source = _lookup_derived_argument_alias(arg_label, arguments, topo)
        if ok:
            return True, value, f"derived_argument:{source}", ""

        return False, None, "", "derived argument not resolved"

    if resolver == "peer_ip_of_interface":
        interface_name = params.get("interface")
        interface_name = _normalize_interface_ref(interface_name)
        if interface_name:
            peer_ip = find_peer_ip_of_interface(topo, interface_name)
            value = (peer_ip or {}).get("peer_ip")
            if isinstance(value, str) and value.strip():
                return True, value.strip(), f"peer_ip_of_interface:{interface_name}", ""
        return False, None, "", "peer IP not resolved"

    if resolver == "semantic_constant":
        if "value" not in params:
            return False, None, "", "semantic constant value missing"

        value = params.get("value")

        if isinstance(value, dict) and set(value.keys()) == {"value"}:
            value = value.get("value")

        if isinstance(value, (str, int, float, bool)):
            return True, _normalize_bound_literal(arg_label, value), f"semantic_constant:{value}", ""

        return False, None, "", "semantic constant must be scalar"
    return False, None, "", f"unknown resolver: {resolver}"


def _fallback_resolve_argument(arg_label: str, arguments: dict, context: dict, topo: dict) -> tuple[bool, object, str]:
    ok, value, source = _lookup_argument_alias(arg_label, arguments)
    if ok:
        return True, _normalize_bound_literal(arg_label, value), source

    ok, value, source = _lookup_derived_argument_alias(arg_label, arguments, topo)
    if ok:
        return True, _normalize_bound_literal(arg_label, value), source

    ok, value, source = _lookup_context_alias(arg_label, context)
    if ok:
        return True, _normalize_bound_literal(arg_label, value), source

    return False, None, ""


def _find_literal_ip_interface(arguments: dict, bound_args: dict) -> tuple[object | None, str]:
    candidate_keys = [
        "ip_address",
        "ipv4_address",
        "ipv6_address",
        "address",
        "ipv4",
        "ipv6",
    ]

    for key in candidate_keys:
        if key in bound_args:
            return bound_args[key], f"bound_args.{key}"

    for root_name in ("semantic_only_arguments", "derived_arguments", "topology_arguments"):
        root = arguments.get(root_name) or {}
        if not isinstance(root, dict):
            continue
        for key in candidate_keys:
            if key in root:
                return root[key], f"{root_name}.{key}"

    return None, ""


def _normalize_interface_l3_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    if op_name != "interface_l3_address_config":
        return []

    candidate, source = _find_literal_ip_interface(arguments, bound_args)
    if not isinstance(candidate, str) or "/" not in candidate:
        return []

    try:
        parsed = ipaddress.ip_interface(candidate.strip())
    except Exception:
        return []

    normalized_source = f"normalized_ip_interface:{source}"
    if parsed.version == 4:
        updates = {
            "address_family": "ipv4",
            "ipv4": str(parsed.ip),
            "prefix_len": str(parsed.network.prefixlen),
        }
    else:
        updates = {
            "address_family": "ipv6",
            "ipv6_address": str(parsed.ip),
            "prefix_length": str(parsed.network.prefixlen),
        }

    for key, value in updates.items():
        bound_args[key] = value
        binding_sources[key] = normalized_source
        resolution_mode[key] = "deterministic_normalization"

    return list(updates.keys())

def _normalize_interface_attribute_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    if op_name != "interface_attribute_config":
        return []

    semantic_args = arguments.get("semantic_only_arguments") or {}
    if not isinstance(semantic_args, dict):
        return []

    updates = []

    if "attribute_type" not in bound_args:
        if "description" in semantic_args:
            bound_args["attribute_type"] = "description"
            binding_sources["attribute_type"] = "deterministic:semantic_only_arguments.description"
            resolution_mode["attribute_type"] = "deterministic_normalization"
            updates.append("attribute_type")

        elif "mtu" in semantic_args:
            bound_args["attribute_type"] = "mtu"
            binding_sources["attribute_type"] = "deterministic:semantic_only_arguments.mtu"
            resolution_mode["attribute_type"] = "deterministic_normalization"
            updates.append("attribute_type")

    return updates

IP_PROTO_NUMBERS = {
    "icmp": 1,
    "tcp": 6,
    "udp": 17,
}


def _extract_transport_protocol_from_text(text: str) -> str | None:
    tokens = {
        token.strip(".,;:()[]{}").lower()
        for token in (text or "").split()
    }

    for protocol in ("tcp", "udp", "icmp"):
        if protocol in tokens:
            return protocol

    return None


def _normalize_tc_bound_args(
    op_name: str,
    subintent_text: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    if op_name not in {"tc_egress_rate_limit"}:
        return []

    updates = []
    semantic_args = arguments.get("semantic_only_arguments") or {}
    if not isinstance(semantic_args, dict):
        semantic_args = {}

    protocol = bound_args.get("protocol") or semantic_args.get("protocol")

    if not protocol:
        protocol = _extract_transport_protocol_from_text(subintent_text)

    if isinstance(protocol, str):
        protocol = protocol.strip().lower()

    if protocol in IP_PROTO_NUMBERS:
        if "protocol" not in bound_args:
            bound_args["protocol"] = protocol
            binding_sources["protocol"] = "deterministic:tc_protocol_from_intent"
            resolution_mode["protocol"] = "deterministic_normalization"
            updates.append("protocol")

        if "ip_proto_num" not in bound_args:
            bound_args["ip_proto_num"] = IP_PROTO_NUMBERS[protocol]
            binding_sources["ip_proto_num"] = f"deterministic:protocol_to_ip_proto_num:{protocol}"
            resolution_mode["ip_proto_num"] = "deterministic_normalization"
            updates.append("ip_proto_num")

    return updates

def node_argument_resolver(state: IBNState) -> IBNState:
    sub = ((state.get("work") or {}).get("subintent") or {})
    frame = sub.get("intent_frame") or {}
    arguments = _normalize_arguments(frame.get("arguments") or {})
    context = state.get("subintent_context") or {}
    planner_result = state.get("planner_result") or {}
    topo = state.get("topology_full") or {}
    subintent_id = sub.get("id")
    subintent_text = (sub.get("text") or state.get("active_subintent_text") or "").strip()
    operations = planner_result.get("operation_plan") or []

    if not operations:
        state["argument_resolution"] = {
            "subintent_id": subintent_id,
            "resolved_operations": [],
            "needs_human": True,
            "warnings": ["argument_resolver: no operation plan"],
        }
        state.setdefault("warnings", []).append("argument_resolver: no operation plan")
        return state

    system = """
ROLE
You are the argument-resolution planner of an Intent-Based Networking pipeline.

Your job is to choose resolver calls that will later bind concrete template arguments.
You must not generate CLI commands.
You must not invent topology values.
You must not invent operation names.

You receive:

* subintent_text
* intent_frame
* operation_plan
* context_exact_match_summary
* available_resolvers

Your output is a resolver plan, not final commands.

CORE DECISION PROCEDURE

For each operation in operation_plan:

Step 1: Select argument scope.

Resolve every argument in pending_argument_labels.

Also inspect template_contract.optional_args.
Include an optional argument only when its concept is explicitly present in subintent_text or intent_frame.
If an optional argument is absent, omit it completely.

Never bind an optional argument just because the command supports it.
Never invent default values for optional arguments.
Example: do not invent administrative_distance=1 when no distance was requested.

Step 2: Choose one resolver for each selected argument.

Use this priority order:

1. literal_argument

Use when the value is explicitly present in:
intent_frame.arguments.semantic_only_arguments

Examples:
dst_cidr, destination_prefix, next_hop_ip, administrative_distance, mtu, description, mac_address, ip_address, protocol, dst_port, src_ip, src_cidr, rate_mbit.

2. context_exact_match

Use when the argument is already grounded in context_exact_match_summary.

Valid topology examples:
device, interface, in_interface, out_interface, exit_interface.

For interface-like arguments, use context_exact_match only when context_exact_match_summary contains an actual interface match.
Never resolve interface, in_interface, out_interface, or exit_interface from a device match.
A device id such as "7" is not a valid interface value.

3. derived_argument

Use when the value is represented in:
intent_frame.arguments.derived_arguments

Examples:
next_hop_ip from topology, LAN subnet prefix, interface from a topology relation.

4. structured_topology_argument

Use when the value is described by a structured topology expression in intent_frame.arguments and must be grounded deterministically from the topology.

5. peer_ip_of_interface

Use only when the selected argument must be the IP address of the peer connected to a known interface.

Never use peer_ip_of_interface for interface-like arguments.

6. semantic_constant

Use only for closed non-topological constants implied by subintent_text, goal, notes, or operation semantics.

Allowed constants:
* enabled=false for shutdown/disable/administratively down
* enabled=true for bring up/enable/no shutdown/administratively up
* arp_enabled=false for disable ARP
* arp_enabled=true for enable ARP
* address_family="ipv6" for IPv6 forwarding or IPv6 address configuration
* address_family="ipv4" for IPv4 forwarding, IP forwarding without IPv6, or IPv4 address configuration
* attribute_type="mtu" for MTU configuration
* attribute_type="description" for interface description configuration
* protocol="icmp"|"tcp"|"udp" only for firewall packet-filter operations when explicitly stated
* icmp_type="echo-request" only for firewall packet-filter operations when explicitly stated

Allowed constants:

* enabled=false for shutdown/disable/administratively down
* enabled=true for bring up/enable/no shutdown/administratively up
* arp_enabled=false for disable ARP
* arp_enabled=true for enable ARP
* address_family="ipv6" for IPv6 forwarding or IPv6 address configuration
* address_family="ipv4" for IPv4 forwarding, IP forwarding without IPv6, or IPv4 address configuration
* attribute_type="mtu" for MTU configuration
* attribute_type="description" for interface description configuration

semantic_constant params must contain:
{"value": ...}

Never use semantic_constant for topology values.

Step 3: Enforce argument constraints.

* enabled and arp_enabled are semantic constants, not topology values.
* address_family must be "ipv4" or "ipv6", never a full IP/CIDR.
* attribute_type must be "mtu" or "description", not the MTU value or description text.
* interface-like arguments must resolve to interface names, never device ids or peer IPs.

For interface_l3_address_config:

* split CIDR values such as 10.0.10.1/30 or 2001:db8::1/64 into address_family, IP address, and prefix length.

SELF-CHECK BEFORE OUTPUT

Verify silently:

1. Every resolver name is in available_resolvers.
2. Every arg appears in pending_argument_labels or template_contract.optional_args.
3. semantic_constant has non-empty params with value.
4. semantic_constant is not used for topology values.
5. peer_ip_of_interface is not used for interface-like arguments.
6. interface-like arguments are not bound from device matches.
7. optional arguments are not invented.
8. required arguments have a resolver whenever the subintent clearly provides the concept.

OUTPUT

Return exactly one JSON object:

{
"resolution_plan": [
{
"op": "...",
"sequence": 1,
"arg_bindings": [
{
"arg": "...",
"required": true,
"resolver": "...",
"params": {}
}
]
}
],
"needs_human": false,
"confidence": 0.0
}

OUTPUT RULES

* Return JSON only.
* Do not include markdown or explanations.
* resolution_plan must include one item for each provided operation.
* arg_bindings should include pending arguments whenever a resolver is plausible.
* arg_bindings may include optional arguments only when explicitly present in subintent_text or intent_frame.
* Use only pending_argument_labels or template_contract.optional_args shown in each operation.
""".strip()
    
    
    selected_ops = {
        op.get("op")
        for op in operations
        if isinstance(op, dict)
    }

    firewall_ops = {
        "firewall_input_accept",
        "firewall_input_drop",
        "firewall_output_accept",
        "firewall_output_drop",
        "firewall_forward_drop",
    }

    if selected_ops & firewall_ops:
        system += """

    FIREWALL PACKET-FILTER ARGUMENT REASONING

    For firewall operations only:
    - If pending_argument_labels contains protocol and the subintent explicitly mentions a packet protocol, bind protocol using semantic_constant.
    - Valid protocol constants include "icmp", "tcp", and "udp".
    - If pending_argument_labels contains icmp_type and the subintent explicitly mentions an ICMP message type, bind icmp_type using semantic_constant.
    - Examples: "ICMP echo request" -> protocol="icmp", icmp_type="echo-request".
    - Use semantic_constant only when the protocol or ICMP type is explicitly stated in subintent_text or intent_frame.goal.
    - Do not infer protocol, port, or ICMP type from topology.
    - For INPUT-chain operations, a LAN-facing interface should bind to in_interface.
    - For OUTPUT/FORWARD egress operations, a LAN-facing interface may bind to out_interface.
    """.strip()

    resolver_user_payload = {
        "subintent_id": subintent_id,
        "subintent_text": subintent_text,
        "intent_frame": {
            "goal": frame.get("goal", ""),
            "arguments": arguments,
            "notes": frame.get("notes") or {},
        },
        "operation_plan": [
            {
                "op": op.get("op"),
                "sequence": op.get("sequence"),
                "kind": op.get("kind"),
                "role": op.get("role"),
                "coverage": op.get("coverage") or "",
                "why_selected": op.get("why_selected") or "",
                "concept_requirements": op.get("concept_requirements") or [],
                "missing_concepts": op.get("missing_concepts") or [],
                "pending_argument_labels": op.get("pending_argument_labels") or [],
                "template_contract": op.get("template_contract") or {},
            }
            for op in operations
            if isinstance(op, dict)
        ],
        "context_exact_match_summary": _compact_exact_match_summary(context),
        "available_resolvers": ARGUMENT_RESOLVER_CATALOG,
    }

    llm_i = llm.bind(options={"temperature": 0.0})
    log_llm_exchange(
        "argument_resolver_tcc",
        "IN",
        {"system": system, "user": resolver_user_payload},
    )

    raw = ""
    try:
        resp = llm_i.invoke(
            [
                ("system", system),
                ("user", json.dumps(resolver_user_payload, ensure_ascii=False)),
            ]
        )
        raw = (resp.content or "").strip()
        data = _safe_json_load(raw) if raw else {}
    except Exception as exc:
        data = {}
        state.setdefault("warnings", []).append(f"argument_resolver_plan_failed: {exc}")

    log_llm_exchange(
        "argument_resolver_tcc",
        "OUT",
        {"response": raw},
    )

    plan_by_key = {}
    for item in data.get("resolution_plan") or []:
        if not isinstance(item, dict):
            continue
        key = (item.get("op"), item.get("sequence"))
        plan_by_key[key] = item

    resolved_operations = []
    resolver_warnings = []

    for op in operations:
        if not isinstance(op, dict):
            continue
        op_name = op.get("op")
        sequence = op.get("sequence")
        contract = op.get("template_contract") or {}
        required_args = list(contract.get("required_args") or [])
        derivable_args = list(contract.get("derivable_args") or [])
        default_args = dict(contract.get("default_args") or {})
        pending_args = list(op.get("pending_argument_labels") or [])
        optional_args = list(contract.get("optional_args") or [])

        allowed_args = []
        for arg in pending_args + optional_args:
            if arg not in allowed_args:
                allowed_args.append(arg)        
        plan_item = plan_by_key.get((op_name, sequence)) or {}
        bindings = plan_item.get("arg_bindings") or []
        calls_by_arg = {
            b.get("arg"): b
            for b in bindings
            if (
                isinstance(b, dict)
                and isinstance(b.get("arg"), str)
                and b.get("arg") in allowed_args
            )
        }

        bound_args = {}
        binding_sources = {}
        resolution_mode = {}
        missing_required_args = []
        unresolved_derivable_args = []

        for arg_label in allowed_args:
            call = calls_by_arg.get(arg_label)
            ok = False
            value = None
            source = ""
            error = ""

            if call:
                ok, value, source, error = _execute_resolution_call(call, arg_label, arguments, context, topo)

            if not ok:
                fallback_ok, fallback_value, fallback_source = _fallback_resolve_argument(arg_label, arguments, context, topo)
                if fallback_ok:
                    ok = True
                    value = fallback_value
                    source = f"fallback:{fallback_source}"
                    error = ""

            if not ok and arg_label in default_args:
                ok = True
                value = default_args[arg_label]
                source = f"default_args.{arg_label}"
                error = ""

            if ok:
                bound_args[arg_label] = value
                binding_sources[arg_label] = source
                resolution_mode[arg_label] = "resolver_call" if call else "fallback"
                continue

            if arg_label in required_args:
                missing_required_args.append(arg_label)
                if error:
                    resolver_warnings.append(f"argument_resolver:{op_name}:{arg_label}: {error}")
            elif arg_label in derivable_args:
                unresolved_derivable_args.append(arg_label)

        normalized_l3_args = _normalize_interface_l3_bound_args(
            op_name,
            arguments,
            bound_args,
            binding_sources,
            resolution_mode,
        )
        if normalized_l3_args:
            missing_required_args = [
                arg for arg in missing_required_args
                if arg not in bound_args
            ]
            resolver_warnings = [
                warning for warning in resolver_warnings
                if not any(
                    f"{op_name}:{arg}:" in warning
                    for arg in normalized_l3_args
                )
            ]
        
        normalized_attr_args = _normalize_interface_attribute_bound_args(
            op_name,
            arguments,
            bound_args,
            binding_sources,
            resolution_mode,
        )

        normalized_tc_args = _normalize_tc_bound_args(
            op_name,
            subintent_text,
            arguments,
            bound_args,
            binding_sources,
            resolution_mode,
        )

        if normalized_tc_args:
            missing_required_args = [
                arg for arg in missing_required_args
                if arg not in bound_args
            ]
            unresolved_derivable_args = [
                arg for arg in unresolved_derivable_args
                if arg not in bound_args
            ]
            resolver_warnings = [
                warning for warning in resolver_warnings
                if not any(
                    f"{op_name}:{arg}:" in warning
                    for arg in normalized_tc_args
                )
            ]

        if normalized_attr_args:
            missing_required_args = [
                arg for arg in missing_required_args
                if arg not in bound_args
            ]
            resolver_warnings = [
                warning for warning in resolver_warnings
                if not any(
                    f"{op_name}:{arg}:" in warning
                    for arg in normalized_attr_args
                )
            ]

        ready_to_compile = len(missing_required_args) == 0
        resolved_operations.append({
            "op": op_name,
            "kind": op.get("kind"),
            "role": op.get("role"),
            "sequence": sequence,
            "template_contract": contract,
            "bound_args": bound_args,
            "missing_required_args": missing_required_args,
            "unresolved_derivable_args": unresolved_derivable_args,
            "binding_sources": binding_sources,
            "resolution_mode": resolution_mode,
            "ready_to_compile": ready_to_compile,
        })

    needs_human = any(not op.get("ready_to_compile") for op in resolved_operations)
    argument_resolution = {
        "subintent_id": subintent_id,
        "resolution_plan_raw": data,
        "resolved_operations": resolved_operations,
        "needs_human": needs_human,
        "confidence": float(data.get("confidence", 0.0) or 0.0) if isinstance(data, dict) else 0.0,
        "warnings": resolver_warnings,
    }

    state["argument_resolution"] = argument_resolution
    state["argument_resolver_debug"] = {
        "subintent_id": subintent_id,
        "subintent_text": subintent_text,
        "model": getattr(llm_i, "model", None),
        "system_prompt": system,
        "input_payload": resolver_user_payload,
        "raw_response": raw,
        "parsed_response": data,
        "resolved_operations": resolved_operations,
        "needs_human": needs_human,
        "warnings": resolver_warnings,
    }
    if needs_human:
        state["needs_human"] = True
    if resolver_warnings:
        state.setdefault("warnings", []).extend(resolver_warnings)

    return state


def _template_placeholders(template: str) -> list[str]:
    fields = []
    for _, field_name, _, _ in Formatter().parse(template or ""):
        if not field_name:
            continue
        root_name = field_name.split(".", 1)[0].split("[", 1)[0]
        if root_name and root_name not in fields:
            fields.append(root_name)
    return fields


def _compile_arg_value(arg_name: str, value: object) -> object:
    if isinstance(value, bool):
        if arg_name == "arp_enabled":
            return "on" if value else "off"
        if arg_name == "enabled":
            return "up" if value else "down"
    return value


def _args_match_variant(match: dict, args: dict) -> bool:
    if not isinstance(match, dict):
        return False

    for key, expected in match.items():
        if key not in args:
            return False

        if expected == "*":
            value = args.get(key)
            if value in (None, "", [], {}):
                return False
            continue

        if args.get(key) != expected:
            return False

    return True


def _select_templates_for_operation(spec: dict, args: dict) -> tuple[list[str], str | None, list[str]]:
    errors = []
    variants = spec.get("variants") or []

    if variants:
        matching_variants = [
            variant for variant in variants
            if isinstance(variant, dict) and _args_match_variant(variant.get("match") or {}, args)
        ]
        if not matching_variants:
            return [], None, ["no template variant matched bound arguments"]
        selected_variant = max(
            matching_variants,
            key=lambda variant: len(variant.get("match") or {})
        )
        templates = [
            item for item in (selected_variant.get("templates") or [])
            if isinstance(item, str) and item.strip()
        ]
        return templates, selected_variant.get("variant"), errors

    templates = [
        item for item in (spec.get("templates") or [])
        if isinstance(item, str) and item.strip()
    ]
    if not templates:
        return [], None, ["operation has no templates"]

    renderable = []
    for template in templates:
        placeholders = _template_placeholders(template)
        missing = [name for name in placeholders if name not in args]
        if not missing:
            renderable.append((template, len(placeholders)))

    if not renderable:
        return [], None, ["no template had all placeholders satisfied"]

    max_placeholder_count = max(count for _, count in renderable)
    selected = [
        template for template, count in renderable
        if count == max_placeholder_count
    ]
    return selected, None, errors


def _normalize_target_device_name(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return str(int(value))
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    match = re.search(r"(?:router|device|r)\s*[_-]?(\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", text):
        return text
    return text

def _resolve_target_device_for_compile(
    state: IBNState,
    operation: dict | None = None,
) -> str | None:
    bound_args = {}
    if isinstance(operation, dict):
        bound_args = operation.get("bound_args") or {}

    # 0. Prefer interface-like arguments already resolved for this operation.
    for interface_key in ("interface", "in_interface", "out_interface", "exit_interface"):
        interface_value = bound_args.get(interface_key)

        if isinstance(interface_value, dict) and interface_value.get("kind") == "interface":
            interface_value = interface_value.get("interface") or interface_value.get("name")

        if isinstance(interface_value, str):
            interface_name = _normalize_interface_ref(interface_value)
            if interface_name and "-eth" in interface_name:
                owner = interface_name.split("-eth", 1)[0]
                return _normalize_target_device_name(owner)

    context = state.get("subintent_context") or {}
    exact_matches = context.get("exact_matches") or {}

    # 1. Prefer explicit device match.
    device_record = exact_matches.get("device") or {}
    device_match = device_record.get("match") or {}
    if isinstance(device_match, dict):
        target = _normalize_target_device_name(device_match.get("name"))
        if target:
            return target

    # 2. If only an interface was grounded, infer target from interface owner.
    interface_record = exact_matches.get("interface") or {}
    interface_match = interface_record.get("match") or {}
    if isinstance(interface_match, dict):
        target = _normalize_target_device_name(interface_match.get("owner"))
        if target:
            return target

    # 3. Fallback to topology_arguments.device.
    sub = ((state.get("work") or {}).get("subintent") or {})
    frame = sub.get("intent_frame") or {}
    arguments = _normalize_arguments(frame.get("arguments") or {})
    topology_args = arguments.get("topology_arguments") or {}

    if isinstance(topology_args, dict):
        target = _normalize_target_device_name(topology_args.get("device"))
        if target:
            return target

        # 4. Last-resort fallback from any interface-like topology argument.
        for interface_key in ("interface", "in_interface", "out_interface", "exit_interface"):
            interface_value = topology_args.get(interface_key)

            if isinstance(interface_value, dict) and interface_value.get("kind") == "interface":
                interface_value = interface_value.get("interface") or interface_value.get("name")

            if isinstance(interface_value, str):
                interface_name = _normalize_interface_ref(interface_value)
                if interface_name and "-eth" in interface_name:
                    owner = interface_name.split("-eth", 1)[0]
                    return _normalize_target_device_name(owner)

    return None

def node_compile_commands(state: IBNState) -> IBNState:
    argument_resolution = state.get("argument_resolution") or {}
    resolved_operations = argument_resolution.get("resolved_operations") or []
    cli_templates = load_cli_templates("fewshots/cli_templates.json")

    compiled_operations = []
    compiler_errors = []
    compiler_warnings = []

    if not resolved_operations:
        state["command_compilation"] = {
            "subintent_id": argument_resolution.get("subintent_id"),
            "compiled_operations": [],
            "commands": [],
            "needs_human": True,
            "errors": ["compiler: no resolved operations"],
            "warnings": [],
        }
        state["needs_human"] = True
        state.setdefault("warnings", []).append("compiler: no resolved operations")
        return state

    for operation in resolved_operations:
        if not isinstance(operation, dict):
            continue

        op_name = operation.get("op")
        sequence = operation.get("sequence")
        op_errors = []
        op_warnings = []
        commands = []
        variant_name = None
        target_device = _resolve_target_device_for_compile(state, operation)
        if not isinstance(op_name, str) or op_name not in cli_templates:
            op_errors.append("operation missing from template catalog")
            raw_args = {}
        elif not target_device:
            op_errors.append("target device not resolved")
            raw_args = dict(operation.get("bound_args") or {})
        elif not operation.get("ready_to_compile"):
            missing = operation.get("missing_required_args") or []
            op_errors.append(f"operation not ready to compile; missing required args: {missing}")
            raw_args = dict(operation.get("bound_args") or {})
        else:
            spec = cli_templates.get(op_name) or {}
            default_args = dict(spec.get("default_args") or {})
            bound_args = dict(operation.get("bound_args") or {})
            raw_args = {**default_args, **bound_args}
            compile_args = {
                key: _compile_arg_value(key, value)
                for key, value in raw_args.items()
            }

            templates, variant_name, selection_errors = _select_templates_for_operation(spec, raw_args)
            op_errors.extend(selection_errors)

            for template in templates:
                placeholders = _template_placeholders(template)
                missing_placeholders = [name for name in placeholders if name not in compile_args]
                if missing_placeholders:
                    op_errors.append(f"template placeholders not satisfied: {missing_placeholders}")
                    continue
                try:
                    commands.append({
                        "target_device": target_device,
                        "command": template.format(**compile_args),
                    })
                except Exception as exc:
                    op_errors.append(f"template render failed: {exc}")

        ready = len(op_errors) == 0 and bool(commands)
        if not commands and not op_errors:
            op_errors.append("no commands rendered")
            ready = False

        compiled_operation = {
            "op": op_name,
            "kind": operation.get("kind"),
            "role": operation.get("role"),
            "sequence": sequence,
            "variant": variant_name,
            "target_device": target_device,
            "commands": commands,
            "args": {
                key: _compile_arg_value(key, value)
                for key, value in raw_args.items()
            },
            "binding_sources": dict(operation.get("binding_sources") or {}),
            "ready_to_execute": ready,
            "errors": op_errors,
            "warnings": op_warnings,
        }
        compiled_operations.append(compiled_operation)

        for error in op_errors:
            compiler_errors.append(f"compiler:{op_name}:{sequence}: {error}")
        for warning in op_warnings:
            compiler_warnings.append(f"compiler:{op_name}:{sequence}: {warning}")

    needs_human = bool(compiler_errors)
    command_compilation = {
        "subintent_id": argument_resolution.get("subintent_id"),
        "compiled_operations": compiled_operations,
        "commands": [
            command
            for operation in compiled_operations
            for command in (operation.get("commands") or [])
        ],
        "needs_human": needs_human,
        "errors": compiler_errors,
        "warnings": compiler_warnings,
    }

    state["command_compilation"] = command_compilation
    if needs_human:
        state["needs_human"] = True
    if compiler_errors or compiler_warnings:
        state.setdefault("warnings", []).extend(compiler_errors + compiler_warnings)

    return state

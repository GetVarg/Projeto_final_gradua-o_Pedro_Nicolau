# pipeline_nodes.py
import re
import json
import ipaddress
from typing import Dict, List, Any, Tuple, Set, Optional

from main import IBNState, Entity, ExecPlan, Intent, Requirement
from tools import load_topologia, normalize_static_route_step, _safe_json_load, _ser, _retrieve_with_schedule, load_command_catalog

import os
from types import SimpleNamespace
from openai import OpenAI

# RAG separado
from rag_topology import (
    get_rag,
    query_from_state_for_rag,
    slice_topology,
    _edge_router_for_host,
    _first_ip_cidr_for_device,
    build_expanded_rag_query,
    get_cmd_rag,
    compact_command_catalog_for_discretize,
    load_cli_command_names
)

DEFAULT_OLLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


class HFRouterChat:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = HF_ROUTER_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key or os.environ["HF_TOKEN"]
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def bind(self, **kwargs):
        options = kwargs.get("options") or {}
        temperature = options.get("temperature", self.temperature)
        max_tokens = options.get("max_tokens", self.max_tokens)
        return HFRouterChat(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
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
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content or ""
        return SimpleNamespace(content=content)


llm = HFRouterChat(model=DEFAULT_OLLAMA_MODEL)


def set_ollama_model(model_name: str) -> None:
    global llm
    llm = HFRouterChat(model=model_name)


def get_ollama_model_name() -> str:
    try:
        return getattr(llm, "model", None) or DEFAULT_OLLAMA_MODEL
    except Exception:
        return DEFAULT_OLLAMA_MODEL

def _clip_text(value: Any, limit: int = 600) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <{len(text) - limit} chars more>"


def _compact_log_payload(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return "<max_depth>"
    if isinstance(value, dict):
        return {str(k): _compact_log_payload(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 5:
            trimmed = [_compact_log_payload(v, depth + 1) for v in value[:5]]
            trimmed.append(f"... <{len(value) - 5} more items>")
            return trimmed
        return [_compact_log_payload(v, depth + 1) for v in value]
    return _clip_text(value)


DEBUG_LLM_IO = False

def log_llm_exchange(tag: str, phase: str, payload: Dict[str, Any]) -> None:
    if not DEBUG_LLM_IO:
        return
    print("\n" + "-" * 88)
    print(f"[LLM {phase}] {tag}")
    print("-" * 88)
    print(json.dumps(_compact_log_payload(payload), ensure_ascii=False, indent=2))


def debug_node_intent_enabled(state: Dict[str, Any]) -> bool:
    return bool(state.get("debug_node_intent", False))

# =========================================================
# Nó orchestrador
# =========================================================

def ensure_cmd_rag_built() -> None:
    rag = get_cmd_rag()
    if rag.list_ops():
        return
    catalog = load_command_catalog("fewshots/cli_templates.json")
    rag.build_from_profile(catalog)

def node_orchestrator_root_grouding(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    if isinstance(work.get("subintents"), list) and len(work["subintents"]) > 0 and "root_intent" in work:
        return state
    state.setdefault("controller", {})
    controller = state["controller"]

    controller.setdefault("rag_min_score_schedule", [0.50, 0.40, 0.30])
    
    tmp = node_entities(state)
    tmp = node_context(tmp)

    state["topology_full"] = tmp.get("topology_full") or state.get("topology_full") or {}
    state["entities"] = tmp.get("entities") or []
    state["entity_selectors"] = tmp.get("entity_selectors") or []
    state["slice_topology"] = tmp.get("slice_topology") or {}
    state["warnings"] = tmp.get("warnings") or state.get("warnings") or []
    state["needs_human"] = bool(tmp.get("needs_human", False))

    state["root_entities"] = [serialize_entity(e) for e in (tmp.get("entities") or [])]
    state["root_entity_selectors"] = tmp.get("entity_selectors") or []
    state["root_topology_slice"] = tmp.get("slice_topology") or {}
    state["root_context"] = tmp.get("context") or {}

    tmp2 = node_discretize_intent(tmp)


    # O node_discretize_intent já escreve em state["work"]
    # Então só garantimos que isso existe
    if "work" not in tmp2 or "subintents" not in tmp2["work"]:
        state["needs_human"] = True
        state.setdefault("warnings", [])
        state["warnings"].append("orchestrator: discretization failed")
        return state

    state["work"] = tmp2["work"]
    state["work"]["cursor"] = 0

    state.setdefault("controller", {})
    controller = state["controller"]
    controller.setdefault("min_intent_confidence", 0.70)
    controller.setdefault("max_refine_depth", 2)

    min_conf = float(controller["min_intent_confidence"])
    max_depth = int(controller["max_refine_depth"])

    # (B) catálogo fechado real (a partir do seu command_profile)
    ensure_cmd_rag_built()
    supported_ops = get_cmd_rag().list_ops()
    supported_set = set(supported_ops)

    def refine_subintent_text(sub_text: str) -> tuple[str, bool]:
        system = """
    You rewrite ONE network sub-intent only if a minimal rewrite is necessary to improve mapping to a closed command catalog.

    PRIMARY GOAL
    - Preserve the original meaning.
    - Preserve the original action granularity.
    - Preserve all concrete anchors exactly.
    - Do not make the text broader, more generic, or more abstract.

    REFINEMENT POLICY
    - If the sub-intent already contains:
    - a local target device/interface
    - a clear action
    - the main parameters or referenced target
    then return it unchanged.

    - Rewrite only to:
    - remove ambiguity in attachment of parameters
    - make the local action explicit
    - clarify which object receives the configuration

    DO NOT
    - split the action
    - introduce a new action
    - rewrite into endpoint-connectivity language
    - convert protocol configuration into generic networking goals
    - replace concrete configuration language with abstract intent language

    SUCCESS CONDITION
    A good refinement is one that stays as close as possible to the original text while making catalog mapping easier.

    Minimal rewrite only.
    If no rewrite needed, copy original text exactly.
    No commentary.
    Return JSON only.

    OUTPUT
    Return JSON only:
    {"refined":"...","changed":true|false,"reason":"..."}
    """.strip()

        payload = {
            "subintent": sub_text,
            "root_intent": state.get("user_intent_text"),
            "supported_ops": supported_ops,
            "topology_slice": state.get("slice_topology"),
        }

        llm_i = llm.bind(options={"temperature": 0})
        log_llm_exchange("orchestrator_refine_subintent", "IN", {
            "system": system,
            "user": payload,
        })

        try:
            resp = llm_i.invoke([
                ("system", system),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ])
            raw = (resp.content or "").strip()

            log_llm_exchange("orchestrator_refine_subintent", "OUT", {
                "response": raw,
            })

            try:
                data = _safe_json_load(raw) if raw else {}
            except Exception as e:
                print("\n[refine_subintent_text][JSON_PARSE_FAILED]")
                print(f"subintent: {sub_text}")
                print(f"error: {e}")
                print("[refine_subintent_text][RAW_RESPONSE]")
                print(raw)
                return sub_text, False

            refined = ""
            changed = False

            if isinstance(data, dict):
                refined = (
                    data.get("refined")
                    or data.get("subintent")
                    or data.get("text")
                    or ""
                )
                changed = bool(data.get("changed", False))
            elif isinstance(data, str):
                refined = data
                changed = False
            else:
                refined = ""
                changed = False

            refined = (refined or "").strip()

            if not refined:
                print("\n[refine_subintent_text][EMPTY_REFINED_FALLBACK]")
                print(f"subintent: {sub_text}")
                print("[refine_subintent_text][PARSED_DATA]")
                print(data)
                return sub_text, False

            original_norm = " ".join((sub_text or "").split()).strip()
            refined_norm = " ".join(refined.split()).strip()

            if not refined_norm:
                return sub_text, False

            # If the model says unchanged, trust that signal.
            if not changed:
                return sub_text, False

            # If text is effectively identical, treat as no progress.
            if refined_norm.lower() == original_norm.lower():
                return sub_text, False

            # Reject obvious over-expansion on route/protocol intents.
            original_norm_l = original_norm.lower()
            refined_norm_l = refined_norm.lower()

            if " between " in refined_norm_l and " between " not in original_norm_l:
                protected_markers = [
                    "static route",
                    "route ",
                    "next-hop",
                    "via ",
                    "administrative distance",
                    "arp",
                    "neighbor",
                    "forwarding",
                    "ospf",
                    "bgp",
                ]
                if any(marker in original_norm_l for marker in protected_markers):
                    print("\n[refine_subintent_text][REJECTED_OVEREXPANDED_REWRITE]")
                    print(f"subintent: {sub_text}")
                    print("[refine_subintent_text][CANDIDATE]")
                    print(refined)
                    return sub_text, False

            # Reject rewrites that become too much longer for no clear reason.
            # This is a lightweight structural guard, not a regex-based semantic hack.
            original_tokens = original_norm.split()
            refined_tokens = refined_norm.split()
            if len(original_tokens) > 0 and len(refined_tokens) > max(len(original_tokens) + 8, int(len(original_tokens) * 1.6)):
                print("\n[refine_subintent_text][REJECTED_LENGTH_EXPANSION]")
                print(f"subintent: {sub_text}")
                print("[refine_subintent_text][CANDIDATE]")
                print(refined)
                return sub_text, False

            return refined, True

        except Exception as e:
            print("\n[refine_subintent_text][LLM_CALL_FAILED]")
            print(f"subintent: {sub_text}")
            print(f"error: {e}")
            return sub_text, False

    subs = state["work"].get("subintents") or []
    enriched = []

    for idx, si in enumerate(subs):
        sid = si.get("id") if isinstance(si, dict) else None
        stext = si.get("text") if isinstance(si, dict) else str(si)
        sid = sid or "S?"
        original_text = (stext or "").strip()
        orig_anchors = si.get("anchors") if isinstance(si, dict) else []

        attempts = []
        depth = 0
        cur_text = original_text
        final = {
            "name": "undetermined",
            "confidence": 0.0,
            "category": "indeterminate",
            "rationale": "",
        }
        status = "PENDING"

        prev_name = None
        prev_conf = None
        prev_text = None

        while True:
            state.setdefault("work", {})
            state["work"]["cursor"] = idx
            state["work"]["subintent"] = {
                "id": sid,
                "text": cur_text,
                "anchors": orig_anchors,
            }

            state["intent"] = None
            state["needs_human"] = False
            state["warnings"] = list(state.get("warnings", []))
        
            state = node_intent(state)

            cls = None
            subs_now = (state.get("work") or {}).get("subintents") or []
            if 0 <= idx < len(subs_now) and isinstance(subs_now[idx], dict):
                cls = subs_now[idx].get("classification")

            name = (cls or {}).get("name", "undetermined")
            conf = float((cls or {}).get("confidence", 0.0) or 0.0)
            cat = (cls or {}).get("category", "indeterminate")
            rat = (cls or {}).get("rationale", "") or ""

            attempts.append({
                "depth": depth,
                "text": cur_text,
                "name": name,
                "confidence": conf,
                "category": cat,
                "rationale": rat,
            })

            if debug_node_intent_enabled(state):
                print("\n[orchestrator][PRE_OK_CHECK]")
                print(json.dumps({
                    "subintent_id": sid,
                    "supported_set": sorted(list(supported_set)),
                    "min_conf": min_conf,
                    "name": name,
                    "conf": conf,
                    "classification_from_work_subintents_idx": cls,
                }, ensure_ascii=False, indent=2))

            ok = (name in supported_set) and (conf >= min_conf)
            if ok:
                final = {
                    "name": name,
                    "confidence": conf,
                    "category": cat,
                    "rationale": rat,
                }
                status = "CLASSIFIED"
                break

            if depth >= max_depth:
                final = {
                    "name": name,
                    "confidence": conf,
                    "category": cat,
                    "rationale": rat,
                }
                status = "NEEDS_HUMAN"
                break

            current_final = {
                "name": name,
                "confidence": conf,
                "category": cat,
                "rationale": rat,
            }


            refined_text, changed = refine_subintent_text(cur_text)


            # no rewrite -> no progress
            if not changed:
                final = current_final
                status = "NEEDS_HUMAN"
                break

            # defensive: same text after normalization -> no progress
            if refined_text.strip() == cur_text.strip():
                final = current_final
                status = "NEEDS_HUMAN"
                break

            # avoid refine loops that keep circling around the same text
            if prev_text is not None and refined_text.strip() == prev_text.strip():
                final = current_final
                status = "NEEDS_HUMAN"
                break

            # if we are not changing the predicted op family and confidence is not improving,
            # refining again is unlikely to help
            if prev_name is not None and prev_conf is not None:
                if name == prev_name and conf <= prev_conf:
                    final = current_final
                    status = "NEEDS_HUMAN"
                    break

            prev_name = name
            prev_conf = conf
            prev_text = cur_text

            depth += 1
            cur_text = refined_text

        orig_notes = si.get("notes", {}) if isinstance(si, dict) else {}
        enriched.append({
            "id": sid,
            "original_text": original_text,
            "text": cur_text,
            "notes": orig_notes,
            "status": status,
            "depth": depth,
            "classification": final,
            "attempts": attempts,
            "anchors": orig_anchors or [],
        })
    state["work"]["subintents"] = enriched
    state["work"]["cursor"] = 0

    return state


# =========================================================
# Nó 0: Discretizar intent
# =========================================================

def _compact_topology_for_discretize(topo: dict) -> dict:
    """
    Build a small, stable, LLM-friendly topology summary for intent discretization.

    Goal:
    - give the discretizer concrete anchors
    - avoid passing the full topology slice
    - keep a predictable schema
    """
    topo = topo or {}
    devices = topo.get("devices") or {}
    networks = topo.get("networks") or {}

    compact_devices = []
    for dev_name, dev_meta in list(devices.items())[:40]:
        iface_names = list((dev_meta.get("interfaces") or {}).keys())[:12]

        compact_devices.append({
            "name": dev_name,
            "kind": dev_meta.get("type") or dev_meta.get("kind") or "",
            "interfaces": iface_names,
        })

    compact_networks = []
    for net_name, net_meta in list(networks.items())[:60]:
        compact_networks.append({
            "name": net_name,
            "kind": net_meta.get("type") or net_meta.get("kind") or "",
        })

    return {
        "devices": compact_devices,
        "networks": compact_networks,
    }

def repair_discretize_json(raw: str, root_text: str) -> dict:
    system = """
You repair the output of a network intent discretizer.

Rules:
- Preserve meaning.
- Prefer one sub-intent if uncertain.
- Do not invent devices, interfaces, IPs, or protocols.
- If the raw output is unusable, return one sub-intent equal to the original intent.

You are not allowed to explain the repair.
Return only valid JSON.
If any extra token is emitted, task fails.
Return strict JSON only in this schema:
{
  "subintents": [
    {"id":"S1","text":"...","notes":{}}
  ]
}
""".strip()

    user = {
        "original_intent": root_text,
        "raw_output": raw,
    }

    resp = llm.bind(options={"temperature": 0}).invoke([
        ("system", system),
        ("user", json.dumps(user, ensure_ascii=False)),
    ])
    return _safe_json_load((resp.content or "").strip())

def node_discretize_intent(state: IBNState) -> IBNState:
    root_text = state["user_intent_text"]

    ensure_cmd_rag_built()

    system = """
You decompose a network management intent into sub-intents only when needed.

You are given:
- intent: the full user intent
- topology_slice: available device, interface, network, or system names

GOAL
- Cover all meaningful requirements in the intent.
- Each sub-intent must express one semantic requirement.
- Split only when the requirements are genuinely independent.

SPLITTING RULES
1) Create separate sub-intents only for requirements that represent different network actions or different configuration objectives.
2) Keep together details that belong to the same configuration objective.
3) Do not create duplicate or overlapping sub-intents.
4) If the intent expresses only one requirement, return one sub-intent.
5) If one clause contains two independent actions, even on the same local device, split into two sub-intents.

GROUNDING VS ACTION RULE
- Do not create a separate sub-intent for information lookup, topology grounding, neighbor discovery, or peer identification when that lookup only serves to fill arguments of another configuration action.
- If the intent says things like "find the neighbor of interface X and configure a route through it", keep everything in ONE sub-intent centered on the route action.
- In such cases, neighbor/peer discovery is auxiliary grounding, not an independent executable action.
- The same applies to phrases like:
  - "find the neighbor of ..."
  - "identify the peer of ..."
  - "discover who is linked to ..."
  - "use the device connected to ..."
when they only specify the next-hop, peer, or target of a route or similar configuration.

GROUNDING-DEPENDENT CLAUSE RULE
- Some clauses do not express an independent network action.
- If a clause only helps determine a required argument of another action, treat it as dependent grounding, not as a separate sub-intent.
- Typical dependent-grounding clauses identify:
  - a next-hop
  - a peer
  - a connected device
  - an owning device
  - an interface-associated target
- A dependent-grounding clause must stay inside the same sub-intent as the action that consumes that information.

DECISION TEST
Before splitting, ask:
- Can the clause be executed as a meaningful network action by itself?
- Or does it only provide information required to complete another action?

If it only provides information required by another action, do NOT split.

LOCAL-ACTION ANCHOR RULE
- When an intent describes configuring something "via", "through", "using", or "attached to" a discovered neighbor/peer/device, the main action is the local configuration action, not the discovery clause itself.
- The sub-intent should be written around the main local action, while preserving the grounding dependency in the text or notes.

SAME-OPERATION PARAMETER RULE
- Do not split a sub-intent when one clause expresses the main operation and another clause only adds a required argument, qualifier, attribute, or modifier of that same operation.
- Keep together all information that belongs to the same configurable object.

Keep together, in the SAME sub-intent:
- route + next-hop source
- route + neighbor/peer lookup
- route + administrative distance
- route + floating/backup/primary qualifier
- route + default-route qualifier
- BGP peer-group creation + remote-as of that peer-group
- BGP peer-group + neighbor binding to that same peer-group
- BGP neighbor + MED/local-pref/weight/as-path-related modifier for that same neighbor
- BGP process + network advertisement for that same process
- interface + MTU/description/admin-state of that same interface

- If one clause only helps determine a required slot or operational qualifier of another clause, they belong to the same sub-intent.
- Do not split a primary configuration action from its own parameters.

ANTI-OVER-SPLITTING RULE
- Prefer ONE sub-intent by default.
- Split only when the intent contains multiple independent actions that could be executed separately without losing meaning.
- If one clause configures a protocol/object and another clause only adds an attribute, parameter, modifier, enablement, or attachment to that same protocol/object, keep them together.
- If one clause only helps resolve a required slot of another clause, keep them together.
- If uncertain between one sub-intent and two, choose one.
- Never split administrative distance, metric, backup/floating qualifiers, remote-as, peer-group binding, or neighbor attributes away from the main object they modify.
- If the second clause cannot stand alone as a distinct network action without losing the target object, keep it in the same sub-intent.

NOTES
- notes stores useful semantic metadata explicitly implied by the intent.
- Use notes only when helpful for later stages.
- Examples: address_family, enabled, protocol, metric, attribute_type, run_condition, administrative_distance, route_kind, grounding_hint, source_interface.
- If there is no useful metadata, use {}.

CLOSED WORLD
- Prefer names that appear in topology_slice or in the intent itself.
- Do not invent missing names.

OUTPUT
Return strict JSON only in this format:

{
  "subintents": [
    {
      "id": "S1",
      "text": "...",
      "notes": {}
    }
  ]
}

CRITICAL OUTPUT CONTRACT:
Return EXACTLY one JSON object.
Start with {
End with }
Only top-level key allowed: subintents
No prose before or after.
No markdown.
No explanations.
Stop immediately after }
""".strip()

    topology_for_discretize = _compact_topology_for_discretize(
        state.get("slice_topology") or state.get("root_topology_slice") or {}
    )

    user_payload = {
        "intent": root_text,
        "topology_slice": topology_for_discretize,
        "catalog": compact_command_catalog_for_discretize(),
    }

    llm_i = llm.bind(options={"temperature": 0})
    log_llm_exchange("discretize_intent", "IN", {
        "system": system,
        "user": user_payload,
    })
    resp = llm_i.invoke([
        ("system", system),
        ("user", json.dumps(user_payload, ensure_ascii=False)),
    ])

    log_llm_exchange("discretize_intent", "OUT", {
        "response": resp.content,
    })
    raw = (resp.content or "").strip()

    try:
        data = _safe_json_load(raw) if raw else {}
    except Exception as e:
        state.setdefault("warnings", [])
        state["warnings"].append(f"discretize_parse_failed: {e}")

        try:
            data = repair_discretize_json(raw, root_text)
            state["warnings"].append("discretize_repaired_with_llm")
        except Exception as repair_e:
            state["warnings"].append(f"discretize_repair_failed: {repair_e}")
            data = {}

    if not isinstance(data, dict):
        data = {}

    subintents = data.get("subintents", []) or []
    if not isinstance(subintents, list):
        subintents = []

    def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    def _meaningful_tokens(text: str) -> set[str]:
        tokens = set()
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", (text or "").lower()):
            if len(tok) < 4:
                continue
            if re.fullmatch(r"\d+(?:-\w+)?", tok):
                continue
            if re.fullmatch(r"[a-z]+\d+(?:-\w+)?", tok):
                continue
            tokens.add(tok)
        return tokens

    def _has_anchor_like_signal(text: str) -> bool:
        t = (text or "").lower()
        return any(marker in t for marker in [
            "/", "::", "-", "bgp", "ospf", "icmp", "mtu", "forwarding",
            "neighbor", "route", "arp", "interface", "traceroute", "ping"
        ])

    root_tokens = _meaningful_tokens(root_text)

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

    # Conservative lexical filter: only apply when there are 3+ subintents.
    # Goal: remove obvious garbage without killing short but valid clauses.
    if len(subintents) > 2 and root_tokens:
        filtered = []
        discarded = []

        for si in subintents:
            text = si.get("text", "") if isinstance(si, dict) else ""
            overlap = _meaningful_tokens(text) & root_tokens

            keep = (
                len(overlap) >= 1
                or _has_anchor_like_signal(text)
            )

            if keep:
                filtered.append(si)
            else:
                discarded.append(text)

        if filtered:
            if discarded:
                state.setdefault("warnings", [])
                state["warnings"].append(
                    f"discretize_filtered_unrelated: discarded {len(discarded)} low-signal subintents"
                )
            subintents = filtered

    if not subintents:
        state.setdefault("warnings", [])
        state["warnings"].append("discretization_degraded: fallback_to_single_subintent")
        subintents = [{"id": "S1", "text": root_text, "notes": {}}]

    state["work"] = {
        "root_intent": root_text,
        "subintents": [
            {
                "id": si.get("id", f"S{i+1}"),
                "text": si.get("text", ""),
                "notes": si.get("notes", {}) or {},
                "status": "PENDING",
                "depth": 0,
            }
            for i, si in enumerate(subintents)
        ],
        "cursor": 0,
        "discretize_debug": {
            "temperature": 0,
            "raw": raw,
        }
    }
    return state


def node_pick_subintent(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    subs = work.get("subintents") or []
    cursor = int(work.get("cursor", 0))

    if cursor >= len(subs):
        state["active_subintent_id"] = None
        state["active_subintent_text"] = None
        return state

    si = subs[cursor] if isinstance(subs[cursor], dict) else {}
    sid = si.get("id") or f"S{cursor + 1}"
    text = (si.get("text") or "").strip()

    # 1) write canonical active fields
    state["active_subintent_id"] = sid
    state["active_subintent_text"] = text

    # 2) snapshot work.subintent for downstream/debug
    work["subintent"] = {
        "id": sid,
        "text": text,
        "notes": si.get("notes", {}) or {},
        "classification": si.get("classification"),
        "status": si.get("status"),
        "depth": si.get("depth"),
        "attempts": si.get("attempts"),
        "original_text": si.get("original_text"),
    }

    state["work"] = work
    return state

def node_store_subintent_result(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    results = work.get("results") or {}

    sid = state.get("active_subintent_id")
    if not sid:
        work["results"] = results
        state["work"] = work
        return state

    plan_obj = state.get("plan")
    if hasattr(plan_obj, "model_dump"):
        plan_dump = plan_obj.model_dump()
    elif hasattr(plan_obj, "dict"):
        plan_dump = plan_obj.dict()
    else:
        plan_dump = plan_obj
    classification = get_active_classification(state) or state.get("intent") or {}

    if hasattr(classification, "model_dump"):
        classification_dump = classification.model_dump()
    elif hasattr(classification, "dict"):
        classification_dump = classification.dict()
    else:
        classification_dump = classification

    plan_items_debug = work.get("plan_items_debug") or {}
    discretize_debug = work.get("discretize_debug") or {}

    results[sid] = {
        "subintent_text": state.get("active_subintent_text"),
        "needs_human": bool(state.get("needs_human", False)),
        "warnings": list(state.get("warnings", [])) if isinstance(state.get("warnings"), list) else [],
        "intent": classification_dump,
        "entities": [
            (e.model_dump() if hasattr(e, "model_dump") else (e.dict() if hasattr(e, "dict") else e))
            for e in (state.get("entities") or [])
        ],
        "entity_selectors": state.get("entity_selectors") or [],
        "requirements": state.get("requirements") or [],
        "plan_items": state.get("plan_items") or [],
        "plan_steps": state.get("plan_steps") or [],
        "plan": plan_dump,
        "cli_commands": state.get("cli_commands"),
        "verification": state.get("verification"),

        # debug payloads persisted into per-subintent results
        "plan_items_debug": {
            "raw": plan_items_debug.get("raw"),
            "parsed_keys": plan_items_debug.get("parsed_keys") or [],
            "active_subintent_id": plan_items_debug.get("active_subintent_id"),
            "op_name": plan_items_debug.get("op_name"),
            "subintents_produced": plan_items_debug.get("subintents_produced"),
            "classification_before_planning": plan_items_debug.get("classification_before_planning"),
            "mono_op_before_planning": plan_items_debug.get("mono_op_before_planning"),
        },
        "discretize_debug": {
            "raw": discretize_debug.get("raw"),
            "temperature": discretize_debug.get("temperature"),
        },
    }

    work["results"] = results
    state["work"] = work
    return state

def node_advance_cursor(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    before = int(work.get("cursor", 0))

    max_subintent_loops = int(state.get("max_subintent_loops", 5))
    if before >= max_subintent_loops:
        state["needs_human"] = True
        state.setdefault("warnings", []).append(
            f"advance_cursor_limit_reached: cursor={before}, limit={max_subintent_loops}"
        )
        state["work"] = work
        return state

    work["cursor"] = before + 1
    state["work"] = work
    return state

def router_has_more_subintents(state: IBNState) -> str:
    work = state.get("work") or {}
    subs = work.get("subintents") or []
    cursor = int(work.get("cursor", 0))

    max_subintent_loops = int(state.get("max_subintent_loops", 5))
    if cursor >= max_subintent_loops:
        state["needs_human"] = True
        state.setdefault("warnings", []).append(
            f"subintent_loop_limit_reached: cursor={cursor}, limit={max_subintent_loops}"
        )
        return "done"

    return "more" if cursor < len(subs) else "done"

def router_refined_or_continue(state: IBNState) -> str:
    if state.get("refined"):
        # limpa flag para não loopar errado
        state["refined"] = False
        return "refined"
    return "continue"

import json

from typing import Dict, Any

import json
from typing import Any, Dict, List

def print_plan_items_full(items: List[Dict[str, Any]], header: str = "PLAN_ITEMS_FULL") -> None:
    print(f"[{header}] count={len(items)}")
    if not items:
        print("  (empty)")
        return

    for i, it in enumerate(items):
        if not isinstance(it, dict):
            print(f"  [{i}] <invalid item: {type(it)}>")
            continue

        op = it.get("op")
        title = it.get("title")
        criterion = it.get("criterion")
        check = it.get("check")
        notes = it.get("notes") or []
        missing = it.get("missing") or []

        scope = it.get("scope") or {}
        endpoints = (scope.get("endpoints") or {})
        resolved = (scope.get("resolved") or {})
        params = (scope.get("params") or {})

        print(f"\n--- [{i}] op={op} ---")
        print(f"title     : {title}")
        print(f"criterion : {criterion}")
        print(f"endpoints : {endpoints}")
        print(f"resolved  : {resolved}")
        print(f"params    : {params}")
        print(f"notes     : {notes}")
        print(f"missing   : {missing}")
        print(f"check     : {check}")

        # Optional: raw JSON view (compact but complete)
        # print(json.dumps(it, ensure_ascii=False, indent=2))
    print()

from typing import Any, Dict

def _score_operation(sub_result: Dict[str, Any]) -> tuple[int, str]:
    intent = sub_result.get("intent") or {}
    op = None
    if isinstance(intent, dict):
        op = intent.get("name")

    plan_items = sub_result.get("plan_items") or []
    plan_steps = sub_result.get("plan_steps") or []

    if op in (None, "", "undetermined", "error"):
        return 0, "Operação não determinada."

    if plan_items:
        item_op = plan_items[0].get("op")
        if item_op == op:
            return 2, f"Operação consistente: classificação={op}, plan_item={item_op}."
        return 1, f"Família parcialmente consistente: classificação={op}, plan_item={item_op}."

    if plan_steps:
        step_op = plan_steps[0].get("op")
        if step_op == op:
            return 2, f"Operação consistente diretamente nos steps: {step_op}."
        return 1, f"Step gerado, mas com divergência em relação à classificação: {step_op}."

    return 0, "Nenhum plan_item ou plan_step foi gerado."


def _score_grounding(sub_result: Dict[str, Any]) -> tuple[int, str]:
    plan_items = sub_result.get("plan_items") or []
    plan_steps = sub_result.get("plan_steps") or []
    warnings = sub_result.get("warnings") or []

    missing = []
    grounded_signals = []

    for item in plan_items:
        if not isinstance(item, dict):
            continue
        for m in item.get("missing") or []:
            if m not in missing:
                missing.append(m)

        scope = item.get("scope") or {}
        resolved = scope.get("resolved") or {}
        endpoints = scope.get("endpoints") or {}

        if endpoints:
            grounded_signals.append(f"endpoints={endpoints}")
        if resolved:
            grounded_signals.append(f"resolved={resolved}")

    step_devices_missing = []
    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        if not step.get("device"):
            step_devices_missing.append(step)

    if missing or step_devices_missing:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if step_devices_missing:
            detail.append("há steps sem device")
        return 0, "; ".join(detail)

    grounding_warnings = [
        w for w in warnings
        if isinstance(w, str) and (
            "Unknown" in w or
            "corrected" in w or
            "cannot derive" in w or
            "cannot resolve" in w or
            "missing" in w
        )
    ]

    if grounding_warnings:
        return 1, "Há sinais de grounding parcial/corrigido: " + " | ".join(grounding_warnings[:3])

    if grounded_signals or plan_steps:
        return 2, "Argumentos resolvidos sem faltantes obrigatórios."

    return 0, "Sem evidência suficiente de grounding."

def node_debug_print(state: Dict[str, Any]) -> Dict[str, Any]:
    return state

def build_capability_profile(cmd_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Cria uma view compacta do catálogo de operações para reduzir prompt/deriva."""
    supported_ops = cmd_profile.get("supported_ops", {}) or {}
    ops = sorted(supported_ops.keys())

    requires_map: Dict[str, List[str]] = {}
    optional_map: Dict[str, List[str]] = {}
    targets_map: Dict[str, List[str]] = {}

    for op, meta in supported_ops.items():
        requires_map[op] = list(meta.get("requires", []) or [])
        optional_map[op] = list(meta.get("optional", []) or [])
        targets_map[op] = list(meta.get("targets", []) or [])

    return {
        "supported_ops": ops,
        "requires": requires_map,
        "optional": optional_map,
        "targets": targets_map,
    }


# =========================================================
# Nó 1: Intent classification
# =========================================================
def _extract_first_json_object(raw: str) -> dict:
    """
    Tenta extrair o primeiro objeto JSON válido de uma string.
    Funciona melhor que json.loads puro quando o modelo devolve
    texto extra antes/depois do objeto.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty LLM response.")

    decoder = json.JSONDecoder()

    # tentativa direta primeiro
    try:
        obj, end = decoder.raw_decode(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # procura o primeiro '{' e tenta raw_decode a partir dali
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    raise ValueError("Could not extract a valid top-level JSON object.")


def _coerce_intent_dict(parsed: dict) -> dict:
    """
    Normaliza a estrutura esperada por node_intent.
    Aceita pequenas variações de nome de campo sem depender de regex.
    """
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a dict.")

    op = parsed.get("op")
    if not op:
        op = parsed.get("operation") or parsed.get("name")

    confidence = parsed.get("confidence")
    if confidence is None:
        confidence = parsed.get("score")

    rationale = parsed.get("rationale")
    if not rationale:
        rationale = parsed.get("reason") or parsed.get("explanation") or ""

    return {
        "op": (op or "").strip(),
        "confidence": confidence if confidence is not None else 0.0,
        "rationale": rationale or "",
    }


def _safe_parse_intent_response(raw: str) -> dict:
    """
    Parser tolerante para a resposta do classificador.
    """
    parsed = _extract_first_json_object(raw)
    return _coerce_intent_dict(parsed)

def node_intent(state: IBNState) -> IBNState:
    """Classifica o intent em categoria macro e operação micro, com guardrail pós-LLM."""
    work = state.get("work") or {}
    sub = work.get("subintent") or {}
    text = sub.get("text") or state.get("user_intent_text") or ""
    debug_node_intent = debug_node_intent_enabled(state)

    text_l = (text or "").lower()

    ensure_cmd_rag_built()
    cmd_rag = get_cmd_rag()

    allowed_ops = cmd_rag.list_ops()
    allowed_ops.append("undetermined")

    # lightweight RAG hints (names + descriptions only)
    op_cards = cmd_rag.retrieve(text, min_score=0.18)
    hints = "\n".join([f"- {d['op']}: {d.get('description','')}" for d in op_cards[:8]])
    allowed_ops_txt = ", ".join(allowed_ops)
    hints_txt = hints if hints else "- (no hints)"
    system = """
You map ONE intent to ONE operation (closed-world).

Choose EXACTLY one string from Allowed ops.
Do not paraphrase.
Do not invent labels.
If uncertain return "undetermined".

Allowed ops (choose exactly one):
%s

Helpful catalog hints (may be incomplete):
%s

Decision rules:
1) Two endpoints + connectivity goal -> connectivity_ensure_pair.
2) "Secure" alone does NOT imply firewall. firewall_allow_pair only when the intent clearly describes allowlisting.
3) Redundancy is NOT a separate operation. If redundancy/failover/backup path is requested for a pair, still use connectivity_ensure_pair and mention redundancy in the rationale.
4) static_route_add requires: destination network + explicit next-hop.
5) qos_limit_iface requires: iface + numeric bandwidth.
6) firewall_drop_icmp_src requires: explicit source CIDR.
7) If QoS prioritization is requested without explicit class/DSCP/marking parameters, prefer qos_prioritize_pair (goal-level).

Return only:

{
  "op":"...",
  "confidence":0.0,
  "rationale":"..."
}

No additional keys.
No commentary.
No text outside JSON.
""".strip() % (allowed_ops_txt, hints_txt)
    user = f'Classify the following intent:\n"""{text}"""'

    try:
        log_llm_exchange("intent_classification", "IN", {
            "system": system,
            "user": user,
        })
        resp = llm.invoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        )

        log_llm_exchange("intent_classification", "OUT", {
            "response": resp.content,
        })

        if debug_node_intent:
            print("\n[node_intent][RAW_LLM_OUTPUT]")
            print(resp.content)

        intent_dict = _safe_parse_intent_response(resp.content)

        op = (intent_dict.get("op") or "").strip()

        try:
            conf = float(intent_dict.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        conf = max(0.0, min(1.0, conf))
        intent_dict["confidence"] = conf
        intent_dict["rationale"] = str(intent_dict.get("rationale") or "")

        # =========================
        # ENUM GUARDRAIL (robust classification)
        # =========================
        supported_ops = set(cmd_rag.list_ops())

        # If op still invalid -> force undetermined
        if op not in supported_ops:
            op = "undetermined"
            intent_dict["confidence"] = 0.0
            intent_dict["rationale"] = "Operation not in supported catalog."

        rationale_l = (intent_dict.get("rationale") or "").strip().lower()
        if rationale_l == "compound_subintent_requires_split":
            op = "undetermined"
            intent_dict["confidence"] = 0.0

        # Guardrail pós-LLM: forwarding sem destino => ip_forward_enable
        if op == "static_route_add":
            mentions_forwarding = any(k in text_l for k in [
                "forward packets", "forwarding", "between its interfaces", "between interfaces", "act as a router"
            ])
            mentions_dest = any(k in text_l for k in ["/", "subnet", "network", "via", "through", "next hop", "nexthop"])

            if mentions_forwarding and not mentions_dest:
                op = "ip_forward_enable"
                intent_dict["rationale"] = (intent_dict.get("rationale") or "").strip()
                if intent_dict["rationale"]:
                    intent_dict["rationale"] += " | "
                intent_dict["rationale"] += "Corrected: forwarding between interfaces without destination implies OS-level IP forwarding."

        try:
            conf = float(intent_dict.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        work = state.get("work") or {}
        subs = work.get("subintents") or []
        cursor = int(work.get("cursor", 0))

        cls = {
            "name": op if op else "undetermined",
            "category": "unknown",
            "confidence": float(conf or 0.0),
            "rationale": (intent_dict.get("rationale") or "")
        }

        active_sub = subs[cursor] if isinstance(subs, list) and 0 <= cursor < len(subs) and isinstance(subs[cursor], dict) else {}
        is_mono_op = cls["name"] not in ("undetermined", "error") and cls["rationale"] != "compound_subintent_requires_split"

        if isinstance(active_sub, dict):
            active_sub["classification"] = cls
            active_sub["status"] = "CLASSIFIED" if is_mono_op else "NEEDS_SPLIT"

        obs = work.get("classification_observability") or {}
        obs_key = active_sub.get("id") or f"S{cursor + 1}"
        obs[obs_key] = {
            "cursor": cursor,
            "subintent_id": active_sub.get("id") or f"S{cursor + 1}",
            "subintent_text": active_sub.get("text") or text,
            "subintents_produced": len(subs) if isinstance(subs, list) else 0,
            "classification": cls,
            "mono_op_before_planning": is_mono_op,
        }
        work["classification_observability"] = obs
        work["subintents"] = subs
        state["work"] = work

        if debug_node_intent and isinstance(subs, list) and 0 <= cursor < len(subs) and isinstance(subs[cursor], dict):
            print("\n[node_intent][CLS_TO_STORE]")
            print(json.dumps(cls, ensure_ascii=False, indent=2))
            print("[node_intent][STATE_WRITE_TARGET]")
            print(json.dumps(subs[cursor]["classification"], ensure_ascii=False, indent=2))
            print("[node_intent][STATE_WRITE_VERIFICATION]")
            print(json.dumps({
                "cursor": cursor,
                "stored_path": f"work.subintents[{cursor}].classification",
                "status": subs[cursor].get("status"),
                "same_object": subs[cursor].get("classification") == cls,
            }, ensure_ascii=False, indent=2))

    except Exception as e:


        # --- Persist error classification by cursor ---
        work = state.get("work") or {}
        subs = work.get("subintents") or []
        cursor = int(work.get("cursor", 0))

        cls = {
            "name": "error",
            "category": "unknown",
            "confidence": 0.0,
            "rationale": str(e)
        }

        if isinstance(subs, list) and 0 <= cursor < len(subs) and isinstance(subs[cursor], dict):
            subs[cursor]["classification"] = cls
            subs[cursor]["status"] = "NEEDS_HUMAN"

        work["subintents"] = subs
        state["work"] = work    

        if debug_node_intent and isinstance(subs, list) and 0 <= cursor < len(subs) and isinstance(subs[cursor], dict):
            print("\n[node_intent][ERROR_STATE_WRITE]")
            print(json.dumps({
                "error": str(e),
                "stored_path": f"work.subintents[{cursor}].classification",
                "stored_value": subs[cursor].get("classification"),
                "status": subs[cursor].get("status"),
            }, ensure_ascii=False, indent=2))

    return state


# =========================================================
# Nó 2: Extração de entidades + validação contra topologia
# =========================================================

import json
import ipaddress
from typing import Any, Dict, List, Set, Tuple, Optional

# =========================================================
# Helpers “puros” (fáceis de testar)
# =========================================================

ENTITY_KEYS = ["routers", "switches", "hosts", "interfaces", "ips", "cidrs", "services"]

def _empty_entities() -> Dict[str, List[str]]:
    return {k: [] for k in ENTITY_KEYS}

def _looks_like_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except Exception:
        return False

def _looks_like_cidr(s: str) -> bool:
    try:
        ipaddress.ip_network(s, strict=False)
        return True
    except Exception:
        return False

def _dedup_sort(ents: Dict[str, List[str]]) -> Dict[str, List[str]]:
    for k in ENTITY_KEYS:
        ents[k] = sorted(set(x for x in ents.get(k, []) if isinstance(x, str) and x.strip()))
    return ents

def _topology_indexes(topo: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Índices para validação rápida."""
    devices = topo.get("devices", {}) or {}
    dev_names = set(devices.keys())

    ifaces: Set[str] = set()
    for dev, meta in devices.items():
        for ifname in (meta.get("interfaces", {}) or {}).keys():
            ifaces.add(ifname)
            if isinstance(ifname, str) and ifname.startswith("eth"):
                ifaces.add(f"{dev}-{ifname}")

    return {"devices": dev_names, "interfaces": ifaces}

def _validate_against_topology(
    cands: Dict[str, List[str]],
    topo: Dict[str, Any]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    idx = _topology_indexes(topo)
    valid = _empty_entities()
    unknown = _empty_entities()

    # devices
    for k in ["routers", "switches", "hosts"]:
        for name in cands.get(k, []):
            if not idx["devices"] or name in idx["devices"]:
                valid[k].append(name)
            else:
                unknown[k].append(name)

    # interfaces
    for itf in cands.get("interfaces", []):
        if not idx["interfaces"] or itf in idx["interfaces"]:
            valid["interfaces"].append(itf)
        else:
            unknown["interfaces"].append(itf)

    # ips/cidrs/services não dá pra validar só por nomes
    for k in ["ips", "cidrs", "services"]:
        valid[k] = list(cands.get(k, []))

    return _dedup_sort(valid), _dedup_sort(unknown)

def _normalize_ip_cidr_buckets(ents: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    - Se algo em hosts parece IP/CIDR, move para ips/cidrs (mantém warning no caller)
    - Normaliza ips e cidrs (parseando com ipaddress)
    """
    moved = []
    keep_hosts = []

    for v in ents.get("hosts", []):
        if not isinstance(v, str):
            continue
        vv = v.strip()
        if _looks_like_cidr(vv):
            ents["cidrs"].append(vv)
            moved.append(vv)
        elif _looks_like_ip(vv):
            ents["ips"].append(vv)
            moved.append(vv)
        else:
            keep_hosts.append(vv)

    ents["hosts"] = keep_hosts

    # normaliza ips/cidrs
    raw = list(ents.get("ips", [])) + list(ents.get("cidrs", []))
    clean_ips: List[str] = []
    clean_cidrs: List[str] = []

    for addr in raw:
        if not isinstance(addr, str):
            continue
        a = addr.strip()
        if "/" in a:
            if _looks_like_cidr(a):
                clean_cidrs.append(str(ipaddress.ip_network(a, strict=False)))
        else:
            if _looks_like_ip(a):
                clean_ips.append(str(ipaddress.ip_address(a)))

    ents["ips"] = sorted(set(clean_ips))
    ents["cidrs"] = sorted(set(clean_cidrs))

    return ents, moved

def _filter_devices_literal(ents: Dict[str, List[str]], text: str) -> Dict[str, List[str]]:
    """Mantém routers/switches/hosts apenas se aparecerem literal no texto."""
    tl = (text or "").lower()
    for k in ["routers", "switches", "hosts"]:
        ents[k] = [v for v in ents.get(k, []) if isinstance(v, str) and v.lower() in tl]
    return ents

def _ensure_keys(ents: Dict[str, Any]) -> Dict[str, List[str]]:
    out = _empty_entities()
    if isinstance(ents, dict):
        for k in ENTITY_KEYS:
            v = ents.get(k, [])
            out[k] = list(v) if isinstance(v, list) else []
    return out

def _entities_llm_to_dict(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Aceita:
    - formato canônico: {"routers":[...], ...}
    - formato alternativo: [{"type":"router","name":"r1"}, ...]
    """
    ents = data.get("entities")

    # formato canônico
    if isinstance(ents, dict):
        return _ensure_keys(ents)

    # formato alternativo (lista flat)
    if isinstance(ents, list):
        buckets = _empty_entities()
        for item in ents:
            if not isinstance(item, dict):
                continue
            t = (item.get("type") or "").strip().lower()
            v = (item.get("value") or item.get("name") or "").strip()
            if not v:
                continue

            if t == "router":
                buckets["routers"].append(v)
            elif t == "switch":
                buckets["switches"].append(v)
            elif t == "host":
                buckets["hosts"].append(v)
            elif t == "interface":
                buckets["interfaces"].append(v)
            elif t == "ip":
                buckets["ips"].append(v)
            elif t in ("cidr", "subnet", "network"):
                buckets["cidrs"].append(v)
            elif t == "service":
                buckets["services"].append(v)
        return buckets

    return _empty_entities()

def _to_entity_list(ents: Dict[str, List[str]], source: str) -> List[Entity]:
    """
    Converte o dicionário de entidades normalizadas em uma lista tipada de Entity.
    """
    out: List[Entity] = []

    for r in ents.get("routers", []):
        out.append(Entity(type="router", value=r, meta={"source": source}))

    for s in ents.get("switches", []):
        out.append(Entity(type="switch", value=s, meta={"source": source}))

    for h in ents.get("hosts", []):
        out.append(Entity(type="host", value=h, meta={"source": source}))

    for itf in ents.get("interfaces", []):
        out.append(Entity(type="interface", value=itf, meta={"source": source}))

    for ip in ents.get("ips", []):
        out.append(Entity(type="ip", value=ip, meta={"source": source}))

    for cidr in ents.get("cidrs", []):
        # usa "subnet" como type canônico (compatível com seu planner)
        out.append(Entity(type="subnet", value=cidr, meta={"source": source}))

    for svc in ents.get("services", []):
        out.append(Entity(type="service", value=svc, meta={"source": source}))

    return out


# =========================================================
# LLM call (mantém sua lógica, mas isolada)
# =========================================================
def _extract_with_llm_entities_and_selectors(llm, text: str) -> Dict[str, Any]:
    entities_system = """
You are a Network Intent Parser. Map user natural language into technical entities.

IMPORTANT SEPARATION
Entities and selectors are independent outputs.
A decision to return "selectors": [] must NEVER reduce, suppress, or remove valid entities.
Even when no selector is appropriate, you must still extract all explicit entities normally.

CORE LOGIC

1) ENTITIES
Extract ONLY values explicitly mentioned in the text:
- device names (e.g., r2, s1, h11, 3035, 2, 4)
- interface names (e.g., r2-eth0, s1-eth3) ONLY if explicitly present
- Put a value in "ips" ONLY if the text contains a literal IPv4 or IPv6 address string.
- Put a value in "cidrs" ONLY if the text contains a literal CIDR string such as 10.1.0.0/24.
- Do NOT place descriptions, roles, or device references inside "ips" or "cidrs".- service names ONLY if explicitly present (dns, http, ssh, icmp, tcp, udp, ospf, bgp)

INVALID IP EXTRACTIONS
The following are NOT valid values for "ips":
- "IP address of host h_2_1"
- "host h_2_1"
- "router 3"
- "gateway of r1"
- "MAC address of host h_2_1"

Only literal addresses are valid in "ips", such as:
- "10.1.0.2"
- "172.16.2.10"
- "2001:db8::1"

CANONICAL DEVICE VALUE RULE
When the text says "router X", "host X", or "switch X", the entity value is only X.
Do NOT include the words "router", "host", or "switch" inside the entity value.

Examples:
- "configure router 3035" -> routers = ["3035"]
- "between router 2 and router 4" -> routers = ["2", "4"]
- "from router 1 to host h_2_1" -> routers = ["1"], hosts = ["h_2_1"]
- "on switch s1" -> switches = ["s1"]
- "use interface r2-eth0" -> interfaces = ["r2-eth0"]

HARD RULES
- NO GUESSING: Never invent device names, interfaces, IPs, CIDRs, or services.
- EXPLICIT ONLY: If a value is not literally in the text, do not output it.
- NO PLACEHOLDERS: Never output placeholder strings.
- If a device is written as "router X", "host X", or "switch X", return only X as the entity value.

SCHEMA
{
  "routers": [],
  "switches": [],
  "hosts": [],
  "interfaces": [],
  "ips": [],
  "cidrs": [],
  "services": []
}

Do not explain missing values.
If no literal IP appears in the text, return "ips": [].
If no literal CIDR appears in the text, return "cidrs": [].
Never put comments, explanations, or inferred text inside any schema field.

Respond ONLY with valid JSON.

Return EXACTLY this schema.
All values must be arrays of strings.
Never output null.
Never output comments.
Never output inferred values.
If no values exist, use [].
""".strip()

    selectors_system = """
You are a Network Intent Parser. Extract only selectors from the intent.

A selector is used only when the text refers to a group of devices.
If the text names specific devices, return no selector.

Create selectors only for collective targets such as:
- all routers
- every host
- each switch
- each device in the network
- all hosts in 10.1.0.0/24
- any router connected to r1

Do NOT create selectors for:
- a single named device
- a fixed list of devices
- configuration parameters

Examples with no selector:
- "configure router 3035"
- "between router 2 and router 4"
- "from host h_1_1 to router 3"

Output rules:
- Return valid JSON only.
- Use exactly this schema.
- kind must be "device_set"
- device_type must be one of: "device", "router", "host", "switch"
- quantifier must be one of: "all", "each", "any", "one"
- role must be one of: "sources", "targets", "next_hops"
- constraints must be a list
- Allowed constraint types: "connected_to", "ip_in_cidr", "device_name", "interface", "ip", "cidr"

Schema:
{
  "selectors": [
    {
      "kind": "device_set",
      "device_type": "device",
      "quantifier": "each",
      "constraints": [],
      "role": "targets"
    }
  ]
}

If there is no group selection, return:
{"selectors":[]}

Examples:
Input: "configure default gateways for each device in the network"
Output:
{
  "selectors": [
    {
      "kind": "device_set",
      "device_type": "device",
      "quantifier": "each",
      "constraints": [],
      "role": "targets"
    }
  ]
}

Input: "configure all routers connected to r1"
Output:
{
  "selectors": [
    {
      "kind": "device_set",
      "device_type": "router",
      "quantifier": "all",
      "constraints": [
        {"type":"connected_to","name":"r1"}
      ],
      "role": "targets"
    }
  ]
}

Input: "between router 2 and router 4"
Output:
{"selectors":[]}

Return EXACTLY this schema.
All values must be arrays of strings.
Never output null.
Never output comments.
Never output inferred values.
If no values exist, use [].
""".strip()

    user = f'IntentText:\\n"""{text}"""'

    log_llm_exchange("entities_extract", "IN", {
        "system": entities_system,
        "user": user,
    })
    entities_raw = llm.invoke(
        [("system", entities_system), ("user", user)]
    ).content
    log_llm_exchange("entities_extract", "OUT", {
        "response": entities_raw,
    })
    log_llm_exchange("entity_selectors_extract", "IN", {
        "system": selectors_system,
        "user": user,
    })
    selectors_raw = llm.invoke(
        [("system", selectors_system), ("user", user)]
    ).content
    log_llm_exchange("entity_selectors_extract", "OUT", {
        "response": selectors_raw,
    })

    try:
        entities_data = _safe_json_load(entities_raw) if entities_raw and entities_raw.strip() else {}
    except Exception as e:
        entities_data = {}
        log_llm_exchange("entities_extract_parse", "OUT", {
            "error": str(e),
            "raw": entities_raw,
        })

    try:
        selectors_data = _safe_json_load(selectors_raw) if selectors_raw and selectors_raw.strip() else {"selectors": []}
    except Exception as e:
        selectors_data = {"selectors": []}
        log_llm_exchange("entity_selectors_extract_parse", "OUT", {
            "error": str(e),
            "raw": selectors_raw,
        })

    if not isinstance(entities_data, dict):
        entities_data = {}

    if not isinstance(selectors_data, dict):
        selectors_data = {"selectors": []}

    return {
        "entities": {
            "routers": entities_data.get("routers", []),
            "switches": entities_data.get("switches", []),
            "hosts": entities_data.get("hosts", []),
            "interfaces": entities_data.get("interfaces", []),
            "ips": entities_data.get("ips", []),
            "cidrs": entities_data.get("cidrs", []),
            "services": entities_data.get("services", []),
        },
        "selectors": selectors_data.get("selectors", []),
    }

# =========================================================
# node_entities (agora fica curto e legível)
# =========================================================
def node_entities(state: "IBNState") -> "IBNState":
    text = state["user_intent_text"]
    topo = load_topologia()
    state["topology_full"] = topo
    data = _extract_with_llm_entities_and_selectors(llm, text)

    selectors = data.get("selectors") or []
    ents = _entities_llm_to_dict(data)          # garante dict sempre
    ents = _dedup_sort(ents)
    ents = _filter_devices_literal(ents, text)  # mesmo filtro que você tinha
    ents, moved = _normalize_ip_cidr_buckets(ents)
    ents = _dedup_sort(ents)


    # valida contra topologia
    valid_ents, unknown = _validate_against_topology(ents, topo)

    # warnings / needs_human
    if moved:
        state.setdefault("warnings", [])
        state["warnings"].append(f"Normalized entities: moved {moved} from hosts -> ips/cidrs")

    has_host_selector = any(
        (isinstance(s, dict) and s.get("kind") == "device_set" and (s.get("device_type") or "").lower() == "host")
        for s in selectors
    )

    unknown_msgs = []
    for k in ["routers", "switches", "hosts", "interfaces"]:
        if unknown.get(k):
            if k == "hosts" and has_host_selector:
                continue
            unknown_msgs.append(f"Unknown {k} referenced by intent: {', '.join(unknown[k])}")

    if unknown_msgs:
        state.setdefault("warnings", [])
        state["warnings"].extend(unknown_msgs)
        state["needs_human"] = True

    # converte para Entity list
    state["entities"] = _to_entity_list(valid_ents, source="LLM_ENTITIES")
    state["entity_selectors"] = selectors
    return state


# =========================================================
# Nó 3: Requirements
# =========================================================

def get_active_intent_classification(state: IBNState) -> dict:
    """
    Fonte de verdade:
    1) work.subintents[cursor].classification (pós-orchestrator)
    2) work.subintent.classification (se você armazenar o ativo ali)
    3) state.intent (fallback compatível)
    Retorna sempre um dict com: name, category, confidence, rationale
    """
    work = state.get("work") or {}

    # (1) via cursor + lista
    subs = work.get("subintents") or []
    cursor = work.get("cursor", 0)
    if isinstance(subs, list) and 0 <= int(cursor) < len(subs):
        si = subs[int(cursor)]
        if isinstance(si, dict):
            cls = si.get("classification")
            if isinstance(cls, dict):
                return {
                    "name": cls.get("name", "undetermined"),
                    "category": cls.get("category", "indeterminate"),
                    "confidence": float(cls.get("confidence", 0.0) or 0.0),
                    "rationale": cls.get("rationale", "") or "",
                    "status": si.get("status", "")
                }

    # (2) via work.subintent
    sub = work.get("subintent")
    if isinstance(sub, dict):
        cls = sub.get("classification")
        if isinstance(cls, dict):
            return {
                "name": cls.get("name", "undetermined"),
                "category": cls.get("category", "indeterminate"),
                "confidence": float(cls.get("confidence", 0.0) or 0.0),
                "rationale": cls.get("rationale", "") or "",
                "status": sub.get("status", "")
            }

    # (3) fallback state.intent
    intent_obj = state.get("intent")
    if intent_obj is not None:
        return {
            "name": getattr(intent_obj, "name", "undetermined"),
            "category": getattr(intent_obj, "category", "indeterminate"),
            "confidence": float(getattr(intent_obj, "confidence", 0.0) or 0.0),
            "rationale": getattr(intent_obj, "rationale", "") or "",
            "status": ""
        }

    return {"name": "undetermined", "category": "indeterminate", "confidence": 0.0, "rationale": "", "status": ""}


def node_requirements(state: IBNState) -> IBNState:
    ok = True
    reqs = []

    intent_cls = get_active_intent_classification(state)
    name = intent_cls["name"]
    category = intent_cls["category"]
    conf = intent_cls["confidence"]

    if intent_cls.get("status") == "NEEDS_HUMAN":
        state["needs_human"] = True
        state.setdefault("warnings", [])
        state["warnings"].append("requirements: active subintent status=NEEDS_HUMAN (skipping requirements)")
        return state

    if name == "undetermined":
        state["needs_human"] = True
        state.setdefault("warnings", [])
        state["warnings"].append("requirements: active intent undetermined")
        return state

    if category == "configuration" and name == "static_route_add":
        ok = validate_static_policy(state["topology_full"])
    reqs.append({"key": "static_policy_check", "ok": ok})
    state["requirements"] = reqs
    return state


def validate_static_policy(topo: Dict) -> bool:
    """Valida uma política simples: routers (exceto r0) precisam ter link para r0."""
    for name, dev in topo.get("devices", {}).items():
        if dev.get("type") == "router" and name != "r0":
            ifaces = dev.get("interfaces", {})
            if not any("r0" in v.get("peer", "") for v in ifaces.values()):
                return False
    return True


# =========================================================
# Nó 4: Anonimização (placeholder)
# =========================================================

def node_anonymize(state: IBNState) -> IBNState:
    """Nó placeholder: mantém o fluxo sem aplicar anonimização neste momento."""
    state["anonymization_map"] = {}
    return state


# =========================================================
# Nó 5: Context (RAG / slicing)
# =========================================================

def _rag_doc_debug(doc):
    return {
        "id": getattr(doc, "id", None),
        "score": getattr(doc, "score", None),
        "text": getattr(doc, "text", None),
        "meta": getattr(doc, "meta", None),
    }

def _count_topology_objects(topo: dict) -> dict:
    devices = topo.get("devices") or {}
    networks = topo.get("networks") or {}
    iface_count = 0
    for meta in devices.values():
        iface_count += len((meta.get("interfaces") or {}))
    return {
        "devices": len(devices),
        "networks": len(networks),
        "interfaces": iface_count,
    }


def _prune_topology_slice_for_llm(topo: dict, max_devices: int = 80, max_networks: int = 120) -> dict:
    devices = topo.get("devices") or {}
    networks = topo.get("networks") or {}

    if len(devices) <= max_devices and len(networks) <= max_networks:
        return topo

    keep_device_names = list(devices.keys())[:max_devices]
    pruned_devices = {name: devices[name] for name in keep_device_names}

    referenced_networks = {}
    for dev_meta in pruned_devices.values():
        for if_meta in (dev_meta.get("interfaces") or {}).values():
            cidr = (if_meta or {}).get("cidr")
            if cidr and cidr in networks:
                referenced_networks[cidr] = networks[cidr]

    # if still small enough, prefer referenced networks
    if len(referenced_networks) <= max_networks:
        pruned_networks = referenced_networks
    else:
        pruned_network_names = list(referenced_networks.keys())[:max_networks]
        pruned_networks = {name: referenced_networks[name] for name in pruned_network_names}

    return {
        "devices": pruned_devices,
        "networks": pruned_networks,
    }

def node_context(state: IBNState) -> IBNState:
    topo = state.get("topology_full") or {}
    if not topo:
        state.setdefault("warnings", [])
        state["warnings"].append("context: missing topology_full")
        state["needs_human"] = True
        return state

    rag = get_rag()
    rag.build(topo)

    controller = state.get("controller") or {}

    base_thresholds = controller.get("rag_base_score_schedule", [0.45, 0.35, 0.28])
    exp_thresholds = controller.get("rag_expanded_score_schedule", [0.60, 0.50, 0.40])

    q_base = query_from_state_for_rag(state)
    q_exp = build_expanded_rag_query(state, topo)

    base_docs, base_thr = _retrieve_with_schedule(rag, q_base, base_thresholds) if q_base else ([], None)

    if q_exp and q_base and q_exp.strip() == q_base.strip():
        exp_docs, exp_thr = [], None
    else:
        exp_docs, exp_thr = _retrieve_with_schedule(rag, q_exp, exp_thresholds) if q_exp else ([], None)

    merged = []
    seen = set()
    for d in (base_docs or []) + (exp_docs or []):
        key = getattr(d, "id", None) or repr(getattr(d, "meta", None)) or repr(d)
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)

    base_ids = {
        getattr(d, "id", None) or repr(getattr(d, "meta", None)) or repr(d)
        for d in (base_docs or [])
    }
    exp_only = []
    for d in (exp_docs or []):
        key = getattr(d, "id", None) or repr(getattr(d, "meta", None)) or repr(d)
        if key not in base_ids:
            exp_only.append(d)

    slice_topo = slice_topology(topo, merged)
    slice_topo_pruned = _prune_topology_slice_for_llm(slice_topo)

    state.setdefault("work", {})
    state["work"]["rag_debug"] = {
        "q_base": q_base,
        "q_exp": q_exp,
        "base_threshold_schedule": base_thresholds,
        "expanded_threshold_schedule": exp_thresholds,
        "base_threshold_used": base_thr,
        "expanded_threshold_used": exp_thr,
        "base_hits_count": len(base_docs or []),
        "expanded_hits_count": len(exp_docs or []),
        "expanded_only_hits_count": len(exp_only),
        "merged_hits_count": len(merged),
        "slice_counts_before_prune": _count_topology_objects(slice_topo),
        "slice_counts_after_prune": _count_topology_objects(slice_topo_pruned),
        "base_hits": [_rag_doc_debug(d) for d in (base_docs or [])],
        "expanded_hits": [_rag_doc_debug(d) for d in (exp_docs or [])],
        "expanded_only_hits": [_rag_doc_debug(d) for d in exp_only],
        "merged_hits": [_rag_doc_debug(d) for d in merged],
    }

    state["slice_topology"] = slice_topo_pruned
    state.setdefault("work", {})
    state["work"]["rag_debug"] = {
        "q_base": q_base,
        "q_exp": q_exp,
        "base_threshold_schedule": base_thresholds,
        "expanded_threshold_schedule": exp_thresholds,
        "base_threshold_used": base_thr,
        "expanded_threshold_used": exp_thr,
        "base_hits_count": len(base_docs or []),
        "expanded_hits_count": len(exp_docs or []),
        "expanded_only_hits_count": len(exp_only),
        "merged_hits_count": len(merged),
        "slice_device_count": len((slice_topo.get("devices") or {})),
        "slice_network_count": len((slice_topo.get("networks") or {})),
        "base_hits": [_rag_doc_debug(d) for d in (base_docs or [])],
        "expanded_hits": [_rag_doc_debug(d) for d in (exp_docs or [])],
        "expanded_only_hits": [_rag_doc_debug(d) for d in exp_only],
        "merged_hits": [_rag_doc_debug(d) for d in merged],
    }

    if not merged:
        state.setdefault("warnings", [])
        state["warnings"].append("RAG returned no hits; topology slice is empty.")
        state["needs_human"] = True
        return state

    counts_after = _count_topology_objects(slice_topo_pruned)

    state.setdefault("warnings", [])
    state["warnings"].append(
        f"RAG base_hits={len(base_docs)} thr={base_thr} | expanded_hits={len(exp_docs)} thr={exp_thr} | expanded_only={len(exp_only)} | merged_hits={len(merged)} | slice_devices={counts_after['devices']} | slice_networks={counts_after['networks']}"
    )
    return state

# =========================================================
# Nó 6: Planning (LLM)
# =========================================================
def serialize_entity(e):
    if hasattr(e, "model_dump"):
        return e.model_dump()
    if hasattr(e, "dict"):
        return e.dict()
    return {
        "type": getattr(e, "type", None),
        "value": getattr(e, "value", None),
        "meta": getattr(e, "meta", None),
    }

def _topo_compact_with_ips(topo: dict) -> dict:
    devices = topo.get("devices") or {}
    out_devices = {}
    for dev, meta in devices.items():
        ifaces = (meta.get("interfaces") or {})
        # pega ip/cidr/peer (bem pouco)
        out_ifaces = {}
        for ifn, im in ifaces.items():
            out_ifaces[ifn] = {
                "ip": (im or {}).get("ip"),
                "cidr": (im or {}).get("cidr"),
                "peer": (im or {}).get("peer"),
                "mac_address": (im or {}).get("mac_address"),
            }
        out_devices[dev] = {"type": meta.get("type"), "interfaces": out_ifaces}
    return {"devices": out_devices, "networks": topo.get("networks") or {}}

GENERIC_CRITERIA = {
    "",
    "observable end-state in the network",
    "observable end-state",
    "end-state",
}

# Canonical titles MUST match the plan_items prompt exactly (op -> title)
CANON_TITLES = {
    "connectivity_ensure_pair": "Establish connectivity between endpoints",
    "connectivity_verify":      "Verify connectivity between endpoints",
    "firewall_allow_pair":      "Apply firewall allow rule for endpoints",
    "qos_prioritize_pair":      "Prioritize traffic between endpoints",
}

def _device_type(topo: dict, dev: str) -> str | None:
    if dev == "controller":
        return "controller"
    meta = ((topo or {}).get("devices") or {}).get(dev) or {}
    t = meta.get("type")
    return t if isinstance(t, str) else None

def _deterministic_plan_for_item(item: dict, topo: dict) -> tuple[list[dict], list[str], bool, set[str]]:
    """
    Returns (steps, warnings, needs_human, covered_notes)

    Deterministic planner:
    - If item.steps already exists, treat them as the base deterministic plan.
    - Only require src_host/dst_host for operations/notes that actually need endpoints.
    - Keep note handling conservative.
    """
    warnings: list[str] = []
    needs_human = False
    covered_notes: set[str] = set()

    DEBUG_DETERMINISTIC_PLAN = False

    def dbg_print(*args, **kwargs):
        if DEBUG_DETERMINISTIC_PLAN:
            print(*args, **kwargs)

    op = item.get("op")
    scope = item.get("scope") or {}
    endpoints = scope.get("endpoints") or {}
    resolved = scope.get("resolved") or {}
    notes = item.get("notes") or []
    item_steps = item.get("steps") or []

    src_host = endpoints.get("src_host")
    dst_host = endpoints.get("dst_host")

    dbg_print("\n[deterministic_plan] ------------------------------")
    dbg_print("[deterministic_plan][op]", op)
    dbg_print("[deterministic_plan][endpoints]", endpoints)
    dbg_print("[deterministic_plan][resolved]", resolved)
    dbg_print("[deterministic_plan][notes]", notes)
    dbg_print("[deterministic_plan][item_steps_count]", len(item_steps) if isinstance(item_steps, list) else "invalid")

    if not isinstance(op, str) or not op:
        warnings.append("deterministic: missing op in plan_item")
        dbg_print("[deterministic_plan][ERROR] missing op")
        return [], warnings, True, covered_notes

    steps: list[dict] = []

    def add_step(device: str, op_: str, args: dict):
        step = {"device": device, "op": op_, "args": args or {}}
        steps.append(step)
        dbg_print("[deterministic_plan][add_step]", step)

    # --------------------------------------------------
    # MODE A: item already contains concrete steps
    # --------------------------------------------------
    if isinstance(item_steps, list) and item_steps:
        dbg_print("[deterministic_plan] using item.steps as base plan")

        for s in item_steps:
            if not isinstance(s, dict):
                warnings.append(f"deterministic: skipping non-dict step: {type(s).__name__}")
                continue

            dev = s.get("device")
            sop = s.get("op") or op
            sargs = s.get("args") or {}

            if not isinstance(dev, str) or not dev.strip():
                warnings.append(f"deterministic: skipping step with invalid device: {dev}")
                continue

            if not isinstance(sargs, dict):
                sargs = {}

            normalized = {
                "device": dev,
                "op": sop,
                "args": dict(sargs),
            }
            steps.append(normalized)
            dbg_print("[deterministic_plan][base_step]", normalized)

    # --------------------------------------------------
    # MODE B: synthesize from endpoints (legacy behavior)
    # --------------------------------------------------
    else:
        dbg_print("[deterministic_plan] item.steps empty; falling back to endpoint-based synthesis")

        if not src_host or not dst_host:
            warnings.append("deterministic: missing op/src_host/dst_host in plan_item")
            dbg_print("[deterministic_plan][ERROR] missing src_host/dst_host for fallback mode")
            return [], warnings, True, covered_notes

        base_args = {"src_host": src_host, "dst_host": dst_host}

        if isinstance(resolved, dict):
            if resolved.get("src_ip"):
                base_args["src_ip"] = resolved.get("src_ip")
            if resolved.get("dst_ip"):
                base_args["dst_ip"] = resolved.get("dst_ip")
            if resolved.get("dst_cidr"):
                base_args["dst_cidr"] = resolved.get("dst_cidr")

        target_dev = src_host

        if "redundancy_required" in notes:
            a1 = dict(base_args)
            a1["strategy"] = "primary"
            a2 = dict(base_args)
            a2["strategy"] = "backup"
            add_step(target_dev, op, a1)
            add_step(target_dev, op, a2)
            covered_notes.add("redundancy_required")
        else:
            add_step(target_dev, op, base_args)

    # --------------------------------------------------
    # NOTE: priority_data
    # --------------------------------------------------
    if "priority_data" in notes:
        dbg_print("[deterministic_plan] applying note: priority_data")
        for s in steps:
            if s.get("op") == op:
                s.setdefault("args", {}).setdefault("constraints", {})
                s["args"]["constraints"]["priority_data"] = True
        covered_notes.add("priority_data")

    # --------------------------------------------------
    # NOTE: redundancy_required
    # If steps already exist and note was not covered above,
    # keep conservative behavior: mark as uncovered warning.
    # --------------------------------------------------
    if "redundancy_required" in notes and "redundancy_required" not in covered_notes:
        warnings.append("deterministic: redundancy_required note present but not expanded for precomputed item.steps")
        dbg_print("[deterministic_plan][WARN] redundancy_required present but base steps already provided")

    # --------------------------------------------------
    # NOTE: secure
    # Only makes sense when we have endpoints or can infer them.
    # --------------------------------------------------
    if "secure" in notes:
        dbg_print("[deterministic_plan] applying note: secure")

        effective_src_host = src_host
        effective_dst_host = dst_host

        src_ip = None
        dst_ip = None

        if isinstance(resolved, dict):
            src_ip = resolved.get("src_ip")
            dst_ip = resolved.get("dst_ip")

        if not src_ip and effective_src_host:
            src_ip, _ = _first_ip_cidr_for_device(topo, effective_src_host)
        if not dst_ip and effective_dst_host:
            dst_ip, _ = _first_ip_cidr_for_device(topo, effective_dst_host)

        if not effective_src_host or not effective_dst_host:
            warnings.append("deterministic: secure note ignored because src_host/dst_host are unavailable")
            dbg_print("[deterministic_plan][WARN] secure ignored: missing endpoints")
        elif not src_ip or not dst_ip:
            needs_human = True
            warnings.append(
                f"planner: secure requested but missing src_ip/dst_ip for {effective_src_host}->{effective_dst_host}"
            )
            dbg_print("[deterministic_plan][WARN] secure needs human: missing src/dst ip")
        else:
            r_src = _edge_router_for_host(topo, effective_src_host)
            r_dst = _edge_router_for_host(topo, effective_dst_host)

            if r_src:
                add_step(r_src, "firewall_allow_pair", {"src_ip": src_ip, "dst_ip": dst_ip})
            else:
                warnings.append(
                    f"planner: could not find edge router for {effective_src_host}; using {effective_src_host} for firewall_allow_pair"
                )
                add_step(effective_src_host, "firewall_allow_pair", {"src_ip": src_ip, "dst_ip": dst_ip})

            if r_dst and r_dst != r_src:
                add_step(r_dst, "firewall_allow_pair", {"src_ip": dst_ip, "dst_ip": src_ip})
            elif (not r_dst) and effective_dst_host != effective_src_host:
                warnings.append(
                    f"planner: could not find edge router for {effective_dst_host}; using {effective_dst_host} for firewall_allow_pair"
                )
                add_step(effective_dst_host, "firewall_allow_pair", {"src_ip": dst_ip, "dst_ip": src_ip})

            covered_notes.add("secure")

    dbg_print("[deterministic_plan][final_steps_count]", len(steps))
    dbg_print("[deterministic_plan][warnings]", warnings)
    dbg_print("[deterministic_plan][needs_human]", needs_human)
    dbg_print("[deterministic_plan][covered_notes]", covered_notes)
    dbg_print("[deterministic_plan] ------------------------------\n")

    return steps, warnings, needs_human, covered_notes

def _resolve_endpoints_from_entities(entities: list[dict]) -> tuple[str | None, str | None]:
    """Pick 2 hosts from extracted entities in order of appearance."""
    hosts = []
    for e in entities or []:
        if (e or {}).get("type") == "host":
            v = (e or {}).get("value")
            if v and v not in hosts:
                hosts.append(v)
    if len(hosts) >= 2:
        return hosts[0], hosts[1]
    if len(hosts) == 1:
        return hosts[0], None
    return None, None

def _default_criterion(op: str, src: str | None, dst: str | None) -> str:
    if op == "connectivity_ensure_pair":
        return f"Reachability between {src or 'src_host'} and {dst or 'dst_host'}"
    if op == "connectivity_verify":
        return f"Connectivity verified between {src or 'src_host'} and {dst or 'dst_host'}"
    if op == "firewall_allow_pair":
        return f"Traffic between {src or 'src_host'} and {dst or 'dst_host'} is allowed by policy"
    if op == "firewall_drop_icmp_src":
        return "ICMP from source CIDR is blocked while other traffic remains unaffected"
    if op == "qos_prioritize_pair":
        return f"Data traffic between {src or 'src_host'} and {dst or 'dst_host'} is prioritized over other traffic"
    return "Desired network behavior is satisfied"


def _deterministic_checks_for_items(plan_items: list[dict], topo: dict) -> tuple[list[dict], list[str], bool]:
    """
    Deterministically map plan_item.op -> verification steps.

    Mapping:
    - connectivity_ensure_pair -> connectivity_verify (ping-like) from src_host to dst_ip
    - firewall_allow_pair      -> connectivity_verify from src_host to dst_ip (basic "not blocked" check)
    - qos_prioritize_pair      -> no deterministic check for now (warning)
    """
    checks: list[dict] = []
    warnings: list[str] = []
    needs_human = False

    def add_check(device: str, op: str, args: dict):
        checks.append({"device": device, "op": op, "args": args})

    for it in plan_items or []:
        if not isinstance(it, dict):
            continue

        op = it.get("op")
        scope = it.get("scope") or {}
        endpoints = (scope.get("endpoints") or {})
        resolved = (scope.get("resolved") or {})

        src_host = endpoints.get("src_host")
        dst_host = endpoints.get("dst_host")

        # Prefer grounded dst_ip; fallback to reading from topology by dst_host
        dst_ip = None
        if isinstance(resolved, dict):
            dst_ip = resolved.get("dst_ip")

        if not dst_ip and dst_host:
            ip, _ = _first_ip_cidr_for_device(topo, dst_host)
            dst_ip = ip

        # ---------- Mapping ----------
        if op in ("connectivity_ensure_pair", "firewall_allow_pair"):
            if not src_host or not dst_ip:
                needs_human = True
                warnings.append(f"checkmap: cannot verify {op} (missing src_host or dst_ip)")
                continue

            # connectivity_verify requires only dst_ip in your catalog :contentReference[oaicite:3]{index=3}
            # device=src_host means "run the verification from this host"
            add_check(device=src_host, op="connectivity_verify", args={"dst_ip": dst_ip})

        elif op == "qos_prioritize_pair":
            # No deterministic QoS verification in Mininet yet
            warnings.append("checkmap: QoS verification not implemented (qos_prioritize_pair)")
            # keep needs_human False: it's just "not verified", not necessarily wrong

        else:
            # unknown/unmapped op -> don't guess checks
            warnings.append(f"checkmap: no deterministic verification mapping for op={op}")

    return checks, warnings, needs_human

def get_active_subintent(state: IBNState) -> dict | None:
    work = state.get("work") or {}

    # 1) preferir snapshot atual
    si = work.get("subintent")
    if isinstance(si, dict) and si.get("id") == state.get("active_subintent_id"):
        return si

    # 2) fallback: procurar no vetor
    sid = state.get("active_subintent_id")
    if not sid:
        return None
    for x in (work.get("subintents") or []):
        if isinstance(x, dict) and x.get("id") == sid:
            return x

    return None


def get_active_classification(state: IBNState) -> dict | None:
    si = get_active_subintent(state) or {}
    return si.get("classification") if si else None

def _normalize_step_device_field(step: dict) -> dict:
    if not isinstance(step, dict):
        return step

    device = step.get("device")

    if isinstance(device, dict):
        identifier = (
            device.get("identifier")
            or device.get("name")
            or device.get("id")
        )
        if isinstance(identifier, str) and identifier.strip():
            step["device"] = identifier.strip()
        else:
            step["device"] = None

    elif isinstance(device, str):
        step["device"] = device.strip()

    else:
        step["device"] = None

    return step

def _normalize_ip_forward_family(step: dict, notes: dict) -> dict:
    if not isinstance(step, dict):
        return step

    if step.get("op") != "ip_forward_enable":
        return step

    args = step.get("args")
    if not isinstance(args, dict):
        args = {}
        step["args"] = args

    af = notes.get("address_family")
    if af in ("ipv4", "ipv6"):
        args["address_family"] = af

    return step


def node_plan_items(state: IBNState) -> IBNState:
    subintent = state.get("active_subintent_text")

    cls = get_active_classification(state) or {}
    op_name = cls.get("name", "undetermined")

    ensure_cmd_rag_built()
    cmd_rag = get_cmd_rag()
    op_spec = cmd_rag.get_op_spec(op_name)
    cli_command_names = load_cli_command_names("fewshots/cli_templates.json")

    if op_name == "undetermined":
        state["needs_human"] = True
        state.setdefault("warnings", [])
        state["warnings"].append("plan_items: undetermined op")
        state["plan_items"] = []
        state["plan_steps"] = []
        return state

    topo = state.get("slice_topology") or state.get("root_topology_slice") or {}
    slice_topo = state.get("slice_topology")
    root_topo = state.get("root_topology_slice")

    if slice_topo is not None:
        topo = slice_topo
        topo_source = "slice_topology"
    elif root_topo is not None:
        topo = root_topo
        topo_source = "root_topology_slice"
    else:
        topo = {}
        topo_source = "empty"

    print({
        "devices": len((topo.get("devices") or {})),
        "networks": len((topo.get("networks") or {})),
    })

    topo_compact = _topo_compact_with_ips(topo)

    DEBUG_LOCAL_NODE_PRINTS = False

    def dbg_print(*args, **kwargs):
        if DEBUG_LOCAL_NODE_PRINTS:
            print(*args, **kwargs)

    entities = [serialize_entity(e) for e in (state.get("entities") or [])]
    active_si = get_active_subintent(state) or {}
    raw_subintent_notes = active_si.get("notes", {})
    notes = raw_subintent_notes if isinstance(raw_subintent_notes, dict) else {}

    dbg_print("[node_plan_items][topo_source]")
    dbg_print(topo_source)
    dbg_print("[node_plan_items][topo_counts]")
    dbg_print({
        "devices": len((topo.get("devices") or {})),
        "networks": len((topo.get("networks") or {})),
    })
    dbg_print("[node_plan_items][RAW_NOTES]")
    dbg_print(notes)

    BASE_PLAN_ITEMS_SYSTEM = """

FAILURE CONDITIONS:
- text outside JSON
- top-level key different from item
- missing item.steps
- step.device not string
- invented command name

You convert ONE network sub-intent into a grounded execution step sequence (STRICT JSON ONLY).

GOAL
Generate the concrete steps for the sub-intent.
Each step must represent one action to be applied on one device.

OUTPUT FORMAT
{
  "item": {
    "op": "...",
    "title": "...",
    "scope": {
      "endpoints": {},
      "resolved": {},
      "params": {}
    },
    "notes": [],
    "missing": [],
    "criterion": "...",
    "steps": [
      {
        "device": "...",
        "op": "...",
        "command": "...",
        "args": {}
      }
    ]
  }
}

INPUTS
- op_spec: selected operation contract from the closed catalog
- topology_slice: valid devices, interfaces, IPs, networks, and relations
- entities: extracted hints
- intent_text: sub-intent text
- notes: attention constraints
- cli_command_names: allowed CLI command names for FRRouting + Mininet guidance

CORE RULES
1) Always return exactly one top-level key: "item".
2) Put all executable steps inside item.steps.
3) Do NOT return top-level "steps".
4) Every step inside item.steps must contain:
   - device
   - op
   - command
   - args
5) step.device must be the exact grounded device identifier as a string.
6) step.op must remain semantically consistent with op_spec.op.
7) step.command must be chosen from cli_command_names.
8) Do NOT invent devices, interfaces, IPs, or networks not present in topology_slice or entities.
9) If a concrete device is explicitly named in the intent, use that exact device in step.device.
10) Never replace a concrete device identifier with only its generic type.
11) Do NOT invent command names outside cli_command_names.
12) args must contain only grounded arguments.
13) If required information is missing, do not guess. Put unresolved fields in item.missing.

DEVICE FIELD SEMANTICS
- step.device must be a string containing only the exact concrete device identifier.
- If the intent explicitly names one device, use that exact identifier in step.device.
- Do not include the device type in step.device.
- Do not return a device object.

NOTES
- Notes are attention constraints.
- They modify how the steps are planned.
- They are not operations.
- Notes may carry qualifiers only, never a second required operational action.
- Notes must never absorb another mandatory obligation from the sub-intent.
- If the text still implies a second independent mandatory action beyond op_spec, do not hide it inside notes.
- Include in notes_used only the notes that actually affected the output.

IMPORTANT
- command is only a command-name hint for later CLI rendering.
- Do NOT render the full CLI line yet.
- Do NOT use templates.
- Stay close to valid FRRouting/Mininet semantics.

GROUP EXPANSION
- If the intent targets multiple devices or interfaces, expand into explicit device-level steps.

Use op_spec as the slot contract for the step.

For each step:
- inspect op_spec to determine which arguments must be filled
- for each argument, determine what semantic role it represents in this operation
- find the corresponding object in topology_slice
- extract the concrete value required for that argument
- emit the final resolved args

Do not stop at object identification.
A step is complete only when every op_spec-required argument has been resolved into an executable value.

EXAMPLES OF COMMAND NAMES
- ip address
- ip route
- no shutdown
- shutdown
- ip forwarding
- no ip forwarding
- mac-address
- router ospf
- network
- ip ospf cost
- ip route add
- ovs-vsctl set-controller

STRICT MODE ENABLED.

You are producing machine-readable output consumed by a parser.

Parser rules:
- one JSON object only
- no prose
- no trailing text
- no explanations
- no markdown

If you violate format, the task fails."""

    STATIC_ROUTE_ADD_PROMPT = """
STATIC ROUTE ATOMICITY RULE
- When the intent describes a single static route, all required arguments for that route must appear in the same step.
- Do not split one static route across multiple steps.
- Invalid: one step with only the destination prefix and another step with only the next-hop.
- Valid: one step containing the grounded arguments for that single static route.
"""

    IP_FORWARD_PROMPT = """
IP_FORWARD FAMILY RULE
- For op ip_forward_enable, determine whether the forwarding context is IPv4 or IPv6.
- Use notes.address_family when available.
- If notes.address_family is "ipv4", every generated step for ip_forward_enable must include:
  args.address_family = "ipv4"
- If notes.address_family is "ipv6", every generated step for ip_forward_enable must include:
  args.address_family = "ipv6"
- Do not invent address_family if it is not supported by the input evidence.

DEVICE-TARGET RULE FOR IP_FORWARD_ENABLE
- ip_forward_enable always targets a Layer-3 device, not an interface.
- If the intent mentions a router/device, step.device must be that device.
- If the intent mentions an interface only as context, do not use that interface as the device target.
- Interfaces may provide contextual grounding, but forwarding is enabled on the owning device.
- If the owning device cannot be identified from the topology, add the missing field instead of guessing.
"""

    NEXT_HOP_NEIGHBOR_PROMPT = """
NEXT-HOP RESOLUTION RULE
When a static-route intent says things like:
- "via that neighbor's IP"
- "via the neighbor of interface X"
- "through the peer of interface X"

you must resolve the next-hop as follows:

1) Find the exact local interface mentioned in the intent or notes.
2) In topology_slice, read the peer of that interface.
3) The next-hop must be the IP address of that peer interface.
4) Do NOT use:
   - an IP from another interface of the remote device
   - a LAN-facing interface IP
   - a host IP behind the neighbor router
5) If the peer interface exists but its IP cannot be found, add that field to item.missing instead of guessing.

EXAMPLE
If the intent references interface 0-eth2:
- find peer(0-eth2) = 9-eth0
- if 9-eth0 has IP 10.0.2.2
- then next-hop = 10.0.2.2

This rule overrides any more "plausible" gateway choice from the destination LAN.
Use the directly connected peer interface IP only.
""".strip()
    
    IPV6_ROUTE_INTERFACE_PROMPT = """
IPV6 STATIC ROUTE VIA INTERFACE RULE

Use this rule only for IPv6 static route sub-intents.

If the sub-intent says things like:
- "add an IPv6 route to <prefix> through interface <iface>"
- "via interface <iface>"
- "through interface <iface>"

then:
1) The correct operation is static_route_add_ipv6_frr.
2) The route step must use:
   - args.dst_cidr = the IPv6 destination prefix
   - args.exit_interface = the exact interface named in the sub-intent
3) Do NOT require a next_hop_ip when the intent explicitly says to route through an interface.
4) Do NOT put exit_interface in notes only; it must appear in step.args.
5) Do NOT invent an IPv6 gateway if the intent specifies an interface-based route.

A valid route-through-interface output is a grounded route step, not a forwarding step.
""".strip()

    IPV6_FORWARDING_PROMPT = """
IPV6 FORWARDING RULE

If the sub-intent says to enable IPv6 forwarding, then:
1) The correct operation is ip_forward_enable.
2) step.args.address_family must be "ipv6".
3) The target device must be the router explicitly named in the sub-intent.
4) Do NOT reinterpret IPv6 forwarding as a route-add operation.
5) Do NOT merge IPv6 forwarding into notes of another route item when it is its own sub-intent.

This sub-intent must produce a forwarding step, not a route step.
""".strip()

    system = BASE_PLAN_ITEMS_SYSTEM

    op_name = (op_spec or {}).get("op")
    sub_l = (subintent or "").lower()


    if op_name == "static_route_add":
        system += "\n\n" + STATIC_ROUTE_ADD_PROMPT

        has_neighbor_word = "neighbor" in sub_l or "peer" in sub_l
        has_interface_hint = "interface" in sub_l or bool(notes.get("interface"))

        if has_neighbor_word and has_interface_hint:
            system += "\n\n" + NEXT_HOP_NEIGHBOR_PROMPT

    elif op_name == "static_route_add_ipv6_frr":
        system += "\n\n" + IPV6_ROUTE_INTERFACE_PROMPT

    elif op_name == "ip_forward_enable":
        system += "\n\n" + IP_FORWARD_PROMPT

        if "ipv6 forwarding" in sub_l:
            system += "\n\n" + IPV6_FORWARDING_PROMPT

    payload = {
        "op_spec": op_spec,
        "topology_slice": topo_compact,
        "entities": entities,
        "intent_text": subintent,
        "notes": notes,
        "cli_command_names": cli_command_names,
    }

    try:
        llm_i = llm.bind(options={"temperature": 0})
        log_llm_exchange("plan_items", "IN", {
            "system": system,
            "user": payload,
        })
        resp = llm_i.invoke([("system", system), ("user", json.dumps(payload, ensure_ascii=False))])
        
        log_llm_exchange("plan_items", "OUT", {
            "response": resp.content,
        })
        raw = (resp.content or "").strip()
        
        try:
            data = _safe_json_load(raw) if raw else {}
        except Exception as e:
            state.setdefault("warnings", [])
            state["warnings"].append(f"plan_items_parse_failed: {e}")
            data = {}
        if not isinstance(data, dict):
            data = {}

        work = state.get("work") or {}
        classification_obs = (work.get("classification_observability") or {}).get(state.get("active_subintent_id")) or {}
        work["plan_items_debug"] = {
            "raw": raw,
            "parsed_keys": sorted(list(data.keys())) if isinstance(data, dict) else [],
            "active_subintent_id": state.get("active_subintent_id"),
            "op_name": op_name,
            "subintents_produced": classification_obs.get("subintents_produced"),
            "classification_before_planning": classification_obs.get("classification"),
            "mono_op_before_planning": classification_obs.get("mono_op_before_planning"),
        }
        state["work"] = work

        item = data.get("item")

        if isinstance(item, dict) and item.get("op") in (None, "", "error", "undetermined"):
            state["plan_items"] = []
            state["plan_steps"] = []
            state["needs_human"] = True
            state.setdefault("warnings", []).append(
                f"plan_items_invalid_op: {item.get('op')}"
            )
            return state

        if isinstance(item, dict):
            item_steps = item.get("steps") or []
            if isinstance(item_steps, list):
                normalized_item_steps = []
                for s in item_steps:
                    if not isinstance(s, dict):
                        continue
                    normalized_item_steps.append(_normalize_step_device_field(s))
                item["steps"] = normalized_item_steps

        steps = data.get("steps") or []
        if isinstance(steps, list):
            normalized_steps = []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                s = _normalize_step_device_field(s)
                s = _normalize_ip_forward_family(s, notes)
                normalized_steps.append(s)
            steps = normalized_steps

        if isinstance(item, dict) and item:
            op = op_spec.get("op")

            if item.get("op") != op:
                state.setdefault("warnings", []).append(
                    f"plan_items_fixup: op corrected {item.get('op')} -> {op}"
                )
                item["op"] = op

            canon_title = CANON_TITLES.get(op)
            if canon_title and item.get("title") != canon_title:
                state.setdefault("warnings", []).append(
                    f"plan_items_fixup: title corrected -> {canon_title}"
                )
                item["title"] = canon_title

            scope = item.get("scope") if isinstance(item.get("scope"), dict) else {}
            item["scope"] = scope

            endpoints = scope.get("endpoints") if isinstance(scope.get("endpoints"), dict) else {}
            scope["endpoints"] = endpoints

            src_host, dst_host = _resolve_endpoints_from_entities(entities)
            endpoints.setdefault("src_host", src_host)
            endpoints.setdefault("dst_host", dst_host)

            resolved = scope.get("resolved") if isinstance(scope.get("resolved"), dict) else {}
            scope["resolved"] = resolved

            topo_full = state.get("topology_full") or {}
            src = endpoints.get("src_host")
            dst = endpoints.get("dst_host")

            src_ip, src_cidr = _first_ip_cidr_for_device(topo_full, src) if src else (None, None)
            dst_ip, dst_cidr = _first_ip_cidr_for_device(topo_full, dst) if dst else (None, None)

            if src_ip:
                if resolved.get("src_ip") not in (None, src_ip):
                    state.setdefault("warnings", []).append(
                        f"plan_items_fixup: resolved.src_ip corrected {resolved.get('src_ip')} -> {src_ip}"
                    )
                resolved["src_ip"] = src_ip

            if src_cidr:
                if resolved.get("src_cidr") not in (None, src_cidr):
                    state.setdefault("warnings", []).append(
                        f"plan_items_fixup: resolved.src_cidr corrected {resolved.get('src_cidr')} -> {src_cidr}"
                    )
                resolved["src_cidr"] = src_cidr

            if dst_ip:
                if resolved.get("dst_ip") not in (None, dst_ip):
                    state.setdefault("warnings", []).append(
                        f"plan_items_fixup: resolved.dst_ip corrected {resolved.get('dst_ip')} -> {dst_ip}"
                    )
                resolved["dst_ip"] = dst_ip

            if dst_cidr:
                if resolved.get("dst_cidr") not in (None, dst_cidr):
                    state.setdefault("warnings", []).append(
                        f"plan_items_fixup: resolved.dst_cidr corrected {resolved.get('dst_cidr')} -> {dst_cidr}"
                    )
                resolved["dst_cidr"] = dst_cidr

            ALLOWED_NOTES = {
                "secure",
                "redundancy_required",
                "priority_data",
                "no_disruption",
                "safest_change",
                "reversible",
                "validate_after",
            }

            raw_notes = item.get("notes")
            item_notes = raw_notes if isinstance(raw_notes, list) else []

            NORMALIZE_NOTE = {
                "redundancy": "redundancy_required",
                "redundant": "redundancy_required",
                "security": "secure",
                "priority": "priority_data",
                "prioritize": "priority_data",
                "priority_class": "priority_data",
            }

            clean = []
            for n in item_notes:
                if not isinstance(n, str):
                    continue
                key = n.strip()
                key = NORMALIZE_NOTE.get(key, key)
                if key in ALLOWED_NOTES and key not in clean:
                    clean.append(key)

            item["notes"] = clean

            crit = item.get("criterion")
            if not isinstance(crit, str):
                crit = ""
            if crit.strip().lower() in GENERIC_CRITERIA:
                item["criterion"] = _default_criterion(op, src, dst)

            missing = item.get("missing")
            if not isinstance(missing, list):
                missing = []

            required = op_spec.get("requires") or []

            missing = [m for m in missing if isinstance(m, str) and m in required]

            if "src_host" in required and not endpoints.get("src_host"):
                missing.append("src_host")
            if "dst_host" in required and not endpoints.get("dst_host"):
                missing.append("dst_host")

            item["missing"] = sorted(set(missing))

            steps_i, warn_i, needs_i, _covered = _deterministic_plan_for_item(
                item,
                state.get("topology_full") or topo
            )
            expanded_steps = []
            expand_warnings = []
            expand_needs_human = False

            for step in steps_i:
                if not isinstance(step, dict):
                    continue
                if step.get("op") != "connectivity_ensure_pair":
                    expanded_steps.append(step)
                    continue

                new_steps, w, nh = _expand_connectivity_ensure_step(
                    step,
                    state.get("topology_full") or topo
                )
                expand_warnings.extend(w)
                expand_needs_human = expand_needs_human or nh
                expanded_steps.extend(new_steps)

            item["steps"] = expanded_steps

            state.setdefault("warnings", []).extend(warn_i)
            state["warnings"].extend(expand_warnings)
            state["needs_human"] = bool(state.get("needs_human", False)) or bool(needs_i) or bool(expand_needs_human)

            state["plan_items"] = [item]
            state["plan_steps"] = expanded_steps
            work = state.get("work") or {}

            items_all = work.get("plan_items_all")
            if not isinstance(items_all, list):
                items_all = []
            items_all.append(item)
            work["plan_items_all"] = items_all

            steps_all = work.get("plan_steps_all")
            if not isinstance(steps_all, list):
                steps_all = []
            steps_all.extend(expanded_steps)
            work["plan_steps_all"] = steps_all

            state["work"] = work

        elif isinstance(steps, list) and steps:
            state["plan_items"] = []
            state["plan_steps"] = steps
            state.setdefault("warnings", []).append(
                "plan_items_schema_mismatch: model returned 'steps' instead of 'item'"
            )

            work = state.get("work") or {}
            steps_all = work.get("plan_steps_all")
            if not isinstance(steps_all, list):
                steps_all = []
            steps_all.extend(steps)
            work["plan_steps_all"] = steps_all
            state["work"] = work
        else:
            state["plan_items"] = []
            state["plan_steps"] = []
            state["needs_human"] = True
            state.setdefault("warnings", []).append(
                "plan_items_empty: model returned neither 'item' nor usable 'steps'"
            )
    except Exception as e:
        state["plan_items"] = []
        state["plan_steps"] = []
        state["needs_human"] = True
        state.setdefault("warnings", [])
        state["warnings"].append(f"plan_items_failed: {e}")

    return state

CIDR_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")


def _stable_item_key(item: dict) -> tuple:
    """Deduplicate plan_items deterministically."""
    scope = item.get("scope") or {}
    endpoints = scope.get("endpoints") or {}
    resolved = scope.get("resolved") or {}
    notes = item.get("notes") or []
    return (
        item.get("op"),
        item.get("title"),
        item.get("criterion"),
        endpoints.get("src_host"),
        endpoints.get("dst_host"),
        resolved.get("dst_ip"),
        resolved.get("dst_cidr"),
        tuple(sorted([n for n in notes if isinstance(n, str)])),
    )


def _step_key(step: dict) -> tuple:
    return (
        step.get("device"),
        step.get("op"),
        json.dumps(step.get("args") or {}, sort_keys=True),
    )


def _router_path_policy(src_router: str, dst_router: str, topo: dict) -> list[str] | None:
    devices = (topo.get("devices") or {})
    if src_router not in devices or dst_router not in devices:
        return None
    if src_router == dst_router:
        return [src_router]
    if "r0" not in devices:
        return None
    return [src_router, "r0", dst_router]


def _expand_connectivity_ensure_step(step: dict, topo: dict) -> tuple[list[dict], list[str], bool]:
    warnings: list[str] = []
    needs_human = False

    args = step.get("args") or {}
    if not isinstance(args, dict):
        return [], ["expand: step.args is not a dict"], True

    src_host = args.get("src_host")
    dst_host = args.get("dst_host")
    dst_cidr = args.get("dst_cidr")

    if not src_host or not dst_host or not dst_cidr:
        return [], ["expand: missing src_host/dst_host/dst_cidr for connectivity_ensure_pair"], True

    _, src_cidr = _first_ip_cidr_for_device(topo, src_host)
    if not src_cidr:
        return [], [f"expand: cannot derive src_cidr from topology for {src_host}"], True

    r_src = _edge_router_for_host(topo, src_host)
    r_dst = _edge_router_for_host(topo, dst_host)
    if not r_src or not r_dst:
        return [], [f"expand: cannot find edge routers (r_src={r_src}, r_dst={r_dst})"], True

    path = _router_path_policy(r_src, r_dst, topo)
    if not path:
        return [], [f"expand: no path found for {r_src}->{r_dst} under current policy"], True

    if len(path) == 1:
        warnings.append(f"expand: {src_host}->{dst_host} share edge router {r_src}; no static routes added")
        return [], warnings, False

    r0 = path[1]
    expanded: list[dict] = []

    def add_step(device: str, op: str, a: dict):
        expanded.append({"device": device, "op": op, "args": a})

    add_step(r_src, "ip_forward_enable", {})
    add_step(r0, "ip_forward_enable", {})
    add_step(r_dst, "ip_forward_enable", {})

    add_step(r_src, "static_route_add", {"target_network": dst_cidr, "exit_gateway": r0})
    add_step(r0, "static_route_add", {"target_network": dst_cidr, "exit_gateway": r_dst})

    add_step(r_dst, "static_route_add", {"target_network": src_cidr, "exit_gateway": r0})
    add_step(r0, "static_route_add", {"target_network": src_cidr, "exit_gateway": r_src})

    if args.get("strategy") == "backup":
        warnings.append("expand: backup strategy requested but r0-hub policy has no alternate path yet")

    return expanded, warnings, needs_human


def node_plan(state: IBNState) -> IBNState:
    work = state.get("work") or {}
    plan_steps = work.get("plan_steps_all") or []
    plan_items = work.get("plan_items_all") or []

    if not plan_steps:
        state["plan"] = ExecPlan(
            steps=[],
            warnings=["No plan_steps available; refusing to plan."],
            needs_human=True,
            dry_run=True,
        )
        return state

    topo = state.get("slice_topology") or state.get("root_topology_slice") or {}
    final_steps: list[dict] = []
    final_warnings: list[str] = []
    final_needs_human = bool(state.get("needs_human", False))

    # --------------------------
    # Add deterministic verification steps (criterion mapping)
    # --------------------------
    seen_items = set()
    uniq_items = []
    for it in plan_items:
        if not isinstance(it, dict):
            continue
        k = _stable_item_key(it)
        if k in seen_items:
            continue
        seen_items.add(k)
        uniq_items.append(it)

    check_steps, check_warnings, check_needs_human = _deterministic_checks_for_items(uniq_items, topo)
    final_steps = list(final_steps) + list(check_steps)
    final_warnings = list(final_warnings) + list(check_warnings)
    final_needs_human = bool(final_needs_human) or bool(check_needs_human)

    # --------------------------
    # Deduplicate final steps (base steps already expanded in node_plan_items)
    # --------------------------
    seen_steps = set()
    dedup_final: list[dict] = []
    for s in list(plan_steps) + list(final_steps):
        if not isinstance(s, dict):
            final_needs_human = True
            final_warnings.append("plan: encountered non-dict step")
            continue
        if not s.get("device") or not s.get("op"):
            final_needs_human = True
            final_warnings.append(f"plan: incomplete step discarded {s}")
            continue
        k = _step_key(s)
        if k in seen_steps:
            continue
        seen_steps.add(k)
        dedup_final.append(s)

    final_steps = dedup_final

    state["plan"] = ExecPlan(
        steps=final_steps,
        warnings=final_warnings,
        needs_human=final_needs_human,
        dry_run=True,
    )

    return state

# =========================================================
# Nó: Step contract (validação contra templates)
# =========================================================

def node_step_contract(state: IBNState) -> IBNState:
    """Valida se os steps do plan cumprem o contrato mínimo dos templates de CLI."""
    plan = state["plan"]
    cli_specs = state.get("cli_templates", {})
    steps = plan.steps if hasattr(plan, "steps") else (plan.get("steps", []) if isinstance(plan, dict) else [])

    errors = []
    for step in steps:
        op = step.get("op")
        args = step.get("args", {})

        spec = cli_specs.get(op)
        if not spec:
            errors.append({"step": step, "error": "unknown_operation"})
            continue

        required = set(spec.get("required_step_args", []))
        missing = required - set(args.keys())

        if missing:
            errors.append({"step": step, "error": "missing_required_args", "details": list(missing)})

    if errors:
        state["fallback"] = {
            "target": "node_plan",
            "reason": "plan_cli_contract_violation",
            "errors": errors
        }
        return state

    return state


# =========================================================
# Nó: CLI generation
# =========================================================
def _is_ip(s: str) -> bool:
    """Checa se string é IP."""
    try:
        ipaddress.ip_address(s)
        return True
    except Exception:
        return False


def resolve_host_ip(host: str, topo: dict) -> Optional[str]:
    """Return the primary IPv4 address of a host (e.g., h42 -> 10.4.0.11)."""
    if not isinstance(host, str) or not host:
        return None
    devices = (topo or {}).get("devices", {}) or {}
    intfs = (devices.get(host, {}) or {}).get("interfaces", {}) or {}
    # pick first valid IP
    for _, meta in intfs.items():
        ip = (meta or {}).get("ip")
        if isinstance(ip, str) and _is_ip(ip):
            return ip
    return None

from copy import deepcopy

def _normalize_host_token(x: Any) -> Optional[str]:
    if not isinstance(x, str) or not x.strip():
        return None
    # "h42-eth0" -> "h42"
    return x.split("-", 1)[0].strip()

def derive_step_args(step: dict, spec: dict, topo: dict) -> dict:
    step = deepcopy(step)
    args = step.get("args") or {}
    if not isinstance(args, dict):
        args = {}

    derivable = spec.get("derivable_step_args", []) or []
    if not derivable:
        step["args"] = args
        return step

    # --- normalize host-like fields first ---
    if "src_host" in args:
        args["src_host"] = _normalize_host_token(args.get("src_host")) or args.get("src_host")
    if "dst_host" in args:
        args["dst_host"] = _normalize_host_token(args.get("dst_host")) or args.get("dst_host")

    # --- derive dst_ip from dst_host ---
    if "dst_ip" in derivable and not args.get("dst_ip"):
        dst_host = args.get("dst_host")
        ip = resolve_host_ip(dst_host, topo) if isinstance(dst_host, str) else None
        if ip:
            args["dst_ip"] = ip

    # (optional) derive src_ip similarly
    if "src_ip" in derivable and not args.get("src_ip"):
        src_host = args.get("src_host")
        ip = resolve_host_ip(src_host, topo) if isinstance(src_host, str) else None
        if ip:
            args["src_ip"] = ip

    step["args"] = args
    return step

def node_generate_cli(state: IBNState) -> IBNState:
    """Traduz o plano abstrato em comandos CLI usando templates + topologia."""
    from copy import deepcopy
    from collections import defaultdict

    plan = state.get("plan")
    topology = state.get("topology_full")

    if not plan or plan.needs_human:
        return state

    # -----------------------------
    # Helpers locais do node_generate_cli
    # -----------------------------

    def _peer_device(peer: str) -> Optional[str]:
        """Extrai nome do device do peer (ex: r0-eth1 -> r0)."""
        if not isinstance(peer, str) or not peer:
            return None
        return peer.split("-", 1)[0]

    def resolve_next_hop_ip(device: str, gateway: str, topo: dict) -> Optional[str]:
        """Resolve IP do next-hop gateway diretamente conectado ao device."""
        devices = (topo or {}).get("devices", {})
        dev_intfs = (devices.get(device, {}) or {}).get("interfaces", {}) or {}

        peer_value = None
        for _, meta in dev_intfs.items():
            peer = (meta or {}).get("peer")
            if _peer_device(peer) == gateway:
                peer_value = peer
                break

        if not peer_value:
            return None

        gw_intfs = (devices.get(gateway, {}) or {}).get("interfaces", {}) or {}

        if isinstance(peer_value, str) and "-" in peer_value:
            ip = (gw_intfs.get(peer_value, {}) or {}).get("ip")
            return ip if isinstance(ip, str) and _is_ip(ip) else None

        for _, gw_meta in gw_intfs.items():
            if _peer_device((gw_meta or {}).get("peer")) == device:
                ip = (gw_meta or {}).get("ip")
                return ip if isinstance(ip, str) and _is_ip(ip) else None

        return None

    def normalize_static_route_step_local(step: dict, topo: dict) -> dict:
        """Garante exit_gateway_ip em static_route_add, sem mutar o step original."""
        step = deepcopy(step)
        args = step.get("args") or {}
        dev = step.get("device")
        gw = args.get("exit_gateway")

        if not dev or not gw:
            return step

        if args.get("exit_gateway_ip"):
            return step

        if isinstance(gw, str) and _is_ip(gw):
            args["exit_gateway_ip"] = gw
            step["args"] = args
            return step

        if isinstance(gw, str):
            nh_ip = resolve_next_hop_ip(dev, gw, topo)
            if nh_ip:
                args["exit_gateway_ip"] = nh_ip
                step["args"] = args

        return step

    def get_template_op_alias(op: str) -> str:
        """
        Mantém o op semântico no plano e só traduz localmente
        para o nome do template concreto.
        """
        alias_map = {
            "static_route_add": "static_route_add_frr",
        }
        return alias_map.get(op, op)

    def normalize_args_for_template(template_op: str, step: dict) -> dict:
        """
        Normaliza nomes de argumentos apenas no momento de renderizar o template.
        Conservador: mexe só no que já precisamos destravar.
        """
        step = deepcopy(step)
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        if template_op == "static_route_add_frr":
            # destino
            if "dst_cidr" not in args:
                args["dst_cidr"] = (
                    args.get("dst_cidr")
                    or args.get("target_network")
                    or args.get("destination_prefix")
                )

            # next hop
            if "next_hop_ip" not in args:
                args["next_hop_ip"] = (
                    args.get("next_hop_ip")
                    or args.get("exit_gateway_ip")
                    or args.get("next_hop")
                )

            # distância administrativa
            if "administrative_distance" not in args:
                distance = (
                    args.get("administrative_distance")
                    or args.get("distance")
                )
                if distance is not None:
                    args["administrative_distance"] = distance

        step["args"] = args
        return step

    # -----------------------------
    # Carrega templates
    # -----------------------------
    with open("fewshots/cli_templates.json", "r") as f:
        templates = json.load(f)
        state["cli_templates"] = templates

    # -----------------------------
    # Normaliza plan antes de gerar CLI
    # -----------------------------
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
    steps = plan_dict.get("steps", []) or []

    normalized_steps = []
    for s in steps:
        if not isinstance(s, dict):
            normalized_steps.append(s)
            continue

        s2 = deepcopy(s)

        if s2.get("op") == "static_route_add":
            s2 = normalize_static_route_step_local(s2, topology)

        dev = s2.get("device")
        if isinstance(dev, str) and "-" in dev:
            root, rest = dev.split("-", 1)
            s2["device"] = root
            s2.setdefault("args", {})
            if isinstance(s2["args"], dict):
                s2["args"].setdefault("device_iface", rest)

        op = s2.get("op")
        template_op = get_template_op_alias(op)

        spec = templates.get(template_op) if isinstance(templates, dict) else None
        if isinstance(spec, dict):
            s2 = normalize_args_for_template(template_op, s2)
            s2 = derive_step_args(s2, spec, topology)

        normalized_steps.append(s2)

    plan_dict["steps"] = normalized_steps

    for s in plan_dict.get("steps", []):
        if isinstance(s, dict) and s.get("op") == "static_route_add":
            args = s.get("args") or {}
            if "exit_gateway_ip" not in args and "next_hop_ip" not in args:
                state["cli_commands"] = {
                    "status": "CLI_FAILED",
                    "error": (
                        f"Unable to resolve next-hop IP for device={s.get('device')} "
                        f"exit_gateway={args.get('exit_gateway')}"
                    ),
                    "raw": json.dumps(plan_dict, indent=2)[:1000],
                }
                state["needs_human"] = True
                return state

    # -----------------------------
    # Deterministic CLI generation from templates (NO LLM)
    # -----------------------------
    def render_step_with_templates(step: dict, templates_spec: dict) -> tuple[list[str], str | None]:
        op = step.get("op")
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        template_op = get_template_op_alias(op)
        step_for_template = normalize_args_for_template(template_op, step)
        args = step_for_template.get("args") or {}

        spec = templates_spec.get(template_op)
        if not spec:
            return [], f"no template for op={op} (template_op={template_op})"

        required = spec.get("required_step_args", []) or []
        missing = [k for k in required if k not in args or args.get(k) is None]
        if missing:
            return [], f"missing required_step_args for op={op} (template_op={template_op}): {missing}"

        defaults = spec.get("default_args", {}) or {}
        merged = dict(defaults)
        merged.update(args)

        variants = spec.get("variants") or []
        if isinstance(variants, list) and variants:
            selected = None
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                match = variant.get("match") or {}
                ok = True
                for key, expected in match.items():
                    if merged.get(key) != expected:
                        ok = False
                        break
                if ok:
                    selected = variant
                    break

            if not selected:
                return [], f"no matching variant for op={op} (template_op={template_op}) with args={args}"

            out_cmds = []
            for tmpl in (selected.get("templates") or []):
                try:
                    out_cmds.append(tmpl.format(**merged))
                except KeyError as e:
                    return [], f"template placeholder missing for op={op} (template_op={template_op}): {e}"
                except Exception as e:
                    return [], f"template render error for op={op} (template_op={template_op}): {e}"
            return out_cmds, None

        out_cmds = []
        for tmpl in (spec.get("templates") or []):
            try:
                out_cmds.append(tmpl.format(**merged))
            except KeyError as e:
                return [], f"template placeholder missing for op={op} (template_op={template_op}): {e}"
            except Exception as e:
                return [], f"template render error for op={op} (template_op={template_op}): {e}"

        return out_cmds, None

    commands_json: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []
    devices_meta = (topology or {}).get("devices", {}) or {}

    for step in plan_dict.get("steps", []):
        if not isinstance(step, dict):
            continue

        dev = step.get("device")
        if not isinstance(dev, str) or not dev.strip():
            errors.append(f"step has missing/invalid device: {dev}")
            continue

        if step.get("op") == "static_route_add":
            meta = devices_meta.get(dev) if isinstance(devices_meta, dict) else None
            dev_type = (meta or {}).get("type")
            if dev_type != "router":
                errors.append(
                    f"device={dev} op=static_route_add: refusing to emit host route; device_type={dev_type}"
                )
                continue

        cmds, err = render_step_with_templates(step, templates)
        if err:
            errors.append(f"device={dev} op={step.get('op')}: {err}")
            continue

        commands_json[dev].extend(cmds)

    if errors:
        state["cli_commands"] = {
            "status": "CLI_FAILED",
            "error": "Template-based CLI generation failed.",
            "details": errors[:50],
            "raw": json.dumps(plan_dict, indent=2)[:1500],
        }
        state["needs_human"] = True
        return state

    state["cli_commands"] = {"status": "CLI_GENERATED", "commands": dict(commands_json)}
    return state

# =========================================================
# Nós finais
# =========================================================

def node_decide_exec(state: IBNState) -> IBNState:
    """Nó placeholder de decisão (mantém fluxo)."""
    return state


def node_execute(state: IBNState) -> IBNState:
    """Executa (dry-run) aplicando steps se não precisar de humano."""
    result = {"status": "DRY_RUN", "applied": [], "skipped": []}
    plan = state.get("plan", ExecPlan())
    steps = plan.steps if hasattr(plan, "steps") else plan.get("steps", [])
    needs_human = plan.needs_human if hasattr(plan, "needs_human") else plan.get("needs_human", False)

    for step in steps:
        if needs_human:
            result["skipped"].append(step)
        else:
            result["applied"].append(step)

    state["exec_result"] = result
    return state


def node_verify(state: IBNState) -> IBNState:
    """Nó placeholder de verificação (mantém fluxo)."""
    state["verification"] = {"status": "verified_by_simulation"}
    return state

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from rag_topology import get_cmd_rag

# --- FUNÇÃO QUE O GRAFO VAI USAR ---
def load_topologia(path=r"C:\Users\pedro\OneDrive\Área de Trabalho\llmToNetworkConfig\Ic-llmToNetworkConfig/ibn_langgraph\dataset\topologias_convertidas\gabriel\10\0.json") -> dict:
    """
    C:\\Users\\pedro\\OneDrive\\Área de Trabalho\\llmToNetworkConfig\\Ic-llmToNetworkConfig\\ibn_langgraph\\dataset\\topologias_convertidas\\topozoo\\Abilene.json
    Lê o arquivo JSON gerado pelo Mininet para fornecer contexto à LLM.
    """
    caminho = Path(path)
    if not caminho.exists():
        print(f"Aviso: Arquivo {path} não encontrado. Certifique-se de que a topologia foi exportada.")
        return {}

    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler a topologia: {e}")
        return {}
        
import ipaddress
from copy import deepcopy

def _is_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except Exception:
        return False

def _peer_device(peer: str) -> str | None:
    # aceita "r0-eth1" ou "r0"
    if not isinstance(peer, str) or not peer:
        return None
    return peer.split("-", 1)[0]

def resolve_next_hop_ip(device: str, gateway: str, topology: dict) -> str | None:
    devices = (topology or {}).get("devices", {})
    dev_intfs = (devices.get(device, {}) or {}).get("interfaces", {}) or {}

    # procura uma interface que tenha peer apontando para o gateway
    peer_intf_name = None
    for ifname, meta in dev_intfs.items():
        peer = (meta or {}).get("peer")
        if _peer_device(peer) == gateway:
            peer_intf_name = peer  # ex: "r0-eth1" ou "r0"
            break

    if not peer_intf_name:
        return None

    # se peer veio só como "r0", não temos nome da intf do outro lado
    # nesse caso, tentamos achar no gateway alguma interface cujo peerDevice == device
    gw_intfs = (devices.get(gateway, {}) or {}).get("interfaces", {}) or {}

    if "-" not in peer_intf_name:
        for gw_if, gw_meta in gw_intfs.items():
            if _peer_device((gw_meta or {}).get("peer")) == device:
                ip = (gw_meta or {}).get("ip")
                return ip if isinstance(ip, str) and _is_ip(ip) else None
        return None

    # caso normal: peer inclui interface do gateway
    ip = (gw_intfs.get(peer_intf_name, {}) or {}).get("ip")
    return ip if isinstance(ip, str) and _is_ip(ip) else None

def normalize_static_route_step(step: dict, topology: dict) -> dict:
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
        nh = resolve_next_hop_ip(dev, gw, topology)
        if nh:
            args["exit_gateway_ip"] = nh
            step["args"] = args

    return step

from typing import Any, Dict, List, Set

def collect_device_values(entities: List[Any], device_names: Set[str]) -> List[str]:
    out: List[str] = []
    for e in entities or []:
        if isinstance(e, dict):
            v = e.get("value")
        else:
            v = getattr(e, "value", None)

        if isinstance(v, str) and v in device_names:
            out.append(v)

    return out  


def _safe_json_load(raw: str) -> dict:
    """Parse JSON de forma robusta sem regex. Tenta JSON puro; se vier com lixo, recorta do primeiro { ao último }."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        i = raw.find("{")
        j = raw.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(raw[i:j+1])
        raise

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
    """Log compacto das trocas com a LLM, habilitado apenas por flag de debug."""
    if not DEBUG_LLM_IO:
        return
    print("\n" + "-" * 88)
    print(f"[LLM {phase}] {tag}")
    print("-" * 88)
    print(json.dumps(_compact_log_payload(payload), ensure_ascii=False, indent=2))


def _ser(e):
        if hasattr(e, "model_dump"):
            return e.model_dump()
        if hasattr(e, "dict"):
            return e.dict()
        return {"type": getattr(e, "type", None), "value": getattr(e, "value", None), "meta": getattr(e, "meta", None)}

def _normalize_traits(traits: Any) -> Dict[str, bool]:
    """
    Normalize subintent traits to a stable, closed schema.
    Always returns keys: redundancy/security/priority as booleans.
    """
    base = {"redundancy": False, "security": False, "priority": False}
    if not isinstance(traits, dict):
        return base

    for k in list(base.keys()):
        v = traits.get(k, False)
        base[k] = True if v is True else False
    return base

def _retrieve_with_schedule(rag, query: str, thresholds: list[float]) -> tuple[list, float | None]:
    docs, used_thr = rag.retrieve_adaptive(query, thresholds=thresholds)
    return docs, (None if not docs else float(used_thr))

def _required_arg_keys_for_op(op: str) -> set[str]:
    ensure_cmd_rag_built()
    spec = get_cmd_rag().get_op_spec(op)
    return set([k for k in (spec.get("requires") or []) if isinstance(k, str)])

def _targets_for_op(op: str) -> set[str]:
    ensure_cmd_rag_built()
    spec = get_cmd_rag().get_op_spec(op)
    return set([k for k in (spec.get("targets") or []) if isinstance(k, str)])

def _allowed_arg_keys_for_op(op: str) -> set[str]:
    ensure_cmd_rag_built()
    spec = get_cmd_rag().get_op_spec(op)
    req = spec.get("requires") or []
    opt = spec.get("optional") or []
    schema = spec.get("args_schema") or {}
    keys = set([k for k in req + opt if isinstance(k, str)])
    if isinstance(schema, dict):
        keys |= set(schema.keys())
    return keys

from pathlib import Path
import json
from typing import Any, Dict

def load_command_catalog(path: str = "data/command_catalog.json") -> Dict[str, Any]:
    """
    Loads the command catalog (closed-world) used by the IBN pipeline.
    """
    p = Path(path)
    if not p.exists():
        print(f"Warning: Command catalog file {path} not found.")
        return {}

    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error while reading command catalog: {e}")
        return {}


def load_cli_templates(path: str = "fewshots/cli_templates.json") -> Dict[str, Any]:
    """Load the concrete CLI template catalog used for rendering commands."""
    p = Path(path)
    if not p.exists():
        print(f"Warning: CLI templates file {path} not found.")
        return {}

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Error while reading CLI templates: {e}")
        return {}


def compact_cli_template_catalog_for_planner(path: str = "fewshots/cli_templates.json") -> List[Dict[str, Any]]:
    """Return a compact template catalog view focused on planning contracts."""
    templates = load_cli_templates(path)
    out: List[Dict[str, Any]] = []

    for op_name, spec in templates.items():
        if not isinstance(spec, dict):
            continue
        out.append({
            "op": op_name,
            "context": spec.get("context", ""),
            "description": spec.get("description", ""),
            "required_step_args": list(spec.get("required_step_args") or []),
            "derivable_step_args": list(spec.get("derivable_step_args") or []),
            "default_args": dict(spec.get("default_args") or {}),
            "has_variants": bool(spec.get("variants")),
            "num_templates": len(spec.get("templates") or []),
        })

    return out


def verification_template_candidates_for_apply_op(apply_op: str) -> List[str]:
    """Return conservative verification template candidates for a given apply op."""
    mapping = {
        "static_route_add_frr": ["connectivity_verify"],
        "static_route_add_ipv6_frr": ["connectivity_verify"],
        "static_route_blackhole_add": [],
        "static_route_del": [],
    }
    return list(mapping.get(apply_op, []))


def ensure_cmd_rag_built() -> None:
    """Garante que o catalogo de comandos esteja carregado no CommandCatalogRAG."""
    rag = get_cmd_rag()
    if rag.list_ops():
        return
    catalog = load_command_catalog("fewshots/cli_templates.json")
    rag.build_from_profile(catalog)


def _normalize_arguments(raw_arguments: Any) -> dict:
    """Normaliza o bloco arguments para o schema topology/semantic do pipeline."""
    if not isinstance(raw_arguments, dict):
        raw_arguments = {}

    topology_arguments = raw_arguments.get("topology_arguments")
    semantic_only_arguments = raw_arguments.get("semantic_only_arguments")
    derived_arguments = raw_arguments.get("derived_arguments")

    if isinstance(topology_arguments, dict) or isinstance(semantic_only_arguments, dict) or isinstance(derived_arguments, dict):
        if not isinstance(topology_arguments, dict):
            topology_arguments = {}
        if not isinstance(semantic_only_arguments, dict):
            semantic_only_arguments = {}
        if not isinstance(derived_arguments, dict):
            derived_arguments = {}

        return {
            "topology_arguments": topology_arguments,
            "semantic_only_arguments": semantic_only_arguments,
            "derived_arguments": derived_arguments,
        }

    return {
        "topology_arguments": raw_arguments,
        "semantic_only_arguments": {},
        "derived_arguments": {},
    }


def _normalize_intent_frame(raw_frame: Any) -> dict:
    """Fecha um intent_frame com defaults seguros para o restante do grafo."""
    if not isinstance(raw_frame, dict):
        raw_frame = {}

    goal = raw_frame.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        goal = "unspecified_goal"

    arguments = _normalize_arguments(raw_frame.get("arguments"))

    notes = raw_frame.get("notes")
    if not isinstance(notes, dict):
        notes = {}

    return {
        "goal": goal.strip(),
        "arguments": arguments,
        "notes": notes,
    }


def _normalize_subintent_record(si: dict, fallback_id: str) -> dict:
    """Estabiliza um subintent com defaults antes da execucao do fluxo."""
    text = si.get("text") if isinstance(si, dict) else ""
    if not isinstance(text, str):
        text = ""

    frame = _normalize_intent_frame(si.get("intent_frame") if isinstance(si, dict) else {})

    notes = frame.get("notes") or {}
    if not isinstance(notes, dict):
        notes = {}

    return {
        "id": si.get("id", fallback_id) if isinstance(si, dict) else fallback_id,
        "text": text,
        "notes": notes,
        "intent_frame": frame,
        "status": "PENDING",
        "depth": 0,
    }


def _merge_unique_list(existing: list, incoming: list) -> list:
    """Merge two lists while preserving order and removing duplicates."""
    merged = list(existing) if isinstance(existing, list) else []
    for item in incoming or []:
        if item not in merged:
            merged.append(item)
    return merged


def _merge_prefer_existing(existing: dict, incoming: dict) -> dict:
    """Merge dictionaries but keep the first meaningful value already seen."""
    merged = dict(existing) if isinstance(existing, dict) else {}
    for key, value in (incoming or {}).items():
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _merge_context_records(global_context: dict, local_context: dict) -> dict:
    """Accumulate local subintent context into a global intent-level context."""
    global_context = global_context if isinstance(global_context, dict) else {}
    local_context = local_context if isinstance(local_context, dict) else {}

    global_slice = global_context.get("slice_topology") or {}
    local_slice = local_context.get("slice_topology") or {}

    global_exact_matches = global_context.get("exact_matches") or {}
    local_exact_matches = local_context.get("exact_matches") or {}

    global_unmatched = global_context.get("unmatched_topology_arguments") or []
    local_unmatched = local_context.get("unmatched_topology_arguments") or []

    merged_exact_matches = dict(global_exact_matches)
    for arg_name, result in local_exact_matches.items():
        if arg_name not in merged_exact_matches:
            merged_exact_matches[arg_name] = result

    merged_unmatched = []
    for item in list(global_unmatched) + list(local_unmatched):
        if item not in merged_unmatched:
            merged_unmatched.append(item)

    merged_matched_entities = []
    seen_entities = set()
    for item in list(global_context.get("matched_topology_entities") or []) + list(local_context.get("matched_topology_entities") or []):
        if not isinstance(item, dict):
            continue
        entity_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        merged_matched_entities.append(item)

    merged_slice_devices = dict(global_slice.get("devices") or {})
    for dev_name, dev_meta in (local_slice.get("devices") or {}).items():
        if dev_name not in merged_slice_devices:
            merged_slice_devices[dev_name] = dev_meta
            continue
        existing = merged_slice_devices[dev_name] or {}
        existing_interfaces = dict(existing.get("interfaces") or {})
        existing_interfaces.update((dev_meta or {}).get("interfaces") or {})
        merged_dev = dict(existing)
        for key, value in (dev_meta or {}).items():
            if key == "interfaces":
                continue
            if key not in merged_dev or merged_dev.get(key) in (None, "", [], {}):
                merged_dev[key] = value
        merged_dev["interfaces"] = existing_interfaces
        merged_slice_devices[dev_name] = merged_dev

    merged_slice_networks = dict(global_slice.get("networks") or {})
    merged_slice_networks.update(local_slice.get("networks") or {})

    merged_subintent_ids = _merge_unique_list(
        global_context.get("subintent_ids") or [],
        [local_context.get("subintent_id")] if local_context.get("subintent_id") else [],
    )
    matched_names = sorted(merged_exact_matches.keys())
    unmatched_names = sorted({
        item.get("argument")
        for item in merged_unmatched
        if isinstance(item, dict) and item.get("argument")
    })

    requested_count = int((global_context.get("diagnostics") or {}).get("requested_topology_argument_count", 0) or 0)
    requested_count += int((local_context.get("diagnostics") or {}).get("requested_topology_argument_count", 0) or 0)
    exact_match_count = len(merged_exact_matches)
    unmatched_count = len(merged_unmatched)
    confidence = 1.0 if requested_count == 0 else round(exact_match_count / max(1, requested_count), 3)

    return {
        "subintent_ids": merged_subintent_ids,
        "goals": _merge_unique_list(
            global_context.get("goals") or [],
            [local_context.get("goal")] if local_context.get("goal") else [],
        ),
        "requested_topology_entities": _merge_prefer_existing(
            global_context.get("requested_topology_entities") or {},
            local_context.get("requested_topology_entities") or {},
        ),
        "semantic_only_arguments": _merge_prefer_existing(
            global_context.get("semantic_only_arguments") or {},
            local_context.get("semantic_only_arguments") or {},
        ),
        "derived_arguments": _merge_prefer_existing(
            global_context.get("derived_arguments") or {},
            local_context.get("derived_arguments") or {},
        ),
        "notes": _merge_prefer_existing(
            global_context.get("notes") or {},
            local_context.get("notes") or {},
        ),
        "exact_matches": merged_exact_matches,
        "matched_topology_entities": merged_matched_entities,
        "unmatched_topology_arguments": merged_unmatched,
        "slice_topology": {
            "devices": merged_slice_devices,
            "networks": merged_slice_networks,
        },
        "confidence": confidence,
        "needs_human": bool(global_context.get("needs_human", False) or local_context.get("needs_human", False)),
        "diagnostics": {
            "processed_subintent_count": len(merged_subintent_ids),
            "requested_topology_argument_count": requested_count,
            "exact_match_count": exact_match_count,
            "unmatched_topology_argument_count": unmatched_count,
            "matched_argument_names": matched_names,
            "unmatched_argument_names": unmatched_names,
        },
        "warnings": _merge_unique_list(
            global_context.get("warnings") or [],
            local_context.get("warnings") or [],
        ),
    }

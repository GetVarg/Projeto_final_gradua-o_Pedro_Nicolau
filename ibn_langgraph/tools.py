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

def normalize_cisco_icmp_type_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Normaliza ICMP echo-request para ACL Cisco.

    Não escolhe a operação. Apenas completa argumentos de uma operação
    de firewall/ACL que já foi escolhida.
    """

    if op_name not in {"firewall_input_accept", "firewall_input_drop"}:
        return []

    updated = []

    semantic_args = (arguments or {}).get("semantic_only_arguments") or {}
    derived_args = (arguments or {}).get("derived_arguments") or {}
    notes = (arguments or {}).get("notes") or {}

    values = []

    for source_name, source_dict in (
        ("bound_args", bound_args),
        ("semantic_only_arguments", semantic_args),
        ("derived_arguments", derived_args),
        ("notes", notes),
    ):
        if not isinstance(source_dict, dict):
            continue

        for key in ("icmp_type", "icmp_message_type", "icmp_kind", "traffic_type", "protocol"):
            value = source_dict.get(key)
            if value is not None:
                values.append((str(value).lower(), f"{source_name}.{key}"))

    joined = " ".join(value for value, _ in values)

    if "echo-request" in joined or "echo_request" in joined or "echo request" in joined or "ping" in joined:
        if bound_args.get("protocol") != "icmp":
            bound_args["protocol"] = "icmp"
            binding_sources["protocol"] = "normalized_cisco_icmp_type"
            resolution_mode["protocol"] = "deterministic_normalization"
            updated.append("protocol")

        if bound_args.get("icmp_type") != "echo-request":
            bound_args["icmp_type"] = "echo-request"
            binding_sources["icmp_type"] = "normalized_cisco_icmp_type"
            resolution_mode["icmp_type"] = "deterministic_normalization"
            updated.append("icmp_type")

    return updated

def validate_bgp_local_as_bound_args(
    op_name: str,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Remove local_as inválido para impedir comandos como:
      router bgp false

    Não inventa ASN. Apenas impede que valor inválido seja compilado.
    """

    bgp_ops = {
        "bgp_neighbor_config",
        "bgp_network_advertise",
        "bgp_neighbor_remove",
        "bgp_peer_group_config",
        "bgp_neighbor_peer_group_bind",
        "bgp_peer_group_remove",
        "bgp_neighbor_med_set",
        "bgp_as_path_multipath_enable",
        "bgp_local_preference_set",
        "bgp_route_map_apply",
        "bgp_peer_group_password_set",
        "bgp_neighbor_password_set",
    }

    if op_name not in bgp_ops:
        return []

    if "local_as" not in bound_args:
        return []

    value = bound_args.get("local_as")

    invalid_values = {
        "",
        "false",
        "true",
        "none",
        "null",
        "router",
        "bgp",
    }

    if isinstance(value, bool) or str(value).strip().lower() in invalid_values:
        bound_args.pop("local_as", None)
        binding_sources.pop("local_as", None)
        resolution_mode.pop("local_as", None)
        return ["local_as"]

    try:
        asn = int(str(value).strip())
    except ValueError:
        bound_args.pop("local_as", None)
        binding_sources.pop("local_as", None)
        resolution_mode.pop("local_as", None)
        return ["local_as"]

    if not (1 <= asn <= 4294967295):
        bound_args.pop("local_as", None)
        binding_sources.pop("local_as", None)
        resolution_mode.pop("local_as", None)
        return ["local_as"]

    bound_args["local_as"] = str(asn)
    return []

def normalize_ipv4_route_cidr_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Normaliza argumentos de rota IPv4 no formato Cisco IOS.

    Entrada esperada:
        dst_cidr = "172.16.9.0/24"

    Saída:
        dst_network = "172.16.9.0"
        dst_mask = "255.255.255.0"

    Também corrige casos em que o modelo colocou o CIDR diretamente em
    dst_network ou dst_mask.
    """

    route_ops = {
        "static_route_add_frr",
        "static_route_del_frr",
        "static_route_blackhole_add",
    }

    if op_name not in route_ops:
        return []

    updated = []

    def _find_candidate_cidr() -> tuple[str | None, str]:
        candidates = []

        for key in ("dst_cidr", "destination_prefix", "prefix", "destination"):
            if key in bound_args:
                candidates.append((bound_args.get(key), f"bound_args.{key}"))

        semantic_args = (arguments or {}).get("semantic_only_arguments") or {}
        if isinstance(semantic_args, dict):
            for key in ("dst_cidr", "destination_prefix", "dst_ip", "prefix", "destination"):
                if key in semantic_args:
                    candidates.append((semantic_args.get(key), f"semantic_only_arguments.{key}"))

        derived_args = (arguments or {}).get("derived_arguments") or {}
        if isinstance(derived_args, dict):
            for key in ("dst_cidr", "destination_prefix", "prefix", "destination"):
                if key in derived_args and isinstance(derived_args.get(key), str):
                    candidates.append((derived_args.get(key), f"derived_arguments.{key}"))

        # Também trata o caso errado: dst_network recebeu "172.16.0.0/24"
        for key in ("dst_network", "dst_mask"):
            if key in bound_args:
                candidates.append((bound_args.get(key), f"bound_args.{key}"))

        for value, source in candidates:
            if isinstance(value, str):
                value = value.strip()
                if "/" in value:
                    return value, source

        return None, ""

    cidr, source = _find_candidate_cidr()
    if not cidr:
        return []

    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return []

    if network.version != 4:
        return []

    normalized_source = f"normalized_ipv4_route_cidr:{source}"

    dst_network = str(network.network_address)
    dst_mask = str(network.netmask)

    if bound_args.get("dst_network") != dst_network:
        bound_args["dst_network"] = dst_network
        binding_sources["dst_network"] = normalized_source
        resolution_mode["dst_network"] = "deterministic_normalization"
        updated.append("dst_network")

    if bound_args.get("dst_mask") != dst_mask:
        bound_args["dst_mask"] = dst_mask
        binding_sources["dst_mask"] = normalized_source
        resolution_mode["dst_mask"] = "deterministic_normalization"
        updated.append("dst_mask")

    # Mantém dst_cidr como argumento auxiliar, caso o template/operação use.
    if "dst_cidr" not in bound_args:
        bound_args["dst_cidr"] = str(network.with_prefixlen)
        binding_sources["dst_cidr"] = normalized_source
        resolution_mode["dst_cidr"] = "deterministic_normalization"
        updated.append("dst_cidr")

    return updated

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


# -------------------------------------
import ipaddress
import os


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _first_present(*items):
    for value, source in items:
        if value not in (None, "", {}, []):
            return value, source
    return None, ""


def _get_nested_arg(arguments: dict, root_name: str, key: str):
    root = _as_dict(_as_dict(arguments).get(root_name))
    return root.get(key)


def _find_arg_by_alias(arguments: dict, bound_args: dict, aliases: list[str]):
    candidates = []

    for alias in aliases:
        if alias in bound_args:
            candidates.append((bound_args.get(alias), f"bound_args.{alias}"))

    for root_name in ("semantic_only_arguments", "derived_arguments", "topology_arguments"):
        root = _as_dict(_as_dict(arguments).get(root_name))
        for alias in aliases:
            if alias in root:
                value = root.get(alias)
                if isinstance(value, dict):
                    # Alguns extractors usam {"kind": "prefix", "prefix": "..."}.
                    for inner_key in ("prefix", "value", "cidr"):
                        if inner_key in value:
                            candidates.append((value.get(inner_key), f"{root_name}.{alias}.{inner_key}"))
                else:
                    candidates.append((value, f"{root_name}.{alias}"))

    for value, source in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip(), source
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value, source

    return None, ""


def _ipv4_network_mask_from_cidr(value: object):
    if not isinstance(value, str) or "/" not in value:
        return None

    try:
        network = ipaddress.ip_network(value.strip(), strict=False)
    except ValueError:
        return None

    if network.version != 4:
        return None

    return str(network.network_address), str(network.netmask), str(network.with_prefixlen)


def _ipv4_network_wildcard_from_cidr(value: object):
    if not isinstance(value, str) or "/" not in value:
        return None

    try:
        network = ipaddress.ip_network(value.strip(), strict=False)
    except ValueError:
        return None

    if network.version != 4:
        return None

    wildcard_int = int(network.hostmask)
    wildcard = str(ipaddress.IPv4Address(wildcard_int))
    return str(network.network_address), wildcard, str(network.with_prefixlen)


def normalize_cisco_route_cidr_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Cisco IPv4 static route:
      dst_cidr "172.16.9.0/24" -> dst_network "172.16.9.0", dst_mask "255.255.255.0"

    Corrige também quando o modelo colocou o CIDR indevidamente em dst_network/dst_mask.
    """

    route_ops = {
        "static_route_add_frr",
        "static_route_del_frr",
        "static_route_blackhole_add",
    }

    if op_name not in route_ops:
        return []

    cidr, source = _find_arg_by_alias(
        arguments,
        bound_args,
        [
            "dst_cidr",
            "destination_prefix",
            "destination_network",
            "prefix",
            "network",
            "destination",
            "dst_network",
            "dst_mask",
        ],
    )

    parsed = _ipv4_network_mask_from_cidr(cidr)
    if not parsed:
        return []

    dst_network, dst_mask, normalized_cidr = parsed
    normalized_source = f"normalized_cisco_route_cidr:{source}"
    updated = []

    if bound_args.get("dst_network") != dst_network:
        bound_args["dst_network"] = dst_network
        binding_sources["dst_network"] = normalized_source
        resolution_mode["dst_network"] = "deterministic_normalization"
        updated.append("dst_network")

    if bound_args.get("dst_mask") != dst_mask:
        bound_args["dst_mask"] = dst_mask
        binding_sources["dst_mask"] = normalized_source
        resolution_mode["dst_mask"] = "deterministic_normalization"
        updated.append("dst_mask")

    if "dst_cidr" not in bound_args:
        bound_args["dst_cidr"] = normalized_cidr
        binding_sources["dst_cidr"] = normalized_source
        resolution_mode["dst_cidr"] = "deterministic_normalization"
        updated.append("dst_cidr")

    return updated


def normalize_cisco_acl_cidr_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Cisco ACL usa wildcard mask:
      src_cidr "172.16.7.0/24" -> src_network "172.16.7.0", src_wildcard "0.0.0.255"
      dst_cidr "172.16.8.0/24" -> dst_network "172.16.8.0", dst_wildcard "0.0.0.255"
    """

    acl_ops = {
        "firewall_input_drop",
        "firewall_input_accept",
        "firewall_output_drop",
        "firewall_output_accept",
        "firewall_forward_drop",
        "firewall_drop_subnet_pair",
    }

    if op_name not in acl_ops:
        return []

    updated = []

    for side in ("src", "dst"):
        cidr_key = f"{side}_cidr"
        network_key = f"{side}_network"
        wildcard_key = f"{side}_wildcard"

        cidr, source = _find_arg_by_alias(
            arguments,
            bound_args,
            [
                cidr_key,
                f"{side}_prefix",
                f"{side}_network",
            ],
        )

        parsed = _ipv4_network_wildcard_from_cidr(cidr)
        if not parsed:
            continue

        network, wildcard, normalized_cidr = parsed
        normalized_source = f"normalized_cisco_acl_cidr:{source}"

        if bound_args.get(network_key) != network:
            bound_args[network_key] = network
            binding_sources[network_key] = normalized_source
            resolution_mode[network_key] = "deterministic_normalization"
            updated.append(network_key)

        if bound_args.get(wildcard_key) != wildcard:
            bound_args[wildcard_key] = wildcard
            binding_sources[wildcard_key] = normalized_source
            resolution_mode[wildcard_key] = "deterministic_normalization"
            updated.append(wildcard_key)

        if cidr_key not in bound_args:
            bound_args[cidr_key] = normalized_cidr
            binding_sources[cidr_key] = normalized_source
            resolution_mode[cidr_key] = "deterministic_normalization"
            updated.append(cidr_key)

    return updated


def normalize_cisco_bgp_network_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Cisco BGP network:
      prefix "172.16.2.0/24" -> network "172.16.2.0", mask "255.255.255.0"
    """

    if op_name != "bgp_network_advertise":
        return []

    cidr, source = _find_arg_by_alias(
        arguments,
        bound_args,
        [
            "prefix",
            "advertise_prefix",
            "advertised_prefix",
            "advertised_network",
            "network_prefix",
            "dst_cidr",
            "destination_prefix",
            "network",
            "mask",
        ],
    )

    parsed = _ipv4_network_mask_from_cidr(cidr)
    if not parsed:
        return []

    network, mask, normalized_cidr = parsed
    normalized_source = f"normalized_cisco_bgp_prefix:{source}"
    updated = []

    if bound_args.get("network") != network:
        bound_args["network"] = network
        binding_sources["network"] = normalized_source
        resolution_mode["network"] = "deterministic_normalization"
        updated.append("network")

    if bound_args.get("mask") != mask:
        bound_args["mask"] = mask
        binding_sources["mask"] = normalized_source
        resolution_mode["mask"] = "deterministic_normalization"
        updated.append("mask")

    if "prefix" not in bound_args:
        bound_args["prefix"] = normalized_cidr
        binding_sources["prefix"] = normalized_source
        resolution_mode["prefix"] = "deterministic_normalization"
        updated.append("prefix")

    return updated


def normalize_cisco_rate_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Normaliza aliases de taxa:
      outbound_rate, police_rate, rate, bandwidth, limit -> rate_mbit
    """

    if op_name not in {"tc_egress_rate_limit", "tc_ingress_police"}:
        return []

    if "rate_mbit" in bound_args:
        return []

    value, source = _find_arg_by_alias(
        arguments,
        bound_args,
        [
            "rate_mbit",
            "rate",
            "bandwidth",
            "limit",
            "rate_limit",
            "outbound_rate",
            "police_rate",
            "ingress_rate",
            "egress_rate",
        ],
    )

    parsed = _parse_rate_mbit_compat(value)
    if parsed is None:
        return []

    bound_args["rate_mbit"] = parsed
    binding_sources["rate_mbit"] = f"normalized_cisco_rate:{source}"
    resolution_mode["rate_mbit"] = "deterministic_normalization"
    return ["rate_mbit"]


def _parse_rate_mbit_compat(value: object) -> int | None:
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
            number = text[:-len(suffix)]
            try:
                return int(float(number))
            except ValueError:
                return None

    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_cisco_bgp_as_bound_args(
    op_name: str,
    arguments: dict,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Normaliza aliases de ASN para operações BGP.

    Casos cobertos:
    - bgp_asn / bgp_process_id / asn -> local_as
    - remote_as pode ser usado como local_as quando o benchmark não fornece outro ASN local.
    - fallback opcional DEFAULT_BGP_LOCAL_AS, por padrão 65000.
    """

    bgp_ops = {
        "bgp_neighbor_config",
        "bgp_network_advertise",
        "bgp_neighbor_remove",
        "bgp_peer_group_config",
        "bgp_neighbor_peer_group_bind",
        "bgp_peer_group_remove",
        "bgp_neighbor_med_set",
        "bgp_as_path_multipath_enable",
        "bgp_local_preference_set",
        "bgp_route_map_apply",
        "bgp_peer_group_password_set",
        "bgp_neighbor_password_set",
    }

    if op_name not in bgp_ops:
        return []

    updated = []

    local_as, local_source = _find_arg_by_alias(
        arguments,
        bound_args,
        [
            "local_as",
            "bgp_asn",
            "bgp_process_id",
            "asn",
            "as_number",
            "local_asn",
        ],
    )

    remote_as, remote_source = _find_arg_by_alias(
        arguments,
        bound_args,
        [
            "remote_as",
            "remote_asn",
            "bgp_asn",
            "bgp_process_id",
            "asn",
            "as_number",
        ],
    )

    if "local_as" not in bound_args:
        if local_as is not None:
            bound_args["local_as"] = str(local_as)
            binding_sources["local_as"] = f"normalized_cisco_bgp_as:{local_source}"
            resolution_mode["local_as"] = "deterministic_normalization"
            updated.append("local_as")
        elif remote_as is not None:
            bound_args["local_as"] = str(remote_as)
            binding_sources["local_as"] = f"normalized_cisco_bgp_as:{remote_source}"
            resolution_mode["local_as"] = "deterministic_normalization"
            updated.append("local_as")
        else:
            default_as = os.environ.get("DEFAULT_BGP_LOCAL_AS", "65000")
            if default_as:
                bound_args["local_as"] = default_as
                binding_sources["local_as"] = "default:DEFAULT_BGP_LOCAL_AS"
                resolution_mode["local_as"] = "deterministic_default"
                updated.append("local_as")

    if op_name in {"bgp_neighbor_config", "bgp_peer_group_config"} and "remote_as" not in bound_args:
        if remote_as is not None:
            bound_args["remote_as"] = str(remote_as)
            binding_sources["remote_as"] = f"normalized_cisco_bgp_as:{remote_source}"
            resolution_mode["remote_as"] = "deterministic_normalization"
            updated.append("remote_as")
        elif "local_as" in bound_args:
            bound_args["remote_as"] = bound_args["local_as"]
            binding_sources["remote_as"] = "default:same_as_local_as"
            resolution_mode["remote_as"] = "deterministic_default"
            updated.append("remote_as")

    return updated

def force_cisco_enable_only_ops(
    op_name: str,
    bound_args: dict,
    binding_sources: dict,
    resolution_mode: dict,
) -> list[str]:
    """
    Operações com nome *_enable não devem compilar para 'no ...'.
    """

    enable_only_ops = {
        "ipv6_forward_enable",
    }

    if op_name not in enable_only_ops:
        return []

    if bound_args.get("enabled") is not True:
        bound_args["enabled"] = True
        binding_sources["enabled"] = "forced_by_enable_operation_name"
        resolution_mode["enabled"] = "deterministic_safety"
        return ["enabled"]

    return []
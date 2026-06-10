# rag_topology.py
"""
RAG utilitÃ¡rio para fatiar a topologia (TopologyRAG).
Este arquivo concentra:
- tokenizaÃ§Ã£o simples
- construÃ§Ã£o de "documentos" por device/network
- retrieval por overlap de tokens
- slicing determinÃ­stico da topologia a partir dos hits
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable, Set

@dataclass
class CmdDoc:
    text: str
    meta: Dict[str, Any]
    tokens: List[str]

class CommandCatalogRAG:
    def __init__(self):
        self.docs: List[CmdDoc] = []
        self.ops_index: Dict[str, Dict[str, Any]] = {}

    def build_from_profile(self, profile: Dict[str, Any]):
        supported = (profile or {}).get("supported_ops") or {}
        self.ops_index = {}

        docs: List[CmdDoc] = []
        for op, meta in supported.items():
            desc = (meta or {}).get("description", "")
            targets = (meta or {}).get("targets", []) or []
            requires = (meta or {}).get("requires", []) or []
            optional = (meta or {}).get("optional", []) or []

            # Text used for retrieval (no requires here is OK, but you can include it if you want)
            txt = f"op {op} | description {desc} | targets {' '.join(targets)}"

            doc_meta = {
                "op": op,
                "description": desc,
                "targets": list(targets),
                "requires": list(requires),
                "optional": list(optional),
                "args_schema": (meta or {}).get("args_schema", (meta or {}).get("schema", {})) or {},
                "examples": (meta or {}).get("examples", []) or [],
            }
            self.ops_index[op] = doc_meta
            docs.append(CmdDoc(text=txt, meta={"op": op}, tokens=_tokenize(txt)))

        self.docs = docs
    
    def list_ops(self) -> List[str]:
        return sorted(self.ops_index.keys())
    

    def retrieve(self, query: str, min_score: float = 0.18) -> List[Dict[str, Any]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scored: List[Tuple[float, CmdDoc]] = []

        for d in self.docs:
            d_set = set(d.tokens or [])
            inter = len(q_set.intersection(d_set))
            if inter == 0:
                continue
            score = inter / max(1, len(q_set))
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for s, d in scored:
            if s < min_score:
                continue
            op = d.meta.get("op")
            if op in self.ops_index:
                m = self.ops_index[op]
                out.append({"op": op, "description": m.get("description", "")})
        return out
    
    def get_op_spec(self, op_name: str) -> Dict[str, Any]:
        m = self.ops_index.get(op_name) or {}
        return {
            "op": op_name,
            "description": m.get("description", ""),
            "targets": m.get("targets", []),
            "requires": m.get("requires", []),
            "optional": m.get("optional", []),
            "args_schema": m.get("args_schema", {}),
            "examples": m.get("examples", []),
        }

def load_cli_command_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    seen = set()

    def add_name(name: str) -> None:
        if isinstance(name, str):
            n = name.strip()
            if n and n not in seen:
                seen.add(n)
                out.append(n)

    if isinstance(data, dict):
        for _, meta in data.items():
            if not isinstance(meta, dict):
                continue

            # caso o catÃ¡logo tenha campo explÃ­cito
            add_name(meta.get("command"))
            add_name(meta.get("command_name"))
            add_name(meta.get("name"))

            # caso tenha exemplos estruturados
            examples = meta.get("examples") or []
            if isinstance(examples, list):
                for ex in examples:
                    if isinstance(ex, dict):
                        add_name(ex.get("command"))
                        add_name(ex.get("command_name"))

    return out

def compact_command_catalog_for_discretize() -> List[Dict[str, str]]:
    """
    Return a compact full-catalog view for discretization.

    Output:
    [
        {"op": "...", "description": "..."},
        ...
    ]

    Source of truth:
    - prefer the already built CommandCatalogRAG
    - if empty, lazily load command_catalog.json and build it
    """
    rag = get_cmd_rag()

    if not rag.list_ops():
        catalog_path = Path("data/command_catalog.json")
        with catalog_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        rag.build_from_profile(profile)

    out: List[Dict[str, str]] = []
    for op in rag.list_ops():
        spec = rag.get_op_spec(op)
        out.append({
            "op": op,
            "description": (spec.get("description") or "").strip(),
        })

    return out

_CMD_RAG = CommandCatalogRAG()

def get_cmd_rag() -> CommandCatalogRAG:
    return _CMD_RAG

@dataclass
class TopoDoc:
    """Documento indexÃ¡vel: texto, metadados, tokens e Ã¢ncoras grounded da topologia."""
    text: str
    meta: Dict[str, Any]
    tokens: List[str]
    anchors: List[str]

class TopologyRAG:
    """RAG simples por overlap de tokens (sem embeddings)."""

    def __init__(self) -> None:
        self.docs: List[TopoDoc] = []

    def build(self, topo: Dict[str, Any]) -> None:
        """ConstrÃ³i/atualiza os docs a partir da topologia completa."""
        docs: List[TopoDoc] = []
        devices = topo.get("devices", {}) or {}
        networks = topo.get("networks", {}) or {}

        for name, meta in devices.items():
            txt = _build_device_doc(name, meta or {})
            docs.append(
                TopoDoc(
                    text=txt,
                    meta={"kind": "device", "name": name},
                    tokens=_tokenize_loose(txt),
                    anchors=_extract_doc_anchors(name, meta or {}),
                )
            )

        for net_name, net_meta in networks.items():
            cidr = (net_meta or {}).get("cidr") or ""
            gw = (net_meta or {}).get("gateway_device") or ""
            txt = f"network {net_name} cidr {cidr} gateway_device {gw}"
            docs.append(
                TopoDoc(
                    text=txt,
                    meta={"kind": "network", "name": net_name},
                    tokens=_tokenize_loose(txt),
                    anchors=_extract_network_anchors(net_name, net_meta or {}),
                )
            )

        self.docs = docs

    def retrieve(self, query: str, min_score: float = 0.18) -> List[TopoDoc]:
        """
        EstratÃ©gia:
        1) tenta matching exato com valores crus da query
        2) se nÃ£o houver match exato em nenhum doc, usa fallback por tokenizaÃ§Ã£o fraca
        """
        query_values = _extract_query_values(query)
        if not query_values:
            return []

        scored: List[Tuple[float, TopoDoc]] = []

        for d in self.docs:
            if not d.tokens and not d.anchors:
                continue

            score = _score_topology_doc(
                query_values=query_values,
                doc_anchors=set(d.anchors or []),
                doc_tokens=set(d.tokens or []),
            )

            if score <= 0.0:
                continue

            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored if s >= min_score]
    
    def retrieve_adaptive(
        self,
        query: str,
        thresholds: Iterable[float] = (0.35, 0.30, 0.25),
    ) -> Tuple[List[TopoDoc], float]:
        """
        Controlled degradation retrieval WITHOUT hit-count controls:

        - tenta scorear todos os docs
        - thresholds continuam valendo
        - o score agora Ã© exact-first com fallback lexical
        """
        query_values = _extract_query_values(query)
        if not query_values:
            return ([], 1.0)

        scored: List[Tuple[float, TopoDoc]] = []

        for d in self.docs:
            if not d.tokens and not d.anchors:
                continue

            score = _score_topology_doc(
                query_values=query_values,
                doc_anchors=set(d.anchors or []),
                doc_tokens=set(d.tokens or []),
            )

            if score <= 0.0:
                continue

            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)

        thr_list = list(thresholds) or [0.25]

        used_thr = thr_list[-1]
        selected: List[TopoDoc] = []

        for thr in thr_list:
            tmp = [doc for s, doc in scored if s >= thr]
            if tmp:
                used_thr = thr
                selected = tmp
                break

        return (selected, float(used_thr))

# Singleton de RAG (mantÃ©m o comportamento global existente no seu script)
_TOPO_RAG = TopologyRAG()


def get_rag() -> TopologyRAG:
    """Retorna o singleton do TopologyRAG."""
    return _TOPO_RAG


def find_interface_by_name(topo: dict, interface_name: str) -> dict | None:
    devices = (topo or {}).get("devices") or {}

    for dev_name, dev_meta in devices.items():
        interfaces = dev_meta.get("interfaces") or {}
        if interface_name in interfaces:
            return {
                "owner": dev_name,
                "name": interface_name,
                "data": interfaces[interface_name],
            }

    return None


def find_ip_in_topology(topo: dict, ip_value: str) -> dict | None:
    devices = (topo or {}).get("devices") or {}

    for dev_name, dev_meta in devices.items():
        interfaces = dev_meta.get("interfaces") or {}
        for if_name, if_meta in interfaces.items():
            if (if_meta or {}).get("ip") == ip_value:
                return {
                    "owner": dev_name,
                    "interface": if_name,
                    "ip": ip_value,
                    "data": if_meta,
                }

    return None


def find_cidr_in_topology(topo: dict, cidr_value: str) -> dict | None:
    devices = (topo or {}).get("devices") or {}
    networks = (topo or {}).get("networks") or {}

    for dev_name, dev_meta in devices.items():
        interfaces = dev_meta.get("interfaces") or {}
        for if_name, if_meta in interfaces.items():
            if (if_meta or {}).get("cidr") == cidr_value:
                return {
                    "source": "interface",
                    "owner": dev_name,
                    "interface": if_name,
                    "cidr": cidr_value,
                    "data": if_meta,
                }

    for net_name, net_meta in networks.items():
        if (net_meta or {}).get("cidr") == cidr_value:
            return {
                "source": "network",
                "name": net_name,
                "cidr": cidr_value,
                "data": net_meta,
            }

    return None


def find_peer_interface(topo: dict, interface_name: str) -> dict | None:
    iface = find_interface_by_name(topo, interface_name)
    if not iface:
        return None

    peer_name = (iface.get("data") or {}).get("peer")
    if not isinstance(peer_name, str) or not peer_name.strip():
        return None

    peer = find_interface_by_name(topo, peer_name.strip())
    if not peer:
        return None

    return {
        "local_interface": interface_name,
        "peer_interface": peer["name"],
        "peer_owner": peer["owner"],
        "peer_data": peer["data"],
    }


def find_peer_ip_of_interface(topo: dict, interface_name: str) -> dict | None:
    peer = find_peer_interface(topo, interface_name)
    if not peer:
        return None

    peer_ip = (peer.get("peer_data") or {}).get("ip")
    if not isinstance(peer_ip, str) or not peer_ip.strip():
        return None

    return {
        "local_interface": interface_name,
        "peer_interface": peer["peer_interface"],
        "peer_owner": peer["peer_owner"],
        "peer_ip": peer_ip.strip(),
        "peer_data": peer["peer_data"],
    }

def _normalize_device_ref(topo: dict, value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    devices = (topo or {}).get("devices") or {}

    if raw in devices:
        return raw

    normalized = re.sub(
        r"^(router|device|host|switch)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()

    if normalized in devices:
        return normalized

    return None

def _normalize_interface_ref(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    raw = re.sub(r"^interface\s+", "", raw, flags=re.IGNORECASE).strip()
    return raw or None

def resolve_structured_topology_argument(arg_name: str, arg_value: Any, topo: dict) -> dict | None:
    if not isinstance(arg_value, dict):
        return None

    kind = arg_value.get("kind")
    if not isinstance(kind, str):
        return None

    if kind == "interface_connecting_to_device":
        local_device = arg_value.get("local_device")
        remote_device = arg_value.get("remote_device")

        local_device = _normalize_device_ref(topo, local_device)
        remote_device = _normalize_device_ref(topo, remote_device)

        if not local_device or not remote_device:
            return None

        devices = (topo or {}).get("devices") or {}
        local_interfaces = ((devices.get(local_device) or {}).get("interfaces") or {})

        for if_name, if_meta in local_interfaces.items():
            peer_name = (if_meta or {}).get("peer")
            if not isinstance(peer_name, str) or not peer_name.strip():
                continue

            peer = find_interface_by_name(topo, peer_name.strip())
            if peer is not None and peer.get("owner") == remote_device:
                return {
                    "kind": "interface",
                    "query": arg_value,
                    "match": {
                        "owner": local_device,
                        "name": if_name,
                        "data": if_meta,
                    },
                }

        return None
    
    if kind == "device_connected_to_interface":
        interface_name = arg_value.get("interface")
        if not isinstance(interface_name, str):
            return None

        peer = find_peer_interface(topo, interface_name.strip())
        if peer is None:
            return None

        peer_owner = peer.get("peer_owner")
        devices = (topo or {}).get("devices") or {}

        if not isinstance(peer_owner, str) or peer_owner not in devices:
            return None

        return {
            "kind": "device",
            "query": arg_value,
            "match": {
                "name": peer_owner,
                "data": devices[peer_owner],
            },
        }

    if kind == "neighbor_ip_connected_to_interface":
        interface_name = arg_value.get("interface")
        interface_name = _normalize_interface_ref(interface_name)
        if not interface_name:
            return None

        peer_ip = find_peer_ip_of_interface(topo, interface_name)
        if peer_ip is None:
            return None

        return {
            "kind": "peer_ip_of_interface",
            "query": arg_value,
            "match": peer_ip,
        }

    if kind == "lan_subnet_of_device":
        device = _normalize_device_ref(topo, arg_value.get("device"))
        if not device:
            return None

        networks = (topo or {}).get("networks") or {}

        for net_name, net_meta in networks.items():
            if (net_meta or {}).get("gateway_device") == device and (net_meta or {}).get("cidr"):
                return {
                    "kind": "cidr",
                    "query": arg_value,
                    "match": {
                        "source": "network",
                        "name": net_name,
                        "cidr": net_meta.get("cidr"),
                        "data": net_meta,
                    },
                }

        devices = (topo or {}).get("devices") or {}
        interfaces = ((devices.get(device) or {}).get("interfaces") or {})

        for if_name, if_meta in interfaces.items():
            cidr = (if_meta or {}).get("cidr")
            if isinstance(cidr, str) and cidr.strip():
                return {
                    "kind": "cidr",
                    "query": arg_value,
                    "match": {
                        "source": "interface",
                        "owner": device,
                        "interface": if_name,
                        "cidr": cidr.strip(),
                        "data": if_meta,
                    },
                }

        return None
    
    return None


def exact_match_argument(
    arg_name: str,
    arg_value: Any,
    arguments: dict,
    notes: dict,
    topo: dict,
) -> dict | None:
    if isinstance(arg_value, bool):
        return None

    structured_match = resolve_structured_topology_argument(arg_name, arg_value, topo)
    if structured_match is not None:
        return structured_match

    if not isinstance(arg_value, str):
        return None

    value = arg_value.strip()
    if not value:
        return None

    devices = (topo or {}).get("devices") or {}
    networks = (topo or {}).get("networks") or {}

    if value in devices:
        return {
            "kind": "device",
            "query": value,
            "match": {
                "name": value,
                "data": devices[value],
            },
        }

    iface = find_interface_by_name(topo, value)
    if iface is not None:
        return {
            "kind": "interface",
            "query": value,
            "match": iface,
        }

    ip_match = find_ip_in_topology(topo, value)
    if ip_match is not None:
        return {
            "kind": "ip",
            "query": value,
            "match": ip_match,
        }

    cidr_match = find_cidr_in_topology(topo, value)
    if cidr_match is not None:
        return {
            "kind": "cidr",
            "query": value,
            "match": cidr_match,
        }

    if value in networks:
        return {
            "kind": "network",
            "query": value,
            "match": {
                "name": value,
                "data": networks[value],
            },
        }

    if arg_name in ("target", "router", "router_id", "forwarding_device", "device"):
        router_name = value.replace("router ", "").strip()
        if router_name in devices:
            return {
                "kind": "device",
                "query": value,
                "match": {
                    "name": router_name,
                    "data": devices[router_name],
                },
            }

    if arg_name in ("interface", "interface_id", "next_hop_interface"):
        interface_name = value.replace("interface ", "").strip()
        iface = find_interface_by_name(topo, interface_name)
        if iface is not None:
            return {
                "kind": "interface",
                "query": value,
                "match": iface,
            }

    if arg_name in ("prefix", "destination_prefix", "destination_network", "network"):
        cidr_match = find_cidr_in_topology(topo, value)
        if cidr_match is not None:
            return {
                "kind": "cidr",
                "query": value,
                "match": cidr_match,
            }

    if arg_name in ("next_hop_router", "next_hop"):
        router_name = value.replace("router ", "").strip()
        if router_name in devices:
            return {
                "kind": "device",
                "query": value,
                "match": {
                    "name": router_name,
                    "data": devices[router_name],
                },
            }

    if arg_name in ("next_hop_source", "neighbor_source"):
        interface_name = (
            arguments.get("interface")
            or arguments.get("interface_id")
            or arguments.get("next_hop_interface")
            or notes.get("interface")
        )
        if isinstance(interface_name, str) and interface_name.strip():
            peer_ip = find_peer_ip_of_interface(topo, interface_name.strip())
            if peer_ip is not None:
                return {
                    "kind": "peer_ip_of_interface",
                    "query": value,
                    "match": peer_ip,
                }

    return None


def _append_unique(items: list[str], value: str | None) -> None:
    """Append a string only once, preserving the original order."""
    if isinstance(value, str) and value and value not in items:
        items.append(value)


def _network_names_for_cidr(topo: dict, cidr: str | None) -> list[str]:
    """Return network names that advertise the given CIDR in the topology."""
    if not isinstance(cidr, str) or not cidr:
        return []

    out = []
    networks = (topo or {}).get("networks") or {}
    for net_name, net_meta in networks.items():
        if (net_meta or {}).get("cidr") == cidr:
            out.append(net_name)
    return out


def _interfaces_for_owner_and_cidr(topo: dict, owner: str | None, cidr: str | None) -> list[str]:
    """Return owner interfaces whose configured CIDR matches the provided prefix."""
    if not isinstance(owner, str) or not owner or not isinstance(cidr, str) or not cidr:
        return []

    out = []
    devices = (topo or {}).get("devices") or {}
    interfaces = ((devices.get(owner) or {}).get("interfaces") or {})
    for if_name, if_meta in interfaces.items():
        if (if_meta or {}).get("cidr") == cidr:
            out.append(if_name)
    return out


def _build_resolved_entities(topo: dict, exact_matches: dict) -> dict:
    """Collapse exact topology matches into a compact entity map for downstream nodes."""
    resolved = {}

    def put(key: str, value: str | None) -> None:
        if isinstance(value, str) and value and key not in resolved:
            resolved[key] = value

    for _, result in (exact_matches or {}).items():
        kind = result.get("kind")
        match = result.get("match") or {}

        if kind == "device":
            put("device", match.get("name"))
            continue

        if kind == "interface":
            put("device", match.get("owner"))
            put("interface", match.get("name"))
            put("interface_ip", (match.get("data") or {}).get("ip"))
            continue

        if kind == "ip":
            put("device", match.get("owner"))
            put("interface", match.get("interface"))
            put("interface_ip", match.get("ip"))
            continue

        if kind == "cidr":
            put("destination_prefix", match.get("cidr"))
            if match.get("source") == "interface":
                put("destination_owner", match.get("owner"))
            elif match.get("source") == "network":
                put("destination_owner", (match.get("data") or {}).get("gateway_device"))
            continue

        if kind == "peer_ip_of_interface":
            put("peer_device", match.get("peer_owner"))
            put("peer_interface", match.get("peer_interface"))
            put("peer_ip", match.get("peer_ip"))

    if "destination_owner" in resolved and "destination_prefix" in resolved:
        owner_ifaces = _interfaces_for_owner_and_cidr(
            topo,
            resolved.get("destination_owner"),
            resolved.get("destination_prefix"),
        )
        if owner_ifaces:
            put("destination_interface", owner_ifaces[0])
            devices = (topo or {}).get("devices") or {}
            if_meta = ((devices.get(resolved["destination_owner"]) or {}).get("interfaces") or {}).get(owner_ifaces[0]) or {}
            put("destination_interface_ip", if_meta.get("ip"))

    return resolved


def _build_context_slice_summary(topo: dict, resolved_entities: dict) -> dict:
    """Build a compact topology slice summary from already resolved entities."""
    devices = []
    interfaces = []
    networks = []

    for key in ("device", "peer_device", "destination_owner"):
        _append_unique(devices, resolved_entities.get(key))

    for key in ("interface", "peer_interface", "destination_interface"):
        _append_unique(interfaces, resolved_entities.get(key))

    devices_map = (topo or {}).get("devices") or {}
    for if_name in list(interfaces):
        owner = if_name.split("-", 1)[0] if isinstance(if_name, str) and "-" in if_name else None
        if_meta = ((devices_map.get(owner) or {}).get("interfaces") or {}).get(if_name) or {}
        for net_name in _network_names_for_cidr(topo, if_meta.get("cidr")):
            _append_unique(networks, net_name)

    for net_name in _network_names_for_cidr(topo, resolved_entities.get("destination_prefix")):
        _append_unique(networks, net_name)

    if "destination_owner" in resolved_entities and "destination_prefix" in resolved_entities:
        for if_name in _interfaces_for_owner_and_cidr(
            topo,
            resolved_entities.get("destination_owner"),
            resolved_entities.get("destination_prefix"),
        ):
            _append_unique(interfaces, if_name)

    return {
        "devices": devices,
        "interfaces": interfaces,
        "networks": networks,
    }


def _match_to_context_entity(arg_name: str, arg_value: str, result: dict) -> dict:
    """Convert one exact topology match into a descriptive context-evidence record."""
    kind = result.get("kind")
    match = result.get("match") or {}

    entity = {
        "label": arg_name,
        "query": arg_value,
        "kind": kind,
        "match": {},
    }

    if kind == "device":
        entity["match"] = {
            "name": match.get("name"),
            "data": match.get("data") or {},
        }
        return entity

    if kind == "interface":
        entity["match"] = {
            "name": match.get("name"),
            "owner": match.get("owner"),
            "data": match.get("data") or {},
        }
        return entity

    if kind == "ip":
        entity["match"] = {
            "ip": match.get("ip"),
            "owner": match.get("owner"),
            "interface": match.get("interface"),
            "data": match.get("data") or {},
        }
        return entity

    if kind == "cidr":
        entity["match"] = {
            "cidr": match.get("cidr"),
            "source": match.get("source"),
            "name": match.get("name"),
            "owner": match.get("owner"),
            "interface": match.get("interface"),
            "data": match.get("data") or {},
        }
        return entity

    if kind == "peer_ip_of_interface":
        entity["match"] = {
            "local_interface": match.get("local_interface"),
            "peer_interface": match.get("peer_interface"),
            "peer_owner": match.get("peer_owner"),
            "peer_ip": match.get("peer_ip"),
            "peer_data": match.get("peer_data") or {},
        }
        return entity

    entity["match"] = match
    return entity


def _build_context_slice_from_matches(topo: dict, matched_entities: list[dict]) -> dict:
    """Build a reduced topology slice containing only matched devices, interfaces, peers, and related networks."""
    topo = topo or {}
    all_devices = topo.get("devices") or {}
    all_networks = topo.get("networks") or {}

    slice_devices: dict[str, dict] = {}
    slice_networks: dict[str, dict] = {}

    def ensure_device(device_name: str | None) -> None:
        if not isinstance(device_name, str) or not device_name:
            return
        dev_meta = all_devices.get(device_name)
        if dev_meta is None:
            return
        if device_name not in slice_devices:
            base = dict(dev_meta)
            base["interfaces"] = {}
            slice_devices[device_name] = base

    def ensure_network(network_name: str | None) -> None:
        if not isinstance(network_name, str) or not network_name:
            return
        net_meta = all_networks.get(network_name)
        if net_meta is not None:
            slice_networks[network_name] = net_meta

    def networks_for_cidr(cidr: str | None) -> list[str]:
        if not isinstance(cidr, str) or not cidr:
            return []
        out = []
        for net_name, net_meta in all_networks.items():
            if (net_meta or {}).get("cidr") == cidr:
                out.append(net_name)
        return out

    def ensure_interface(owner: str | None, ifname: str | None) -> None:
        if not isinstance(owner, str) or not owner or not isinstance(ifname, str) or not ifname:
            return
        dev_meta = all_devices.get(owner) or {}
        iface = (dev_meta.get("interfaces") or {}).get(ifname)
        if iface is None:
            return

        ensure_device(owner)
        slice_devices[owner]["interfaces"][ifname] = iface

        cidr = (iface or {}).get("cidr")
        for net_name in networks_for_cidr(cidr):
            ensure_network(net_name)

        peer = (iface or {}).get("peer")
        if isinstance(peer, str) and "-" in peer:
            peer_owner = peer.split("-", 1)[0]
            ensure_device(peer_owner)
            peer_meta = all_devices.get(peer_owner) or {}
            peer_iface = (peer_meta.get("interfaces") or {}).get(peer)
            if peer_iface is not None:
                slice_devices[peer_owner]["interfaces"][peer] = peer_iface
                peer_cidr = (peer_iface or {}).get("cidr")
                for net_name in networks_for_cidr(peer_cidr):
                    ensure_network(net_name)

    for entity in matched_entities or []:
        if not isinstance(entity, dict):
            continue
        kind = entity.get("kind")
        match = entity.get("match") or {}

        if kind == "device":
            ensure_device(match.get("name"))
            continue

        if kind == "interface":
            ensure_interface(match.get("owner"), match.get("name"))
            continue

        if kind == "ip":
            ensure_interface(match.get("owner"), match.get("interface"))
            continue

        if kind == "cidr":
            ensure_network(match.get("name"))
            ensure_interface(match.get("owner"), match.get("interface"))
            if isinstance(match.get("data"), dict):
                gateway_device = (match.get("data") or {}).get("gateway_device")
                ensure_device(gateway_device)
            continue

        if kind == "peer_ip_of_interface":
            local_if = match.get("local_interface")
            if isinstance(local_if, str) and "-" in local_if:
                ensure_interface(local_if.split("-", 1)[0], local_if)
            ensure_interface(match.get("peer_owner"), match.get("peer_interface"))

    return {
        "devices": slice_devices,
        "networks": slice_networks,
    }


def query_from_state_for_rag(state: Dict[str, Any]) -> str:
    """
    Build a compact RAG query from explicit grounded values.

    Strategy:
    - prefer exact values already extracted from entities
    - include selector grounded constraints when present
    - keep lightweight selector hints only as weak fallback context
    """
    ents = state.get("entities") or []
    selectors = state.get("entity_selectors") or []

    values: List[str] = []

    for e in ents:
        if e is None:
            continue

        value = (
            getattr(e, "value", None)
            if hasattr(e, "value")
            else (e.get("value") if isinstance(e, dict) else None)
        )

        if isinstance(value, str) and value.strip():
            values.append(value.strip())

    for s in selectors:
        if not isinstance(s, dict):
            continue

        for c in (s.get("constraints") or []):
            if not isinstance(c, dict):
                continue

            for key in ["cidr", "name", "ip", "interface"]:
                value = c.get(key)
                if isinstance(value, str) and value.strip() and value.strip() != "...":
                    values.append(value.strip())

    kind_hints: List[str] = []
    for s in selectors:
        if not isinstance(s, dict):
            continue

        kind = (s.get("kind") or "").strip().lower()
        device_type = (s.get("device_type") or "").strip().lower()

        if kind == "device_set" and device_type in {"device", "host", "router", "switch"}:
            kind_hints.append(device_type)

    return " ".join(_ordered_unique(values + kind_hints)).strip()

# -----------------------------
# Helpers internos (RAG)
# -----------------------------

import re
from typing import List


_IP_CIDR_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b")


def _ordered_unique(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip().lower()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)

    return out


def _tokenize_loose(s: str) -> List[str]:
    """
    TokenizaÃ§Ã£o fraca usada apenas como fallback.
    NÃ£o tenta inferir tipo de device por padrÃ£o de nome.
    """
    if not s:
        return []

    s_low = s.lower()

    specials = [m.group(0) for m in _IP_CIDR_RE.finditer(s_low)]
    s_wo_ips = _IP_CIDR_RE.sub(" ", s_low)

    chars: List[str] = []
    for ch in s_wo_ips:
        chars.append(ch if ch.isalnum() else " ")

    base_tokens = [t for t in "".join(chars).split() if t]
    base_tokens = [t for t in base_tokens if len(t) > 1]

    return _ordered_unique(specials + base_tokens)

def _tokenize(s: str) -> List[str]:
    """
    Backward-compatible tokenizer wrapper.

    Keeps older callers working while the topology RAG uses
    exact-first retrieval with loose-token fallback.
    """
    return _tokenize_loose(s)

def _extract_query_values(query: str) -> List[str]:
    """
    Extrai valores crus da query do RAG.
    MantÃ©m identificadores completos, sem tentar classificar por regex.
    """
    if not query:
        return []

    raw_parts = [part.strip().lower() for part in query.split() if isinstance(part, str)]
    return _ordered_unique(raw_parts)


def _extract_doc_anchors(device_name: str, meta: Dict[str, Any]) -> List[str]:
    """
    Extrai Ã¢ncoras grounded do prÃ³prio documento de device.
    Fonte de verdade = topologia, nÃ£o convenÃ§Ã£o de nome.
    """
    anchors: List[str] = []

    if isinstance(device_name, str) and device_name.strip():
        anchors.append(device_name)

    ifaces = meta.get("interfaces", {}) or {}
    for ifname, ifmeta in ifaces.items():
        if isinstance(ifname, str) and ifname.strip():
            anchors.append(ifname)
            anchors.append(f"{device_name}-{ifname}")

        if isinstance(ifmeta, dict):
            peer = ifmeta.get("peer")
            ip = ifmeta.get("ip") or ifmeta.get("ipv4") or ifmeta.get("address")
            cidr = ifmeta.get("cidr")

            if isinstance(peer, str) and peer.strip():
                anchors.append(peer)

            if isinstance(ip, str) and ip.strip():
                anchors.append(ip)

            if isinstance(cidr, str) and cidr.strip():
                anchors.append(cidr)

    for key in ["ip", "ipv4"]:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            anchors.append(value)

    return _ordered_unique(anchors)


def _extract_network_anchors(net_name: str, net_meta: Dict[str, Any]) -> List[str]:
    anchors: List[str] = []

    if isinstance(net_name, str) and net_name.strip():
        anchors.append(net_name)

    cidr = net_meta.get("cidr")
    gw = net_meta.get("gateway_device")

    if isinstance(cidr, str) and cidr.strip():
        anchors.append(cidr)

    if isinstance(gw, str) and gw.strip():
        anchors.append(gw)

    return _ordered_unique(anchors)


def _score_topology_doc(
    query_values: List[str],
    doc_anchors: set[str],
    doc_tokens: set[str],
) -> float:
    """
    Exact-first scoring:
    - primeiro tenta matching exato com os valores crus da query
    - sÃ³ usa tokenizaÃ§Ã£o fraca se nenhum valor exato casar
    """
    q_values = set(_ordered_unique(query_values))
    if not q_values:
        return 0.0

    exact_hits = len(q_values.intersection(doc_anchors))
    if exact_hits > 0:
        exact_score = exact_hits / max(1, len(q_values))

        if exact_hits >= 2:
            exact_score += 0.08
        elif exact_hits == 1:
            exact_score += 0.03

        return min(exact_score, 1.0)

    q_tokens = set()
    for value in q_values:
        q_tokens.update(_tokenize_loose(value))

    if not q_tokens:
        return 0.0

    weak_hits = len(q_tokens.intersection(doc_tokens))
    if weak_hits == 0:
        return 0.0

    return weak_hits / max(1, len(q_tokens))


def _build_device_doc(name: str, meta: Dict[str, Any]) -> str:
    """Serializa um device em texto (nome, tipo, interfaces, ip, peer) para indexaÃ§Ã£o no RAG."""
    parts: List[str] = []
    parts.append(f"device {name}")

    dtype = meta.get("type")
    if dtype:
        parts.append(f"type {dtype}")

    ifaces = meta.get("interfaces", {}) or {}
    for ifname, ifmeta in ifaces.items():
        parts.append(f"interface {ifname}")
        if isinstance(ifmeta, dict):
            ip = ifmeta.get("ip") or ifmeta.get("ipv4") or ifmeta.get("address")
            peer = ifmeta.get("peer")
            if ip:
                parts.append(f"ip {ip}")
            if peer:
                parts.append(f"peer {peer}")

    for k in ["ip", "ipv4"]:
        v = meta.get(k)
        if isinstance(v, str) and v:
            parts.append(f"{k} {v}")

    return " | ".join(parts)

def slice_topology(topo: Dict[str, Any], docs: List[TopoDoc]) -> Dict[str, Any]:
    """Fatia a topologia trazendo devices citados + gateway devices de networks citadas."""
    devices = topo.get("devices", {}) or {}
    networks = topo.get("networks", {}) or {}

    picked_devices: List[str] = []
    picked_networks: List[str] = []

    for d in docs:
        if d.meta.get("kind") == "device":
            name = d.meta.get("name")
            if isinstance(name, str) and name in devices:
                picked_devices.append(name)

    for d in docs:
        if d.meta.get("kind") == "network":
            net_name = d.meta.get("name")
            if isinstance(net_name, str) and net_name in networks:
                picked_networks.append(net_name)

                net_meta = networks.get(net_name, {}) or {}
                gw = net_meta.get("gateway_device")
                if isinstance(gw, str) and gw in devices:
                    picked_devices.append(gw)

    seen = set()
    ordered_devices: List[str] = []
    for x in picked_devices:
        if x not in seen:
            seen.add(x)
            ordered_devices.append(x)

    seen_n = set()
    ordered_networks: List[str] = []
    for n in picked_networks:
        if n not in seen_n:
            seen_n.add(n)
            ordered_networks.append(n)

    sliced_devices = {name: devices[name] for name in ordered_devices if name in devices}
    sliced_networks = {name: networks[name] for name in ordered_networks if name in networks}

    return {
        "devices": sliced_devices,
        "networks": sliced_networks,
    }

def _device_of_peer(peer: str | None) -> str | None:
    if not isinstance(peer, str) or not peer or "-" not in peer:
        return None
    return peer.split("-", 1)[0]

def _edge_router_for_host(topo: dict, host: str) -> str | None:
    """
    Best-effort deterministic: host -> (switch) -> router using peer links.
    Returns router name if found, else None.
    """
    devices = (topo or {}).get("devices") or {}
    h = devices.get(host) or {}
    h_ifaces = h.get("interfaces") or {}

    # 1) host -> switch
    sw = None
    for _, im in h_ifaces.items():
        peer = (im or {}).get("peer")
        sw = _device_of_peer(peer)
        if sw and (devices.get(sw) or {}).get("type") == "switch":
            break
        sw = None

    if not sw:
        return None

    # 2) switch -> router
    sw_ifaces = (devices.get(sw) or {}).get("interfaces") or {}
    for _, sim in sw_ifaces.items():
        peer = (sim or {}).get("peer")
        r = _device_of_peer(peer)
        if r and (devices.get(r) or {}).get("type") == "router":
            return r

    return None

def _first_ip_cidr_for_device(topo: dict, dev: str) -> tuple[str | None, str | None]:
    """Returns (ip, cidr) from the first interface that has an IP."""
    devices = (topo or {}).get("devices") or {}
    meta = devices.get(dev) or {}
    ifaces = meta.get("interfaces") or {}
    for _, im in ifaces.items():
        ip = (im or {}).get("ip")
        cidr = (im or {}).get("cidr")
        if ip:
            return ip, cidr
    return None, None

from typing import Any, Dict, Iterable, List, Set

def collect_device_values(entities: Iterable[Any], device_names: Set[str]) -> List[str]:
    """
    Collect entity values that are valid device names in the current topology.
    Entities can be dict-like {"type": "...", "value": "..."} or objects with .type/.value.
    Returns values in encounter order, without duplicates.
    """
    out: List[str] = []
    seen: Set[str] = set()

    for e in entities or []:
        if isinstance(e, dict):
            v = e.get("value")
        else:
            v = getattr(e, "value", None)

        if not isinstance(v, str):
            continue

        v = v.strip()
        if not v or v not in device_names:
            continue

        if v in seen:
            continue
        seen.add(v)
        out.append(v)

    return out

def build_expanded_rag_query(state: Dict[str, Any], topology: Dict[str, Any]) -> str:
    devices = (topology.get("devices") or {})
    device_names: Set[str] = set(devices.keys())

    anchors: List[str] = []
    anchors += collect_device_values(state.get("entities_local") or [], device_names)
    if not anchors:
        anchors += collect_device_values(state.get("root_entities") or [], device_names)
    if not anchors:
        anchors += collect_device_values(state.get("entities") or [], device_names)

    # de-dup preserving order
    seen = set()
    anchors = [x for x in anchors if not (x in seen or seen.add(x))]

    # -------- NEW: deterministic topology bridging (hosts -> edge routers -> r0) --------
    bridge_entities: List[str] = []

    # Add r0 hub if exists (your lab assumption) â€” safe, closed-world
    if "r0" in device_names:
        bridge_entities.append("r0")

    # If we have hosts among anchors, add their edge routers
    for a in anchors:
        if a in device_names and (devices.get(a) or {}).get("type") == "host":
            r = _edge_router_for_host(topology, a)
            if r and r in device_names:
                bridge_entities.append(r)

    # de-dup bridge entities preserving order
    seen2 = set()
    bridge_entities = [x for x in bridge_entities if not (x in seen2 or seen2.add(x))]

    return " ".join(anchors + bridge_entities).strip()

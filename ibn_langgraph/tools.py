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
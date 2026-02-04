import json
import os
from pathlib import Path

# --- FUNÇÃO QUE O GRAFO VAI USAR ---
def load_topologia(path="data/topologia_augmented.json") -> dict:
    """
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

import re, os
import json
from typing import Dict, List, Any, Tuple, Set, Optional  # O correto é typing, não types
from main import IBNState, Entity, ExecPlan, Intent, Requirement # Importe todos os tipos necessários
from tools import load_topologia, normalize_static_route_step

from langchain_ollama import ChatOllama
OLLAMA_MODEL = "llama3.2:3b"
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json")


def node_command_profile(state: IBNState) -> IBNState:
    # Catálogo fechado de comandos permitidos por tipo de tarefa
    # (poucas famílias, bem ancoradas no seu ambiente real)
    state["command_profile"] = {
        "environment": {
            "emulator": "mininet",
            "routers_stack": "linux+frr",
            "switches_stack": "ovs",
        },

        "supported_ops": {
            "static_route_add": {
                "description": "Add a static route on a router to reach a destination subnet via a next-hop gateway.",
                "targets": ["router"],
                "requires": ["target_network", "exit_gateway"]
            },

            "static_route_del": {
                "description": "Remove a static route from a router so it no longer routes traffic to a destination subnet.",
                "targets": ["router"],
                "requires": ["dst_cidr"]
            },

            "ip_forward_enable": {
                "description": "Enable global IP forwarding on a router or host (OS-level forwarding).",
                "targets": ["router", "host"],
                "requires": []
            },

            "ip_forward_disable": {
                "description": "Disable global IP forwarding on a router or host.",
                "targets": ["router", "host"],
                "requires": []
            },

            "icmp_ping": {
                "description": "Perform a ping to test connectivity between devices.",
                "targets": ["router", "host"],
                "requires": ["dst_ip"],
                "optional": ["count"]
            },

            "firewall_drop_icmp_src": {
                "description": "Install a firewall rule to drop ICMP traffic coming from a source CIDR.",
                "targets": ["router"],
                "requires": ["src_cidr"]
            },

            "firewall_allow_icmp_src": {
                "description": "Allow ICMP traffic from a source CIDR by removing or overriding drop rules.",
                "targets": ["router"],
                "requires": ["src_cidr"]
            },

            "interface_up": {
                "description": "Bring a network interface up.",
                "targets": ["router", "host", "switch"],
                "requires": ["interface"]
            },

            "interface_down": {
                "description": "Bring a network interface down.",
                "targets": ["router", "host", "switch"],
                "requires": ["interface"]
            },

            "qos_limit_iface": {
                "description": "Apply a bandwidth limit on a network interface.",
                "targets": ["router", "host"],
                "requires": ["iface", "rate_mbit"]
            },

            "ovs_set_controller": {
                "description": "Configure an Open vSwitch bridge to connect to an SDN controller.",
                "targets": ["switch"],
                "requires": ["bridge", "controller_ip", "controller_port"]
            },

            "sdn_flow_add": {
                "description": "Add a flow rule to an SDN switch via the controller.",
                "targets": ["sdn_controller"],
                "requires": ["match", "actions", "priority"]
            },

            "sdn_flow_del": {
                "description": "Remove a flow rule from an SDN switch via the controller.",
                "targets": ["sdn_controller"],
                "requires": ["match"]
            },

            "route_verify": {
                "description": "Verify that a route to a destination subnet exists and is active.",
                "targets": ["router"],
                "requires": ["dst_cidr"]
            },

            "connectivity_verify": {
                "description": "Verify end-to-end connectivity to a destination IP or subnet.",
                "targets": ["router", "host"],
                "requires": ["dst_ip"]
            }
        }

    }
    return state

def build_capability_profile(cmd_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    View compacta para o LLM:
    - mantém somente o que ele precisa para montar steps válidos
    - reduz tamanho do prompt e chance de alucinação/deriva
    """
    supported_ops = cmd_profile.get("supported_ops", {}) or {}

    ops = sorted(supported_ops.keys())

    requires_map: Dict[str, List[str]] = {}
    optional_map: Dict[str, List[str]] = {}
    targets_map: Dict[str, List[str]] = {}

    for op, meta in supported_ops.items():
        req = meta.get("requires", []) or []
        opt = meta.get("optional", []) or []
        tgt = meta.get("targets", []) or []
        requires_map[op] = list(req)
        optional_map[op] = list(opt)
        targets_map[op] = list(tgt)

    # Mantém somente o essencial: listas curtas e fáceis de obedecer
    return {
        "supported_ops": ops,
        "requires": requires_map,
        "optional": optional_map,
        "targets": targets_map,
    }


def node_intent(state: IBNState) -> IBNState:
    text = state["user_intent_text"]
    text_l = (text or "").lower()

    system = """
You are an Intent Classifier for an Intent-Based Networking (IBN) middleware.

You MUST output a TWO-LEVEL classification:
1) "category" (MACRO): one of
- configuration
- monitoring
- control_security
- diagnostic
- removal_cleanup

2) "name" (MICRO OPERATION): one of
- static_route_add
- static_route_del
- ip_forward_enable
- ip_forward_disable
- interface_up
- interface_down
- qos_limit_iface
- icmp_ping
- route_verify
- connectivity_verify
- firewall_drop_icmp_src
- firewall_allow_icmp_src
- ovs_set_controller
- sdn_flow_add
- sdn_flow_del

MAPPING RULES (CRITICAL):
- static_route_add ONLY when the intent mentions a destination network/IP (CIDR or address) AND implies a path/next-hop (e.g., "via", "through", "using router X").
- If the user wants to ADD/CONFIGURE reachability via static route => category="configuration", name="static_route_add"
- If the user wants to REMOVE/DELETE/UNDO a static route => category="removal_cleanup", name="static_route_del"

- ip_forward_enable when the intent asks to enable packet forwarding at OS/kernel level.
  This INCLUDES phrases like:
  "forward packets between interfaces", "act as a router", "enable forwarding",
  even if NO destination network and NO next-hop is provided.
  => category="configuration", name="ip_forward_enable"

- ip_forward_disable ONLY when the user explicitly asks to disable IP forwarding
  => category="removal_cleanup", name="ip_forward_disable"

- If the user asks to ping/test reachability => category="monitoring", name="icmp_ping"
- If the user asks to verify a route exists => category="monitoring", name="route_verify"
- If the user asks to verify connectivity end-to-end => category="monitoring", name="connectivity_verify"
- If the user asks to block ICMP from a source => category="control_security", name="firewall_drop_icmp_src"
- If the user asks to allow/unblock ICMP from a source => category="control_security", name="firewall_allow_icmp_src"
- If the user asks to bring an interface up/down => choose category accordingly, name "interface_up" or "interface_down"
- If the user asks for bandwidth limiting => category="control_security", name="qos_limit_iface"
- If the user asks to set OVS controller => category="configuration", name="ovs_set_controller"
- If the user asks to add/del SDN flows => category="configuration" for add, "removal_cleanup" for del, with names "sdn_flow_add"/"sdn_flow_del"

CRITICAL DISAMBIGUATION:
- If the intent is about forwarding behavior (router acts as router / forward between interfaces)
  and it does NOT mention a destination network/IP, it MUST be ip_forward_enable (NOT static_route_add).

GENERAL DISAMBIGUATION:
- If both add and remove are mentioned, choose the PRIMARY GOAL, and lower confidence if unclear.
- If none fits well, still choose the closest "name", set confidence < 0.5, and explain.

Respond ONLY in valid JSON:
{
  "category": "one of the MACRO categories",
  "name": "one of the MICRO operations",
  "confidence": 0.0,
  "rationale": "short explanation"
}
No markdown, no extra text.
""".strip()

    user = f'Classify the following intent:\n"""{text}"""'

    try:
        resp = llm.invoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        )
        intent_dict = json.loads(resp.content)

        name = (intent_dict.get("name") or "unknown").strip()
        category = (intent_dict.get("category") or "unknown").strip()

        # --- Guardrail barato pós-LLM (sem regex): rota sem destino => não é rota ---
        # Se o classificador insistiu em static_route_add, mas o texto não tem cara de rota,
        # converte para ip_forward_enable quando a frase for sobre "forward between interfaces".
        if name == "static_route_add":
            mentions_forwarding = any(k in text_l for k in [
                "forward packets", "forwarding", "between its interfaces", "between interfaces", "act as a router"
            ])
            mentions_dest = any(k in text_l for k in ["/", "subnet", "network", "via", "through", "next hop", "nexthop"])

            # se parece forwarding e não parece ter destino, corrige
            if mentions_forwarding and not mentions_dest:
                name = "ip_forward_enable"
                category = "configuration"
                intent_dict["rationale"] = (intent_dict.get("rationale") or "").strip()
                if intent_dict["rationale"]:
                    intent_dict["rationale"] += " | "
                intent_dict["rationale"] += "Corrected: forwarding between interfaces without destination implies OS-level IP forwarding."

        # confidence sempre float
        try:
            conf = float(intent_dict.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        state["intent"] = Intent(
            name=name,
            category=category if category else "unknown",
            confidence=conf,
            rationale=(intent_dict.get("rationale") or "")
        )

    except Exception as e:
        state["intent"] = Intent(name="error", category="unknown", confidence=0.0, rationale=str(e))
        state["needs_human"] = True

    return state


# =========================================================
# Índices da topologia: sets para validação rápida
# =========================================================
def _topology_indexes(topo: Dict[str, Any]) -> Dict[str, Set[str]]:
    devices = topo.get("devices", {}) or {}
    dev_names = set(devices.keys())

    # Tenta coletar interfaces (você pode ajustar se seu JSON tiver outro formato)
    ifaces = set()
    for dev, meta in devices.items():
        interfaces_obj = meta.get("interfaces", {}) or {}
        for ifname in interfaces_obj.keys():
            # Se o JSON já guarda "r1-eth0" como key, isso já entra direto.
            ifaces.add(ifname)

            # Se o JSON guarda só "eth0", adiciona também "r1-eth0" como variante.
            if ifname.startswith("eth"):
                ifaces.add(f"{dev}-{ifname}")

    return {"devices": dev_names, "interfaces": ifaces}


# =========================================================
# AJUSTE 1 (Camada B): valida (não só filtra)
# Retorna:
#  - valid: entidades que existem na topologia
#  - unknown: entidades citadas no texto que NÃO existem
# =========================================================
def _validate_against_topology(
    cands: Dict[str, List[str]],
    topo: Dict[str, Any]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:

    idx = _topology_indexes(topo)

    valid = {k: [] for k in cands.keys()}
    unknown = {k: [] for k in cands.keys()}

    # Devices
    for k in ["routers", "switches", "hosts"]:
        for name in cands.get(k, []):
            if not idx["devices"]:
                # Sem topologia: não dá pra validar
                valid[k].append(name)
            elif name in idx["devices"]:
                valid[k].append(name)
            else:
                unknown[k].append(name)

    # Interfaces
    for itf in cands.get("interfaces", []):
        if not idx["interfaces"]:
            valid["interfaces"].append(itf)
        elif itf in idx["interfaces"]:
            valid["interfaces"].append(itf)
        else:
            unknown["interfaces"].append(itf)

    # IPs/CIDRs/Services: normalmente não dá pra validar só com "devices/interfaces"
    # Mantemos como "valid" para serem usados no contexto/plan.
    for k in ["ips", "cidrs", "services"]:
        valid[k] = list(cands.get(k, []))

    # Determinístico
    for k in valid:
        valid[k] = sorted(set(valid[k]))
        unknown[k] = sorted(set(unknown[k]))

    return valid, unknown

def _extract_with_llm_entities_and_selectors(text: str, topo: Dict[str, Any]) -> Dict[str, Any]:
    devices_map = topo.get("devices", {}) or {}
    device_names = sorted(devices_map.keys())

    system = """
    You are a Network Intent Parser. Your goal is to map user natural language into technical entities and selectors.

    CORE LOGIC:
    1. ENTITIES: Extract names of devices (r1, h1), IPs (10.0.0.1), CIDRs, or specific interface names (eth0) EXPLICITLY mentioned.
    2. SELECTORS (GROUPS): Create a selector ONLY if the user refers to a collective set of devices using quantifiers like "all", "every", "any", "each", or "hosts in...".
    3. DISCRIMINATION: 
       - If a device is mentioned by name (e.g., "r2"), it is an ENTITY. 
       - Do NOT create a selector for a specific device's attributes (like "its interfaces" or "its ports") unless those attributes are explicitly named.
       - If the user says "router r2", everything related to r2 should stay in "entities".

    ROLE DEFINITIONS:
    - "sources": The initiator of traffic or the primary subject of configuration.
    - "targets": The destination network, IP, or device.
    - "next_hops": Gateway devices or intermediate nodes.

    HARD RULES:
    - NO PLACEHOLDERS: Never output literals like "<INTERFACE>", "<CIDR>", "<DEVICE_NAME>", or "string".
    - EXPLICIT ONLY: If a specific name or value is not in the text, do not include it in a constraint.
    - SELECTOR VS ENTITY: "Make router r2 forward" -> r2 is an entity. "Make all routers forward" -> selector.
    - If the user refers to "interfaces" of a specific entity without naming them, ignore the interfaces in the JSON and focus on the named entity.

    SCHEMA:
    {
      "entities": {
        "routers": [], "switches": [], "hosts": [], "interfaces": [], "ips": [], "cidrs": [], "services": []
      },
      "selectors": [
        {
          "kind": "device_set",
          "device_type": "host|router|switch|device",
          "quantifier": "all|any|single|each",
          "constraints": [
            {"type": "ip_in_cidr", "cidr": "<CIDR>"},
            {"type": "connected_to", "name": "<DEVICE_NAME>"}
          ],
          "role": "sources|targets|next_hops"
        }
      ]
    }

    STRICT CONDITION: If no group quantifiers (all, every, each, etc.) are used for the main subject, "selectors" MUST be an empty list [].
    Respond ONLY with valid JSON.
    """.strip()

    user = f"""
    IntentText:
    \"\"\"{text}\"\"\"

    """.strip()

    raw = llm.invoke([("system", system), ("user", user)]).content
    data = json.loads(raw)

    ents = data.get("entities") or {}
    selector = data.get("selectors") or []

    out = {
        "entities": {
            "routers": ents.get("routers", []) or [],
            "switches": ents.get("switches", []) or [],
            "hosts": ents.get("hosts", []) or [],
            "interfaces": ents.get("interfaces", []) or [],
            "ips": ents.get("ips", []) or [],
            "cidrs": ents.get("cidrs", []) or [],
            "services": ents.get("services", []) or [],
        },
        "selectors": selector,
    }
    for k in out["entities"]:
        out["entities"][k] = sorted(set(out["entities"][k]))
    return out

import ipaddress

def _iter_device_ips(device_meta: Dict[str, Any]):
    ifaces = device_meta.get("interfaces", {}) or {}
    for _, imeta in ifaces.items():
        for k in ["ip", "ipv4", "address", "addr"]:
            v = imeta.get(k)
            if isinstance(v, str) and v:
                yield v.split("/")[0]
    for k in ["ip", "ipv4"]:
        v = device_meta.get(k)
        if isinstance(v, str) and v:
            yield v.split("/")[0]


def _merge_entities(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {}
    for k in ["routers", "switches", "hosts", "interfaces", "ips", "cidrs", "services"]:
        out[k] = sorted(set((a.get(k) or []) + (b.get(k) or [])))
    return out


def _to_entity_list(ents: Dict[str, List[str]], source: str) -> List[Entity]:
    out: List[Entity] = []
    for r in ents["routers"]:
        out.append(Entity(type="router", value=r, meta={"source": source}))
    for s in ents["switches"]:
        out.append(Entity(type="switch", value=s, meta={"source": source}))
    for h in ents["hosts"]:
        out.append(Entity(type="host", value=h, meta={"source": source}))
    for itf in ents["interfaces"]:
        out.append(Entity(type="interface", value=itf, meta={"source": source}))
    for ip in ents["ips"]:
        out.append(Entity(type="ip", value=ip, meta={"source": source}))
    for cidr in ents["cidrs"]:
        out.append(Entity(type="subnet", value=cidr, meta={"source": source}))
    for svc in ents["services"]:
        out.append(Entity(type="service", value=svc, meta={"source": source}))
    return out


# =========================================================
# AJUSTE 2: construir slice de contexto a partir das entidades válidas
# - Pega o objeto do device na topologia
# - Inclui vizinhos 1-hop via "peer" (quando presente)
# - Também inclui interfaces citadas explicitamente
# =========================================================
def _parse_peer_device(peer_value: Any) -> str:
    """
    Tenta extrair o nome do device a partir de um campo "peer".
    Ex.: "r0-eth1" -> "r0"
         "r0"      -> "r0"
    """
    if not peer_value:
        return ""
    peer = str(peer_value)
    # se vier "r0-eth1" ou "s2-eth3"
    m = re.match(r"^([a-z]\d+)", peer)
    return m.group(1) if m else ""


def build_context_slice(topo: Dict[str, Any], ents_valid: Dict[str, List[str]]) -> Dict[str, Any]:
    devices = topo.get("devices", {}) or {}

    requested_devices = set(ents_valid["routers"] + ents_valid["switches"] + ents_valid["hosts"])
    requested_ifaces = set(ents_valid["interfaces"])

    included_devices = set()
    sliced_devices: Dict[str, Any] = {}

    def include_device(name: str):
        if name in devices and name not in included_devices:
            included_devices.add(name)
            sliced_devices[name] = devices[name]

    # inclui os devices citados
    for d in requested_devices:
        include_device(d)

    # inclui vizinhos 1-hop (se existir campo peer nas interfaces)
    for d in list(included_devices):
        meta = devices.get(d, {}) or {}
        ifaces = meta.get("interfaces", {}) or {}
        for ifname, ifmeta in ifaces.items():
            peer_dev = _parse_peer_device(ifmeta.get("peer"))
            if peer_dev:
                include_device(peer_dev)

    # opcional: se foi citada uma interface "r2-eth0" mas device não foi citado,
    # tenta incluir o device mesmo assim
    for itf in requested_ifaces:
        m = re.match(r"^([a-z]\d+)-", itf)
        if m:
            include_device(m.group(1))

    return {
        "devices": sliced_devices,
        # você pode acrescentar aqui "links" ou "policies" se existirem no seu JSON
    }


def node_entities(state: IBNState) -> IBNState:
    import ipaddress

    text = state["user_intent_text"]
    topo = load_topologia()
    state["topology_full"] = topo

    parsed = _extract_with_llm_entities_and_selectors(text, topo)

    print("--------------------------PARSED--------------------------")
    print(parsed)

    # 1) filtro literal só para nomes de dispositivos
    for cat in ["routers", "switches", "hosts"]:
        if cat in parsed["entities"]:
            parsed["entities"][cat] = [
                val for val in parsed["entities"][cat]
                if isinstance(val, str) and val.lower() in (text or "").lower()
            ]

    ents = parsed["entities"]

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

    # garante chaves
    for k in ["ips", "cidrs", "hosts", "routers", "switches", "interfaces", "services"]:
        ents.setdefault(k, [])

    moved_from_hosts = []
    remaining_hosts = []

    for v in ents["hosts"]:
        if not isinstance(v, str):
            continue
        vv = v.strip()
        if _looks_like_cidr(vv):
            ents["cidrs"].append(vv)
            moved_from_hosts.append(vv)
        elif _looks_like_ip(vv):
            ents["ips"].append(vv)
            moved_from_hosts.append(vv)
        else:
            remaining_hosts.append(vv)

    ents["hosts"] = remaining_hosts

    # 3) Unifica IPs/CIDRs vindos de ips+cidrs (remove duplicata e corrige mistura)
    raw_addr_list = list(ents.get("ips", [])) + list(ents.get("cidrs", []))

    clean_ips = []
    clean_cidrs = []

    for addr in raw_addr_list:
        if not isinstance(addr, str):
            continue
        a = addr.strip()
        if "/" in a:
            # CIDR
            if _looks_like_cidr(a):
                clean_cidrs.append(str(ipaddress.ip_network(a, strict=False)))
        else:
            # IP
            if _looks_like_ip(a):
                clean_ips.append(str(ipaddress.ip_address(a)))

    ents["ips"] = sorted(set(clean_ips))
    ents["cidrs"] = sorted(set(clean_cidrs))

    # (opcional) log pra você ver quando a LLM erra o tipo
    if moved_from_hosts:
        state.setdefault("warnings", [])
        state["warnings"].append(
            f"Normalized entities: moved {moved_from_hosts} from hosts -> ips/cidrs"
        )

    selectors = parsed.get("selectors", [])
    has_host_selector = any(
        (s.get("kind") == "device_set" and (s.get("device_type") or "").lower() == "host")
        for s in (selectors or [])
    )

    valid_ents, unknown_ab = _validate_against_topology(ents, topo)

    unknown_msgs = []
    for k in ["routers", "switches", "hosts", "interfaces"]:
        if unknown_ab.get(k):
            if k == "hosts" and has_host_selector:
                continue  # não trava o pipeline por isso
            unknown_msgs.append(f"Unknown {k} referenced by intent: {', '.join(unknown_ab[k])}")

    if unknown_msgs:
        state.setdefault("warnings", [])
        state["warnings"].extend(unknown_msgs)
        state["needs_human"] = True

    state["entities"] = _to_entity_list(valid_ents, source="LLM_ENTITIES")
    state["entity_selectors"] = selectors

    return state


# ===== NÓ 3: Requisitos (Validar se o pedido faz sentido) =====
def node_requirements(state: IBNState) -> IBNState:
    reqs = []
    ok = True
    if state["intent"].category == "configuration" and state["intent"].name == "static_route_add":
        ok = validate_static_policy(state["topology"])
    state["requirements"] = [{"key": "static_policy_checjk", "ok": ok}]
    return state


def validate_static_policy(topo: Dict) -> bool:
    for name, dev in topo.get("devices", {}).items():
        if dev.get("type") == "router" and name != "r0":
            ifaces = dev.get("interfaces", {})
            if not any("r0" in v.get("peer", "") for v in ifaces.values()):
                return False
    return True

# ===== NÓ 4: Anonimização (Opcional para seu teste) =====
def node_anonymize(state: IBNState) -> IBNState:
    # Apenas passa adiante para não complicar o teste agora
    state["anonymization_map"] = {}
    return state

def _resolve_selectors_against_topology(selectors: List[Dict[str, Any]], topo: Dict[str, Any]) -> Dict[str, List[str]]:
    devices = topo.get("devices", {}) or {}

    def all_of_type(dtype: str) -> List[str]:
        if dtype in ("device", "any", "unknown"):
            return list(devices.keys())
        return [n for n, m in devices.items() if (m.get("type") == dtype)]

    def connected_to(name: str) -> Set[str]:
        out = set()
        for dev, meta in devices.items():
            ifaces = meta.get("interfaces", {}) or {}
            for _, imeta in ifaces.items():
                peer = imeta.get("peer")
                if isinstance(peer, str) and name in peer:
                    out.add(dev)
        return out

    resolved = {"routers": [], "switches": [], "hosts": [], "interfaces": [], "ips": [], "cidrs": [], "services": []}

    for sel in selectors or []:
        if sel.get("kind") != "device_set":
            continue

        dtype = (sel.get("device_type") or "device").strip().lower()
        quant = (sel.get("quantifier") or "unknown").strip().lower()
        constraints = sel.get("constraints") or []

        candidate = set(all_of_type(dtype))

        for c in constraints:
            ctype = (c.get("type") or "").strip().lower()

            if ctype == "ip_in_cidr":
                cidr = c.get("cidr")

                # ✅ sanitização barata (sem regex): precisa ser string e ser um CIDR parseável
                if not isinstance(cidr, str) or not cidr:
                    continue

                try:
                    net = ipaddress.ip_network(cidr, strict=False)
                except ValueError:
                    # Ignora constraint inválido ao invés de quebrar o programa
                    # (se quiser “subir” isso pra warnings/needs_human, veja a nota abaixo)
                    continue

                keep = set()
                for dev in candidate:
                    meta = devices.get(dev) or {}
                    for ip_str in _iter_device_ips(meta):
                        try:
                            ip = ipaddress.ip_address(ip_str)
                        except ValueError:
                            continue
                        if ip in net:
                            keep.add(dev)
                            break
                candidate = keep

            elif ctype == "name_in_topology":
                name = c.get("name")
                if isinstance(name, str) and name:
                    candidate = candidate.intersection({name})

            elif ctype == "connected_to":
                name = c.get("name")
                if isinstance(name, str) and name:
                    candidate = candidate.intersection(connected_to(name))

        chosen = sorted(candidate)
        if quant == "single" and chosen:
            chosen = chosen[:1]

        if dtype == "host":
            resolved["hosts"].extend(chosen)
        elif dtype == "router":
            resolved["routers"].extend(chosen)
        elif dtype == "switch":
            resolved["switches"].extend(chosen)
        else:
            for dev in chosen:
                t = (devices.get(dev) or {}).get("type")
                if t == "host":
                    resolved["hosts"].append(dev)
                elif t == "router":
                    resolved["routers"].append(dev)
                elif t == "switch":
                    resolved["switches"].append(dev)

    for k in ["routers", "switches", "hosts"]:
        resolved[k] = sorted(set(resolved[k]))
    return resolved

def _resolve_selectors_against_topology_by_role(selectors, topo):
    def _empty_bucket():
        return {"routers": [], "switches": [], "hosts": [], "interfaces": [], "ips": [], "cidrs": [], "services": []}

    out = {
        "sources": _empty_bucket(),
        "targets": _empty_bucket(),
        "next_hops": _empty_bucket(),
        "unknown": _empty_bucket(),
    }

    for sel in selectors or []:
        role = (sel.get("role") or "unknown").strip().lower()
        if role not in out:
            role = "unknown"

        resolved_one = _resolve_selectors_against_topology([sel], topo)
        for k in out[role].keys():
            out[role][k].extend(resolved_one.get(k, []) or [])

    for role, bucket in out.items():
        for k in bucket.keys():
            bucket[k] = sorted(set(bucket[k]))

    return out


def node_context(state: IBNState) -> IBNState:

    topo = state.get("topology_full") or {}
    selectors = state.get("entity_selectors") or []

    extracted_entities = state.get("entities") or []


    print("------------------------Entidades------------------------")
    print(extracted_entities)

    ents_valid = {
        "routers": [],
        "switches": [],
        "hosts": [],
        "interfaces": [],
        "ips": [],
        "cidrs": [],
        "services": [],
    }

    type_map = {
        "router": "routers",
        "switch": "switches",
        "host": "hosts",
        "interface": "interfaces",
        "ip": "ips",
        "subnet": "cidrs",
        "cidr": "cidrs",
        "service": "services",
    }

    for ent in extracted_entities:
        if ent is None:
            continue

        etype = (ent.type if hasattr(ent, "type") else ent.get("type")).lower()
        val = (ent.value if hasattr(ent, "value") else ent.get("value"))

        bucket = type_map.get(etype)
        if bucket and val:
            ents_valid[bucket].append(val)

    # dedup inicial
    for k in ents_valid:
        ents_valid[k] = sorted(set(ents_valid[k]))

    # 2️⃣ Resolve selectors POR ROLE
    resolved_by_role = _resolve_selectors_against_topology_by_role(selectors, topo)

    # 3️⃣ Guarda candidatos de next-hop (não entram no slice)
    state["selector_next_hops_candidates"] = resolved_by_role.get("next_hops", {})

    scoped = _merge_entities(
        resolved_by_role.get("sources", {}),
        resolved_by_role.get("targets", {}),
    )

    for k in ents_valid:
        ents_valid[k] = sorted(set(ents_valid[k] + scoped.get(k, [])))

    if not any(ents_valid[k] for k in ents_valid):
        state.setdefault("warnings", [])
        state["warnings"].append(
            "No entities to slice topology. Refusing to send full topology. Human input required."
        )
        state["needs_human"] = True
        state["topology"] = {"devices": {}}
        state["topology_slice_entities"] = ents_valid
        
        return state

    state["topology"] = build_context_slice(topo, ents_valid)
    state["topology_slice_entities"] = ents_valid

    return state

# ===== NÓ 6: Planejamento (A LLM gera os comandos aqui!) =====

import re
from typing import Any, Dict, List, Optional, Set

CIDR_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")

def entities_to_jsonable(entities: List[Any]) -> List[dict]:
    out = []
    for e in entities or []:
        if isinstance(e, dict):
            out.append(e)
        else:
            out.append({"type": e.type, "value": e.value, "meta": getattr(e, "meta", {})})
    return out

def extract_fields_from_entities(entities: List[dict]) -> Dict[str, Optional[str]]:
    routers = [e["value"] for e in entities if e.get("type") == "router"]

    dst_cidr = None
    dst_ip = None

    for e in entities:
        if e.get("type") in ("ip", "subnet"):
            v = (e.get("value") or "").strip()

            # CIDR
            m = CIDR_RE.search(v)
            if m and not dst_cidr:
                dst_cidr = m.group(0)
                continue

            # IP puro (sem /)
            # (aceita "10.4.0.1" e também "10.4.0.1/24" -> cai no CIDR acima)
            if "/" not in v and not dst_ip:
                dst_ip = v

    return {
        "routers": routers,
        "dst_cidr": dst_cidr,
        "dst_ip": dst_ip,
    }


def router_neighbors_from_interfaces(topology: dict, router: str) -> List[str]:
    devices = topology.get("devices", {})
    d = devices.get(router, {})
    if d.get("type") != "router":
        return []

    neigh = []
    for ifname, iface in (d.get("interfaces") or {}).items():
        peer = (iface or {}).get("peer")
        if not peer or "-" not in peer:
            continue
        peer_dev = peer.split("-", 1)[0]
        if devices.get(peer_dev, {}).get("type") == "router":
            neigh.append(peer_dev)

    # unique preserving order
    out = []
    seen: Set[str] = set()
    for x in neigh:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def find_gateway_device_for_cidr(topology: dict, cidr: str) -> Optional[str]:
    # usa a tabela networks (bem confiável no seu JSON)
    for net in (topology.get("networks") or {}).values():
        if net.get("cidr") == cidr:
            return net.get("gateway_device")
    return None

def build_planner_context(topology: dict, entities: List[Any], intent_text: str) -> dict:
    ent = entities_to_jsonable(entities)
    fields = extract_fields_from_entities(ent)

    routers = fields["routers"]
    dst_cidr = fields["dst_cidr"]
    dst_ip = fields["dst_ip"]



    # Heurística mínima: se tiver 2 routers, assume [src, via] pela ordem do intent
    # (ideal: pegar src_router do selector/intent parser)
    src_router = routers[1] if len(routers) >= 2 else (routers[0] if routers else None)
    via_router = routers[0] if len(routers) >= 2 else None

    allowed = router_neighbors_from_interfaces(topology, src_router) if src_router else []
    if via_router:
        allowed = [via_router]  # trava

    dst_gateway = find_gateway_device_for_cidr(topology, dst_cidr) if dst_cidr else None

    # contexto mínimo e “seguro”
    return {
        "facts": {
            "intent": intent_text,
            "src_router": src_router,
            "dst_cidr": dst_cidr,
            "dst_ip": dst_ip,
            "via_router": via_router,
            "dst_gateway_device": dst_gateway,
            "allowed_next_hops": allowed,
        }
    }

def node_plan(state: IBNState) -> IBNState:
    # topo_context = state.get("topology", {"devices": {}})
    intent_obj = state.get("intent", {})
    if hasattr(intent_obj, "model_dump"):
        intent_obj = intent_obj.model_dump()
    elif hasattr(intent_obj, "dict"):
        intent_obj = intent_obj.dict()

    intent_macro = (intent_obj.get("category") or "").strip().lower()  # configuration/monitoring/etc
    intent_op    = (intent_obj.get("name") or "").strip().lower()      # static_route_add/etc


    with open("fewshots/planning_fewshots.json") as f:
        FEWSHOTS = json.load(f)["planning_fewshots"]

    topology = state.get("topology", {})
    entities  = state.get("entities", [])
    
    intent_text = state.get("user_intent_text", "")

    cmd_profile = state.get("command_profile", {})

    supported_ops = sorted((cmd_profile.get("supported_ops") or {}).keys())
    allowed_ops_str = "|".join(supported_ops)

    cap_profile = build_capability_profile(cmd_profile)

    topo_context = build_planner_context(
        topology=topology,
        entities=entities,
        intent_text=intent_text
    )

    if not topo_context:
        state["plan"] = ExecPlan(
            steps=[],
            warnings=["Empty sliced topology; refusing to plan."],
            needs_human=True,
            dry_run=True
        )
        return state

    BASE_SYSTEM_PROMPT = """
        You are an Intent-Based Networking (IBN) assistant.
        Goal:
            You are an IBN Middleware Assistant. Your job is to decompose a network Intent into a sequence of atomic steps.
        
        Rules:
        - DISCRETIZATION: Every configuration MUST be its own step. 
        Example: If adding a route with a rule, Step 1 is the route, Step 2 is the rule.
        - ATOMICITY: One 'op' per step. One 'device' per step.
        - Respond STRICTLY in valid JSON — no text, explanations, or comments outside the JSON block.
        - Do NOT invent devices, interfaces, IPs, subnets, bridges, or next-hops.
            You may only use values that appear in NetworkContext (or explicitly in the Intent object).

        - Do NOT output CLI commands (no vtysh, ip, ovs-vsctl, tc, iptables, etc.).

        - Each step MUST be machine-actionable using:
            - "device": target node name
            - "op": MUST be EXACTLY one of: [__ALLOWED_OPS__]
            - "args": a JSON object with parameters (only from context/intent)

        - If any essential data is missing, ambiguous, or could affect operational safety, 
        set "needs_human": true and include a clear warning message.
        - Always prefer safety and human oversight over automation.

        CLOSED-WORLD CONSTRAINT (MANDATORY):
        - You MUST treat NetworkContext as a CLOSED WORLD.
        - step.op MUST be one of CapabilityProfile.supported_ops.
        - step.args MUST use ONLY keys allowed by CapabilityProfile.requires[step.op] and CapabilityProfile.optional[step.op].
        - Every VALUE inside args (e.g., "r1", "10.3.0.0/24") MUST appear verbatim in NetworkContext.facts.
        - If a required value is not present in NetworkContext.facts, you MUST NOT invent it; instead set "needs_human": true and explain what is missing.
        - If any step would require touching a device not listed in NetworkContext.facts.src_router, you MUST set "needs_human": true.

            
        Expected JSON output:
            "Plan": {
                "steps": [
                    {
                        "device": "string",
                        "op": "ONE OF: [__ALLOWED_OPS__]",
                        "args": { "key": "value" },
                    }
                ],
                "warnings": ["string"],
                "needs_human": false
            }	

    """

    BASE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT.replace("__ALLOWED_OPS__", allowed_ops_str)


    STATIC_ROUTE_ADD_BLOCK = """
        SPECIALIZATION: static_route_add
        
        OPERATOR RULES (CRITICAL):
            1. SINGLE DEVICE FOCUS: Generate steps ONLY for the router explicitly mentioned in the Intent (e.g., if the Intent says "on router r2", ONLY create steps where "device" is "r2").
            2. NO REDUNDANCY: Do not attempt to configure neighbor routers or transit routers unless explicitly asked.
            3. SCOPE: Your task is finished as soon as the mentioned router has a pointer (exit_gateway) to the next hop.
            4. DATA INTEGRITY: Every step must have 'target_network' and 'exit_gateway' using names found in the NetworkContext peers.
        Every step MUST have 'target_network' (e.g., 10.3.0.0/24) AND 'exit_gateway'.
        Never use a switch name (e.g., s1, s2) as 'exit_gateway', even if it is a peer. Find the ROUTER behind it.
        'exit_gateway' MUST be the NAME of the neighbor router

        If the target network is directly connected to a neighbor, only one step is needed.
    """

    STATIC_ROUTE_DEL_BLOCK = """
        SPECIALIZATION: static_route_del

        MANDATORY RULES:
        - Generate EXACTLY 1 step.
        - step.device MUST equal NetworkContext.facts.src_router
        - step.op MUST be "static_route_del"
        - step.args MUST be exactly: {"dst_cidr": NetworkContext.facts.dst_cidr}
        - Do NOT add verify steps or any extra cleanup operations.
    """.strip()

    IP_FORWARD_ENABLE_BLOCK = """
        SPECIALIZATION: ip_forward_enable

        MANDATORY RULES:
        - Generate EXACTLY 1 step.
        - step.device MUST equal NetworkContext.facts.src_router
        - step.op MUST be "ip_forward_enable"
        - step.args MUST be {}
    """.strip()

    ICMP_PING_BLOCK = """
    SPECIALIZATION: icmp_ping

    MANDATORY RULES:
    - Generate EXACTLY 1 step.
    - step.device MUST equal NetworkContext.facts.src_router.
    - step.op MUST be "icmp_ping".
    - step.args MUST be exactly:
    {"dst_ip": NetworkContext.facts.dst_ip}
    OR {"dst_ip": NetworkContext.facts.dst_ip, "count": <small int>}
    - NEVER generate static routes, firewall rules, interface changes, or any other ops for a ping intent.
    - If dst_ip is missing in NetworkContext.facts, set needs_human=true.
    """.strip()

    FIREWALL_DROP_ICMP_BLOCK = """
        SPECIALIZATION: firewall_drop_icmp_src

        MANDATORY RULES:
        - Generate EXACTLY 1 step.
        - step.device MUST be the router where the security policy is applied (facts.src_router).
        - step.op MUST be "firewall_drop_icmp_src".
        - step.args MUST include "src_cidr".
        - The "src_cidr" MUST be a valid subnet string from NetworkContext.
    """.strip()

    INTERFACE_UP_DOWN_BLOCK = """
        SPECIALIZATION: interface_up / interface_down

        MANDATORY RULES:
        - step.device MUST be the device owner of the interface.
        - step.op MUST be "interface_up" or "interface_down" as specified in the intent.
        - step.args MUST include "interface" name (e.g., "eth0").
        - If the interface name is not explicit, use NetworkContext to find the interface connecting to the mentioned peer.
    """.strip()

    QOS_LIMIT_BLOCK = """
        SPECIALIZATION: qos_limit_iface

        MANDATORY RULES:
        - step.device MUST be the device where the limit is applied.
        - step.op MUST be "qos_limit_iface".
        - step.args MUST include "iface" and "rate_mbit".
        - If "rate_mbit" is not specified in the Intent, set "needs_human": true and ask for the bandwidth limit.
    """.strip()

    VERIFY_BLOCK = """
        SPECIALIZATION: connectivity_verify / route_verify

        MANDATORY RULES:
        - This is a monitoring task.
        - step.op MUST be "connectivity_verify" (for end-to-end) or "route_verify" (for table check).
        - step.args MUST include "dst_ip" (for connectivity) or "dst_cidr" (for route).
        - Generate steps ONLY for the device performing the verification.
    """.strip()

    
    TASK_BY_CAT = {
        "static_route_add": STATIC_ROUTE_ADD_BLOCK,
        "static_route_del": STATIC_ROUTE_DEL_BLOCK,
        "ip_forward_enable": IP_FORWARD_ENABLE_BLOCK,
        "icmp_ping": ICMP_PING_BLOCK,
        "firewall_drop_icmp_src": FIREWALL_DROP_ICMP_BLOCK,
        "firewall_allow_icmp_src": FIREWALL_DROP_ICMP_BLOCK.replace("drop", "allow"),
        "interface_up": INTERFACE_UP_DOWN_BLOCK,
        "interface_down": INTERFACE_UP_DOWN_BLOCK,
        "qos_limit_iface": QOS_LIMIT_BLOCK,
        "connectivity_verify": VERIFY_BLOCK,
        "route_verify": VERIFY_BLOCK
    }

    task_block = TASK_BY_CAT.get(intent_op)

    def render_planning_fewshots(fewshots: list[dict]) -> str:
        blocks = []
        for fs in fewshots:
            blocks.append(
                "FACTS:\n"
                f"{json.dumps(fs['facts'], indent=2)}\n\n"
                "OUTPUT:\n"
                f"{json.dumps(fs['output'], indent=2)}"
            )
        return "\n\n---\n\n".join(blocks)
    
    fewshots_filtered = [
        fs for fs in (FEWSHOTS or [])
        if (fs.get("category") or "").strip().lower() == intent_op

    ]

    # 2) Se não tiver nenhum, não passa exemplos (ou cai num fallback)
    MAX_FEWSHOTS = 1  # recomendo 1 pro llama3.2:3b
    fewshots_filtered = fewshots_filtered[:MAX_FEWSHOTS]

    fewshots_text = render_planning_fewshots(fewshots_filtered) if fewshots_filtered else ""
    

    user = f"""
        Below is a generic example of how to map facts into a plan.
        The example uses symbolic placeholders. Do NOT reuse the literal names.
        Always use ONLY values from NetworkContext.facts.

        EXAMPLE:
        {fewshots_text}

        ----

        NOW SOLVE THE REAL TASK BELOW.

        Intent:
        {json.dumps(intent_obj, indent=2)}

        CapabilityProfile:
        {json.dumps(cap_profile, indent=2)}

        NetworkContext:
        {json.dumps(topo_context, indent=2)}

        TASK: {task_block}
    """

    try:


        resp = llm.invoke([{"role": "system", "content": BASE_SYSTEM_PROMPT},
                            {"role": "user", "content": user}])
        raw = resp.content

        print(f"DEBUG FULL LLM RESPONSE: {raw}") # Adicione isso

        match = re.search(r"\{.*\}\s*$", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)

        plan_data = data.get("Plan", data)

        steps = plan_data.get("steps", []) or []
        warnings = plan_data.get("warnings", []) or []
        needs_human = bool(plan_data.get("needs_human", False))

        needs_human = bool(needs_human)

        state["plan"] = ExecPlan(
            steps=steps,
            warnings=warnings,
            needs_human=needs_human,
            dry_run=True
        )

    except Exception as e:
        state["plan"] = ExecPlan(steps=[], warnings=[f"LLM error: {e}"], needs_human=True)
    return state

def node_generate_cli(state: IBNState) -> IBNState:
    import json
    import ipaddress
    from copy import deepcopy

    plan = state.get("plan")
    profile = state.get("command_profile", {})
    topology = state.get("topology", {})

    if not plan or plan.needs_human:
        return state

    # ---------------------------------------------------------------------
    # Helpers: resolução determinística de next-hop (portável, sem regex)
    # ---------------------------------------------------------------------
    def _is_ip(s: str) -> bool:
        try:
            ipaddress.ip_address(s)
            return True
        except Exception:
            return False

    def _peer_device(peer: str):
        # aceita "r0-eth1" ou "r0"
        if not isinstance(peer, str) or not peer:
            return None
        return peer.split("-", 1)[0]

    def resolve_next_hop_ip(device: str, gateway: str, topo: dict):
        """
        Retorna o IP da interface do gateway que está ligada diretamente ao device.
        Exige topo['devices'][dev]['interfaces'][if]['peer'/'ip'].
        """
        devices = (topo or {}).get("devices", {})
        dev_intfs = (devices.get(device, {}) or {}).get("interfaces", {}) or {}

        # acha uma interface do device cujo peer aponta para gateway
        peer_value = None
        for ifname, meta in dev_intfs.items():
            peer = (meta or {}).get("peer")
            if _peer_device(peer) == gateway:
                peer_value = peer  # ex: "r0-eth1" ou "r0"
                break

        if not peer_value:
            return None

        gw_intfs = (devices.get(gateway, {}) or {}).get("interfaces", {}) or {}

        # Caso 1: peer inclui interface do gateway (ex: "r0-eth1")
        if isinstance(peer_value, str) and "-" in peer_value:
            ip = (gw_intfs.get(peer_value, {}) or {}).get("ip")
            return ip if isinstance(ip, str) and _is_ip(ip) else None

        # Caso 2: peer veio só como "r0" -> procura no gateway uma interface cujo peerDevice == device
        for gw_if, gw_meta in gw_intfs.items():
            if _peer_device((gw_meta or {}).get("peer")) == device:
                ip = (gw_meta or {}).get("ip")
                return ip if isinstance(ip, str) and _is_ip(ip) else None

        return None

    def normalize_static_route_step(step: dict, topo: dict) -> dict:
        """
        Para static_route_add, garante args.exit_gateway_ip a partir de args.exit_gateway.
        Não altera o step original (cópia).
        """
        step = deepcopy(step)
        args = step.get("args") or {}
        dev = step.get("device")
        gw = args.get("exit_gateway")

        if not dev or not gw:
            return step

        if args.get("exit_gateway_ip"):
            return step

        # Gateway literal como IP
        if isinstance(gw, str) and _is_ip(gw):
            args["exit_gateway_ip"] = gw
            step["args"] = args
            return step

        # Gateway como nome de device (ex: r0)
        if isinstance(gw, str):
            nh_ip = resolve_next_hop_ip(dev, gw, topo)
            if nh_ip:
                args["exit_gateway_ip"] = nh_ip
                step["args"] = args

        return step

    # ---------------------------------------------------------------------
    # Carrega templates
    # ---------------------------------------------------------------------
    with open("fewshots/cli_templates.json", "r") as f:
        templates = json.load(f)

    # ---------------------------------------------------------------------
    # Normaliza o plan ANTES de enviar à LLM (ex: exit_gateway -> exit_gateway_ip)
    # ---------------------------------------------------------------------
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
    steps = plan_dict.get("steps", []) or []
    normalized_steps = []

    for s in steps:
        if isinstance(s, dict) and s.get("op") == "static_route_add":
            normalized_steps.append(normalize_static_route_step(s, topology))
        else:
            normalized_steps.append(s)

    plan_dict["steps"] = normalized_steps

    # Lista de devices permitidos (evita a LLM inventar keys extras como "r0")
    plan_devices = sorted({
        s.get("device")
        for s in plan_dict.get("steps", [])
        if isinstance(s, dict) and s.get("device")
    })

    # (Opcional, mas recomendado) Fail-fast se static_route_add não tiver exit_gateway_ip
    for s in plan_dict.get("steps", []):
        if isinstance(s, dict) and s.get("op") == "static_route_add":
            args = s.get("args") or {}
            if "exit_gateway_ip" not in args:
                state["cli_commands"] = {
                    "status": "CLI_FAILED",
                    "error": f"Unable to resolve exit_gateway_ip for device={s.get('device')} exit_gateway={args.get('exit_gateway')}",
                    "raw": json.dumps(plan_dict, indent=2)[:1000],
                }
                state["needs_human"] = True
                return state

    # ---------------------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------------------
    system_prompt = f"""
        You are a Network CLI Generator.
        Translate the Abstract Plan steps into concrete commands using the provided Syntax Templates and Topology.

        SYNTAX TEMPLATES:
        {json.dumps(templates, indent=2)}

        TOPOLOGY CONTEXT:
        {json.dumps(topology, indent=2)}

        RULES:
        1. For each step in the plan, find the matching 'op' in the Templates.
        2. Replace placeholders (like {{interface}} or {{dst_ip}}) with real values from the plan's 'args' or Topology.
        3. If the device stack is 'linux+frr', prioritize 'vtysh' for routing and 'ip/sysctl' for system.
        4. Return ONLY a JSON object mapping EACH device name from the plan to a list of commands.
           - Output keys MUST be exactly the allowed devices.
           - Do NOT include any device not present in the plan.
           - Do NOT add helper steps unless explicitly present in the plan.
        5. Do not use backslash escapes like \'. If you need quotes inside a command, use single quotes ' WITHOUT escaping.
    """

    user_content = (
        f"Allowed devices (MUST be the ONLY JSON keys): {plan_devices}\n"
        f"Abstract Plan to translate:\n{json.dumps(plan_dict, indent=2)}"
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        raw = (response.content or "").strip()

        i = raw.find("{")
        j = raw.rfind("}")
        if i == -1 or j == -1 or j <= i:
            raise ValueError("LLM did not return a JSON object.")

        blob = raw[i:j + 1]
        blob = blob.replace("\\'", "'")  # tolera escape inválido

        commands_json = json.loads(blob)

        # valida chaves = devices do plano (normalizado)
        if plan_devices and set(commands_json.keys()) != set(plan_devices):
            raise ValueError(f"CLI keys must match plan devices {set(plan_devices)}, got {set(commands_json.keys())}")


        state["cli_commands"] = {
            "status": "CLI_GENERATED",
            "commands": commands_json
        }

    except Exception as e:
        state["cli_commands"] = {
            "status": "CLI_FAILED",
            "error": str(e),
            "raw": (raw[:1000] if 'raw' in locals() else "")
        }
        state["needs_human"] = True

    return state


# ===== NÓS DE FINALIZAÇÃO (Apenas para fechar o grafo) =====
def node_decide_exec(state: IBNState) -> IBNState:
    return state

def node_execute(state: IBNState) -> IBNState:
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
    state["verification"] = {"status": "verified_by_simulation"}
    return state
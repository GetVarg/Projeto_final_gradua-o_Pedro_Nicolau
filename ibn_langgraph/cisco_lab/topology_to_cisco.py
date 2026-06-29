#!/usr/bin/env python3
import argparse
import ipaddress
import json
from pathlib import Path


DEFAULT_TOPOLOGY = Path("dataset/topologias_convertidas/gabriel/10/0.json")
DEFAULT_OUTPUT_DIR = Path("cisco_lab/generated")


def load_topology(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        topology = json.load(fp)

    if not isinstance(topology.get("devices"), dict):
        raise ValueError("Topology JSON must contain a 'devices' object.")

    return topology


def router_hostname(device_id: str) -> str:
    return f"R{device_id}"


def parse_iface_name(iface_name: str) -> tuple[str, str]:
    device_id, _sep, suffix = iface_name.partition("-")
    return device_id, suffix


def ios_interface_name(iface_name: str, interface_prefix: str) -> str:
    _device_id, suffix = parse_iface_name(iface_name)
    if suffix.startswith("eth") and suffix[3:].isdigit():
        return f"{interface_prefix}{suffix[3:]}"
    return iface_name


def ip_mask(cidr: str) -> str:
    return str(ipaddress.ip_network(cidr, strict=False).netmask)


def wildcard_mask(cidr: str) -> str:
    network = ipaddress.ip_network(cidr, strict=False)
    wildcard = int(network.hostmask)
    return str(ipaddress.IPv4Address(wildcard))


def network_address(cidr: str) -> str:
    return str(ipaddress.ip_network(cidr, strict=False).network_address)


def is_ipv4_cidr(cidr: str) -> bool:
    return ipaddress.ip_network(cidr, strict=False).version == 4


def interface_description(iface_data: dict) -> str:
    peer = iface_data.get("peer")
    return f"connected_to {peer}" if peer else "generated_from_topology"


def router_config(device_id: str, device_data: dict, interface_prefix: str, ospf_process: int) -> str:
    hostname = router_hostname(device_id)
    lines = [
        "version 15.2",
        f"hostname {hostname}",
        "no ip domain-lookup",
        "ip cef",
        "ipv6 unicast-routing",
        "!",
    ]

    interfaces = device_data.get("interfaces", {})
    for iface_name in sorted(interfaces):
        iface_data = interfaces[iface_name]
        ios_name = ios_interface_name(iface_name, interface_prefix)
        cidr = iface_data.get("cidr")
        ip_addr = iface_data.get("ip")
        lines.extend(
            [
                f"interface {ios_name}",
                f" description {interface_description(iface_data)}",
            ]
        )
        if ip_addr and cidr and is_ipv4_cidr(cidr):
            lines.append(f" ip address {ip_addr} {ip_mask(cidr)}")
        lines.extend([" no shutdown", "!"])

    networks = sorted(
        {
            iface_data["cidr"]
            for iface_data in interfaces.values()
            if iface_data.get("cidr") and is_ipv4_cidr(iface_data["cidr"])
        },
        key=lambda value: ipaddress.ip_network(value, strict=False),
    )
    if networks:
        router_id = f"10.255.0.{int(device_id) + 1}" if device_id.isdigit() else "10.255.0.1"
        lines.extend(
            [
                f"router ospf {ospf_process}",
                f" router-id {router_id}",
            ]
        )
        for cidr in networks:
            lines.append(f" network {network_address(cidr)} {wildcard_mask(cidr)} area 0")
        lines.append("!")

    lines.extend(["end", ""])
    return "\n".join(lines)


def build_topology_map(topology: dict, interface_prefix: str) -> dict:
    routers = {}
    hosts = {}

    for device_id, device_data in topology["devices"].items():
        if device_data.get("type") == "router":
            routers[device_id] = {
                "hostname": router_hostname(device_id),
                "interfaces": {
                    iface_name: {
                        "ios_interface": ios_interface_name(iface_name, interface_prefix),
                        "ip": iface_data.get("ip"),
                        "cidr": iface_data.get("cidr"),
                        "peer": iface_data.get("peer"),
                    }
                    for iface_name, iface_data in sorted(device_data.get("interfaces", {}).items())
                },
            }
        elif device_data.get("type") == "host":
            hosts[device_id] = {
                "interfaces": device_data.get("interfaces", {}),
            }

    return {
        "routers": routers,
        "hosts": hosts,
        "interface_prefix": interface_prefix,
        "notes": [
            "Interface names are mapped from dataset names such as 0-eth2 to Cisco names such as GigabitEthernet0/2.",
            "Management IPs are not present in the topology JSON; fill cisco_lab/inventory.json after importing the configs into your emulator.",
        ],
    }


def build_inventory_skeleton(topology: dict) -> dict:
    devices = {}
    for device_id, device_data in topology["devices"].items():
        if device_data.get("type") != "router":
            continue
        devices[device_id] = {
            "hostname": router_hostname(device_id),
            "protocol": "telnet",
            "host": "127.0.0.1",
            "port": None,
            "username": "",
            "password": "",
            "enable_password": "",
        }

    return {
        "defaults": {
            "protocol": "telnet",
            "username": "",
            "password": "",
            "enable_password": "",
            "command_timeout": 10,
        },
        "devices": devices,
    }


def write_outputs(topology: dict, output_dir: Path, interface_prefix: str, ospf_process: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    for device_id, device_data in topology["devices"].items():
        if device_data.get("type") != "router":
            continue
        config = router_config(device_id, device_data, interface_prefix, ospf_process)
        (configs_dir / f"{router_hostname(device_id)}.cfg").write_text(config, encoding="utf-8")

    topology_map = build_topology_map(topology, interface_prefix)
    (output_dir / "topology_map.json").write_text(
        json.dumps(topology_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inventory = build_inventory_skeleton(topology)
    inventory_path = output_dir / "inventory.example.json"
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Cisco IOS initial router configs from a converted topology JSON."
    )
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interface-prefix", default="GigabitEthernet0/")
    parser.add_argument("--ospf-process", type=int, default=1)
    args = parser.parse_args()

    topology = load_topology(args.topology)
    write_outputs(topology, args.output_dir, args.interface_prefix, args.ospf_process)
    print(f"Generated Cisco configs in {args.output_dir / 'configs'}")
    print(f"Generated topology map at {args.output_dir / 'topology_map.json'}")
    print(f"Generated inventory skeleton at {args.output_dir / 'inventory.example.json'}")


if __name__ == "__main__":
    main()

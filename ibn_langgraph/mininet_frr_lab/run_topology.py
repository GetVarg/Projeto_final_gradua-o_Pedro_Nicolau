#!/usr/bin/env python3
import argparse
import ipaddress
import json
import os
import shutil
import time
from pathlib import Path

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import Node
from mininet.topo import Topo


FRR_BASE_DIR = Path("/tmp/tcc-mininet-frr")
ZEBRA_BIN = "/usr/lib/frr/zebra"
OSPFD_BIN = "/usr/lib/frr/ospfd"
STATICD_BIN = "/usr/lib/frr/staticd"
BGPD_BIN = "/usr/lib/frr/bgpd"


class LinuxRouter(Node):
    def config(self, **params):
        super().config(**params)
        self.cmd("sysctl -w net.ipv4.ip_forward=1")
        self.cmd("sysctl -w net.ipv6.conf.all.forwarding=1")
        self.cmd("sysctl -w net.ipv4.conf.all.rp_filter=0")

    def terminate(self):
        self.cmd("pkill -f '/usr/lib/frr/(zebra|ospfd|staticd|bgpd)'")
        super().terminate()


def router_name(device_id: str) -> str:
    return f"r{device_id}"


def prefix_len(cidr: str) -> int:
    return ipaddress.ip_network(cidr, strict=False).prefixlen


def interface_ip_with_prefix(iface_data: dict) -> str:
    return f"{iface_data['ip']}/{prefix_len(iface_data['cidr'])}"


def parse_iface_name(iface_name: str) -> tuple[str, str]:
    device_id, _sep, _suffix = iface_name.partition("-")
    return device_id, iface_name


def load_topology(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        topology = json.load(fp)

    if not isinstance(topology.get("devices"), dict):
        raise ValueError("Topology JSON must contain a 'devices' object.")

    return topology


class JsonTopology(Topo):
    def build(self, topology: dict):
        self.topology = topology
        devices = topology["devices"]

        for device_id, device_data in devices.items():
            if device_data.get("type") == "router":
                self.addNode(router_name(device_id), cls=LinuxRouter, dataset_id=device_id)
            elif device_data.get("type") == "host":
                self.addHost(device_id)
            else:
                raise ValueError(f"Unsupported device type for {device_id}: {device_data.get('type')}")

        added_links = set()
        for device_id, device_data in devices.items():
            for iface_name, iface_data in device_data.get("interfaces", {}).items():
                peer_iface = iface_data.get("peer")
                if not peer_iface:
                    continue

                peer_device_id, peer_iface_name = parse_iface_name(peer_iface)
                link_key = tuple(sorted([iface_name, peer_iface_name]))
                if link_key in added_links:
                    continue

                if peer_device_id not in devices:
                    raise ValueError(f"Peer device {peer_device_id} referenced by {iface_name} does not exist.")

                node_a = router_name(device_id) if devices[device_id].get("type") == "router" else device_id
                node_b = router_name(peer_device_id) if devices[peer_device_id].get("type") == "router" else peer_device_id

                self.addLink(
                    node_a,
                    node_b,
                    cls=TCLink,
                    intfName1=iface_name,
                    intfName2=peer_iface_name,
                )
                added_links.add(link_key)


def configure_interfaces(net: Mininet, topology: dict) -> None:
    for device_id, device_data in topology["devices"].items():
        node_name = router_name(device_id) if device_data.get("type") == "router" else device_id
        node = net.get(node_name)

        node.cmd("ip link set lo up")
        for iface_name, iface_data in device_data.get("interfaces", {}).items():
            node.cmd(f"ip addr flush dev {iface_name}")
            node.cmd(f"ip addr add {interface_ip_with_prefix(iface_data)} dev {iface_name}")
            node.cmd(f"ip link set {iface_name} up")

        if device_data.get("type") == "host":
            configure_host_default_route(node, device_data, topology)


def configure_host_default_route(host: Node, host_data: dict, topology: dict) -> None:
    interfaces = host_data.get("interfaces", {})
    if len(interfaces) != 1:
        return

    iface_data = next(iter(interfaces.values()))
    peer_iface = iface_data.get("peer")
    if not peer_iface:
        return

    peer_device_id, peer_iface_name = parse_iface_name(peer_iface)
    peer_device = topology["devices"].get(peer_device_id, {})
    gateway = peer_device.get("interfaces", {}).get(peer_iface_name, {}).get("ip")
    if gateway:
        host.cmd("ip route flush default")
        host.cmd(f"ip route add default via {gateway}")


def write_frr_configs(router_id: str, router_data: dict) -> Path:
    name = router_name(router_id)
    config_dir = FRR_BASE_DIR / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.chmod(0o777)

    zebra_conf = config_dir / "zebra.conf"
    ospfd_conf = config_dir / "ospfd.conf"
    staticd_conf = config_dir / "staticd.conf"
    bgpd_conf = config_dir / "bgpd.conf"

    zebra_conf.write_text(
        "\n".join(
            [
                "frr defaults traditional",
                f"hostname {name}",
                "service integrated-vtysh-config",
                f"log file {config_dir}/zebra.log",
                "",
            ]
        ),
        encoding="utf-8",
    )

    networks = sorted(
        {iface_data["cidr"] for iface_data in router_data.get("interfaces", {}).values()},
        key=lambda cidr: ipaddress.ip_network(cidr, strict=False),
    )
    network_lines = [f" network {cidr} area 0.0.0.0" for cidr in networks]
    interface_lines = []
    for iface_name in sorted(router_data.get("interfaces", {})):
        interface_lines.extend(
            [
                f"interface {iface_name}",
                " ip ospf network point-to-point",
                " ip ospf hello-interval 1",
                " ip ospf dead-interval 4",
                "exit",
            ]
        )

    ospfd_conf.write_text(
        "\n".join(
            [
                "frr defaults traditional",
                f"hostname {name}",
                "service integrated-vtysh-config",
                f"log file {config_dir}/ospfd.log",
                *interface_lines,
                "router ospf",
                f" ospf router-id 10.255.0.{int(router_id) + 1}",
                *network_lines,
                "exit",
                "",
            ]
        ),
        encoding="utf-8",
    )

    common_config = "\n".join(
        [
            "frr defaults traditional",
            f"hostname {name}",
            "service integrated-vtysh-config",
            "",
        ]
    )
    staticd_conf.write_text(common_config, encoding="utf-8")
    bgpd_conf.write_text(common_config, encoding="utf-8")

    return config_dir


def start_frr(net: Mininet, topology: dict) -> None:
    if FRR_BASE_DIR.exists():
        shutil.rmtree(FRR_BASE_DIR)
    FRR_BASE_DIR.mkdir(parents=True, exist_ok=True)
    FRR_BASE_DIR.chmod(0o777)

    for device_id, device_data in topology["devices"].items():
        if device_data.get("type") != "router":
            continue

        name = router_name(device_id)
        router = net.get(name)
        config_dir = write_frr_configs(device_id, device_data)
        zserv = config_dir / "zserv.api"
        zebra_pid = config_dir / "zebra.pid"
        ospfd_pid = config_dir / "ospfd.pid"
        staticd_pid = config_dir / "staticd.pid"
        bgpd_pid = config_dir / "bgpd.pid"

        router.cmd(
            f"{ZEBRA_BIN} "
            f"-d -f {config_dir}/zebra.conf "
            f"-i {zebra_pid} "
            f"-z {zserv} "
            "-A 127.0.0.1"
        )
        router.cmd(
            f"{OSPFD_BIN} "
            f"-d -f {config_dir}/ospfd.conf "
            f"-i {ospfd_pid} "
            f"-z {zserv} "
            "-A 127.0.0.1"
        )
        router.cmd(
            f"{STATICD_BIN} "
            f"-d -f {config_dir}/staticd.conf "
            f"-i {staticd_pid} "
            f"-z {zserv} "
            "-A 127.0.0.1"
        )
        router.cmd(
            f"{BGPD_BIN} "
            f"-d -f {config_dir}/bgpd.conf "
            f"-i {bgpd_pid} "
            f"-z {zserv} "
            "-A 127.0.0.1"
        )

    time.sleep(8)


def smoke_test(net: Mininet, topology: dict) -> None:
    hosts = sorted(
        device_id for device_id, device_data in topology["devices"].items() if device_data.get("type") == "host"
    )
    if len(hosts) < 2:
        return

    src = net.get(hosts[0])
    dst_iface = next(iter(topology["devices"][hosts[-1]]["interfaces"].values()))
    dst_ip = dst_iface["ip"]
    info(f"\n*** Smoke test: {hosts[0]} -> {hosts[-1]} ({dst_ip})\n")
    result = src.cmd(f"ping -c 3 -W 2 {dst_ip}")
    info(result)


def run(topology_path: Path, run_smoke_test: bool) -> None:
    topology = load_topology(topology_path)
    topo = JsonTopology(topology)
    net = Mininet(topo=topo, controller=None, autoSetMacs=True, link=TCLink)

    try:
        info("*** Starting Mininet\n")
        net.start()
        configure_interfaces(net, topology)
        info("*** Starting FRR daemons per router\n")
        start_frr(net, topology)

        if run_smoke_test:
            smoke_test(net, topology)

        info("*** Topology ready. Use 'exit' to stop the lab.\n")
        CLI(net)
    finally:
        info("*** Stopping Mininet\n")
        net.stop()
        if FRR_BASE_DIR.exists():
            shutil.rmtree(FRR_BASE_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate a converted topology with Mininet and FRRouting.")
    parser.add_argument("--topology", type=Path, default=Path("/lab/topology.json"))
    parser.add_argument("--no-smoke-test", action="store_true")
    parser.add_argument("--log-level", default=os.environ.get("MININET_LOG_LEVEL", "info"))
    args = parser.parse_args()

    setLogLevel(args.log_level)
    run(args.topology, run_smoke_test=not args.no_smoke_test)


if __name__ == "__main__":
    main()

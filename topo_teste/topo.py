#!/usr/bin/env python3
# ==========================================================
# topo.py - topologia com 1 roteador central (r0)
# e 4 roteadores de borda (r1..r4)
# LAN = eth1 ; P2P(core) = eth0
# ==========================================================

from mininet.net import Mininet
from mininet.node import Node, OVSBridge
from mininet.link import TCLink
from mininet.cli import CLI


class LinuxRouter(Node):
    """Roteador Linux com IP forwarding ativado"""
    def config(self, **params):
        super().config(**params)
        self.cmd('sysctl -w net.ipv4.ip_forward=1')

    def terminate(self):
        self.cmd('sysctl -w net.ipv4.ip_forward=0')
        super().terminate()


def run():
    net = Mininet(link=TCLink)

    # ------------------------------------------------------
    # Roteadores
    # ------------------------------------------------------
    r0 = net.addHost('r0', cls=LinuxRouter, ip=None)
    r1 = net.addHost('r1', cls=LinuxRouter, ip=None)
    r2 = net.addHost('r2', cls=LinuxRouter, ip=None)
    r3 = net.addHost('r3', cls=LinuxRouter, ip=None)
    r4 = net.addHost('r4', cls=LinuxRouter, ip=None)

    # ------------------------------------------------------
    # LANs e Hosts (exemplo LAN1 completa)
    # ------------------------------------------------------
    # LAN1
    s1 = net.addSwitch('s1', cls=OVSBridge)

    h11 = net.addHost('h11', ip='10.1.0.10/24', defaultRoute='via 10.1.0.1')
    h12 = net.addHost('h12', ip='10.1.0.11/24', defaultRoute='via 10.1.0.1')
    net.addLink(s1, h11)
    net.addLink(s1, h12)
    net.addLink(s1, r1, intfName2='r1-eth1')  # LAN = eth1

    # LAN2
    s2 = net.addSwitch('s2', cls=OVSBridge)

    h21 = net.addHost('h21', ip='10.2.0.10/24', defaultRoute='via 10.2.0.1')
    h22 = net.addHost('h22', ip='10.2.0.11/24', defaultRoute='via 10.2.0.1')
    net.addLink(s2, h21)
    net.addLink(s2, h22)
    net.addLink(s2, r2, intfName2='r2-eth1')

    # LAN3
    s3 = net.addSwitch('s3', cls=OVSBridge)

    h31 = net.addHost('h31', ip='10.3.0.10/24', defaultRoute='via 10.3.0.1')
    h32 = net.addHost('h32', ip='10.3.0.11/24', defaultRoute='via 10.3.0.1')
    net.addLink(s3, h31)
    net.addLink(s3, h32)
    net.addLink(s3, r3, intfName2='r3-eth1')

    # LAN4
    s4 = net.addSwitch('s4', cls=OVSBridge)

    h41 = net.addHost('h41', ip='10.4.0.10/24', defaultRoute='via 10.4.0.1')
    h42 = net.addHost('h42', ip='10.4.0.11/24', defaultRoute='via 10.4.0.1')
    net.addLink(s4, h41)
    net.addLink(s4, h42)
    net.addLink(s4, r4, intfName2='r4-eth1')

    # ------------------------------------------------------
    # Links P2P (Roteadores ↔ Core)
    # ------------------------------------------------------
    net.addLink(r1, r0, intfName1='r1-eth0', intfName2='r0-eth1')  # R1–R0
    net.addLink(r2, r0, intfName1='r2-eth0', intfName2='r0-eth2')  # R2–R0
    net.addLink(r3, r0, intfName1='r3-eth0', intfName2='r0-eth3')  # R3–R0
    net.addLink(r4, r0, intfName1='r4-eth0', intfName2='r0-eth4')  # R4–R0

    # ------------------------------------------------------
    # Inicia rede
    # ------------------------------------------------------
    net.start()

    # ------------------------------------------------------
    # Configuração de IPs para o core (r0)
    # ------------------------------------------------------
    r0.cmd('ip addr add 10.255.1.1/31 dev r0-eth1')
    r0.cmd('ip addr add 10.255.2.1/31 dev r0-eth2')
    r0.cmd('ip addr add 10.255.3.1/31 dev r0-eth3')
    r0.cmd('ip addr add 10.255.4.1/31 dev r0-eth4')

    # ------------------------------------------------------
    # Configuração de IPs nas bordas
    # ------------------------------------------------------
    r1.cmd('ip addr add 10.255.1.0/31 dev r1-eth0')
    r1.cmd('ip addr add 10.1.0.1/24 dev r1-eth1')

    r2.cmd('ip addr add 10.255.2.0/31 dev r2-eth0')
    r2.cmd('ip addr add 10.2.0.1/24 dev r2-eth1')

    r3.cmd('ip addr add 10.255.3.0/31 dev r3-eth0')
    r3.cmd('ip addr add 10.3.0.1/24 dev r3-eth1')

    r4.cmd('ip addr add 10.255.4.0/31 dev r4-eth0')
    r4.cmd('ip addr add 10.4.0.1/24 dev r4-eth1')

    # ------------------------------------------------------
    # Rotas estáticas (básicas, redundantes com frr.conf)
    # ------------------------------------------------------
    r1.cmd('ip route add default via 10.255.1.1')
    r2.cmd('ip route add default via 10.255.2.1')
    r3.cmd('ip route add default via 10.255.3.1')
    r4.cmd('ip route add default via 10.255.4.1')

    r0.cmd('ip route add 10.1.0.0/24 via 10.255.1.0')
    r0.cmd('ip route add 10.2.0.0/24 via 10.255.2.0')
    r0.cmd('ip route add 10.3.0.0/24 via 10.255.3.0')
    r0.cmd('ip route add 10.4.0.0/24 via 10.255.4.0')

    print("\n✅ Topologia iniciada!")
    print("Use 'vtysh' em cada roteador para verificar rotas.\n")

    CLI(net)
    net.stop()


if __name__ == '__main__':
    run()


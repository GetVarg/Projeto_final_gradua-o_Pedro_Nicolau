from mininet.net import Mininet
import json

def export_topology(net, output="topologia.json"):
    topo = {"devices": {}}
    for node in net.host + net.switches + net.routers:
        dev = {"interfaces": {}}
        for intf in node.intfList():
            if intf.link:
                peer = intf.link.intf1 if intf.intf2 == intf else intf.link.intf2:
                dev["interfaces"][intf.name] = {"peer": peer.name}
            topo["devices"][node.name] = dev
    with open(output, "w") as f:
        json.dump(topo, f, indent=2)
    print(f">>>>Topologia exportada para {output}")

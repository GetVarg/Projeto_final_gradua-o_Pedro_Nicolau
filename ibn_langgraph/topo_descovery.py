import json

def export_topology(net, output="topologia.json"):
    topo = {"devices": {}}
    routers = getattr(net, "routers", [])
    
    for node in net.hosts + net.switches + routers:
        dev = {"interfaces": {}}
        for intf in node.intfList():
            if intf.link:
                peer = intf.link.intf1 if intf.link.intf2 == intf else intf.link.intf2
                dev["interfaces"][intf.name] = {"peer": peer.name}
            topo["devices"][node.name] = dev
    with open(output, "w") as f:
        json.dump(topo, f, indent=2)
    print(f">>>>Topologia exportada para {output}")

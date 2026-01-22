import json
import os
from datetime import datetime
from grafo_final import build_graph  # Importa sua função que compila o grafo

intents_para_testar = [

    # =========================
    # static_route_add
    # =========================

    # direto
    "Configure a static route on router r1 to reach the 10.3.0.0/24 network via router r0.",

    # mais aberto
    "Make router r2 able to reach the 10.4.0.0/24 subnet using r0 as the next hop.",

    # técnico
    "Add a static routing entry on r3 so that traffic destined to 10.2.0.0/24 is forwarded through router r0.",


    # =========================
    # static_route_del
    # =========================

    # direto
    "Remove the static route to 10.3.0.0/24 from router r1.",

    # mais aberto
    "Router r2 should no longer route traffic to the 10.4.0.0/24 network.",

    # técnico
    "Delete the static routing entry for subnet 10.2.0.0/24 configured on router r3.",


    # =========================
    # ip_forward_enable
    # =========================

    # direto
    "Enable IP forwarding on router r1.",

    # mais aberto
    "Make router r2 forward packets between its interfaces.",

    # técnico
    "Enable net.ipv4.ip_forward on router r3.",


    # =========================
    # icmp_ping
    # =========================

    # direto
    "Ping 10.3.0.1 from router r1.",

    # mais aberto
    "Check connectivity from r2 to the host at 10.4.0.1.",

    # técnico
    "Send ICMP echo requests from router r3 to IP address 10.2.0.1.",


    # =========================
    # firewall_drop_icmp_src
    # =========================

    # direto
    "Block ICMP traffic from the 10.1.0.0/24 subnet on router r1.",

    # mais aberto
    "Prevent hosts in network 10.2.0.0/24 from sending ping requests through router r2.",

    # técnico
    "Install a firewall rule on router r3 to drop ICMP packets originating from 10.3.0.0/24."
]

# intents_para_testar = [
#     # 1. Static routing
#     "Configure a static route on router r1 to reach the 10.4.0.0/24 network via router r0.",

#     # 2. LAN connectivity to gateway
#     "Ensure that all hosts in the 10.1.0.0/24 subnet can ping their gateway at 10.1.0.1.",

#     # 3. Dynamic routing (OSPF)
#     "Configure OSPF on all routers to enable automatic neighbor discovery.",

#     # 4. Default route adjustment
#     "Remove the default route from router r2 and configure a new default route via interface r2-eth0 toward router r0.",

#     # 5. IP forwarding
#     "Enable IP packet forwarding on router r3 to allow traffic between switch s3 and router r0.",

#     # 6. Access control between LANs
#     "Block any host connected to switch s1 from accessing host h41 over HTTP.",

#     # 7. Network isolation (security zone)
#     "Create an isolated network zone for the finance department hosts h21 and h22.",

#     # 8. Restricted management access
#     "Allow only SSH traffic originating from router r1 to access the central router r0.",

#     # 9. ICMP filtering
#     "Configure a firewall on router r0 to drop ICMP packets originating from the 10.3.0.0/24 network.",

#     # 10. Switch isolation
#     "Isolate switch s4 from the rest of the topology, allowing only management access via router r0.",

#     # 11. SDN switch mode (conditional)
#     "If supported, configure all switches to operate in SDN mode using OpenFlow 1.3.",

#     # 12. SDN controller assignment
#     "If SDN is enabled, configure the SDN controller at IP 192.168.56.101 to manage switch s2.",

#     # 13. Traffic redirection / inspection
#     "Ensure that all HTTP traffic originating from the 10.2.0.0/24 network is explicitly forwarded through router r0 for inspection.",

#     # 14. Physical topology validation
#     "Verify that there is no redundant physical path between routers r1 and r4 in the current topology.",

#     # 15. Bandwidth limitation
#     "Limit the bandwidth of interface r1-eth1 to 10 Mbps.",

#     # 16. Routing loop detection
#     "Check for potential routing loops between routers r1, r0, and r2.",

#     # 17. Resource monitoring
#     "Monitor CPU utilization on all FRR routers.",

#     # 18. Route analysis
#     "Analyze the routing table of router r4 and confirm the path to the 10.3.0.0/24 LAN.",

#     # 19. Failure simulation
#     "Simulate a failure on the r0–r1 link and verify the resulting loss of connectivity for host h11.",

#     # 20. Inventory validation
#     "Generate an inventory report comparing the physical topology with the JSON topology description."
# ]


# intents_para_testar = [
#     "Garanta que todos os hosts da sub-rede 10.1.0.0/24 consigam pingar o gateway 10.255.1.1."
# ]

def rodar_bateria_de_testes():
    import time
    app = build_graph()
    resultados = []
    
    print(f"Iniciando bateria de {len(intents_para_testar)} testes...")

    for i, texto_intencao in enumerate(intents_para_testar):
        print(f"[{i+1}/{len(intents_para_testar)}] Testando: {texto_intencao[:50]}...")
        
        # Configuração inicial do estado para cada teste
        # O thread_id permite que o SQLite do LangGraph armazene cada teste separadamente
        config = {"configurable": {"thread_id": f"teste_{i+1}"}}
        estado_inicial = {
            "user_intent_text": texto_intencao,
            "needs_human": False,
            "topology": {},  # Inicializa vazio para evitar KeyError
            "entities": [],
            "plan": {
                "steps": [],
                "warnings": [],
                "needs_human": True,
                "dry_run": True
            },
            "requirements": []
        }

        try:
            # Executa o grafo completo
            inicio = time.time()
            final_state = app.invoke(estado_inicial, config)
            fim = time.time()
            # Armazena o que a LLM gerou no nó 'plan' ou 'context'
            intent_obj = final_state.get("intent")
            plan_obj = final_state.get("plan")

            resultado_teste = {
                "id": i + 1,
                "intent_original": texto_intencao,
                "classificacao": intent_obj.model_dump() if hasattr(intent_obj, "model_dump") else intent_obj,
                "entidades_extraidas": [e.model_dump() if hasattr(e, "model_dump") else e for e in final_state.get("entities", [])],
                "entity_selectors": final_state.get("entity_selectors", []),
                "plan": plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj,
                "status_verificacao": final_state.get("verification", {})
            }
            resultados.append(resultado_teste)
            
            print(f"[OK] Teste {i+1} finalizado. Intent detectada: {intent_obj.name}. Tempo de excução: {fim - inicio}")
        except Exception as e:
            print(f"Erro no teste {i+1}: {e}")
            resultados.append({"id": i+1, "erro": str(e)})

    # Salva o log completo em JSON para sua análise de IC
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_saida = f"logs_teste/resultado_llm_{timestamp}.json"
    
    os.makedirs("logs_teste", exist_ok=True)
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    
    print(f"\nTestes concluídos! Log salvo em: {arquivo_saida}")

if __name__ == "__main__":
    rodar_bateria_de_testes()
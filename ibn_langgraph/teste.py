import json
import os
from datetime import datetime
from grafo_final import build_graph  # Importa sua função que compila o grafo


intents_para_testar = [
    # =========================
    # static_route_add
    # =========================
    "Configure a static route on router r1 to reach the 10.3.0.0/24 network via router r0.",
    "Make router r2 able to reach the 10.4.0.0/24 subnet using r0 as the next hop.",
    "Add a static routing entry on r3 so that traffic destined to 10.2.0.0/24 is forwarded through router r0.",

    # =========================
    # static_route_del
    # =========================
    "Remove the static route to 10.3.0.0/24 from router r1.",
    "Router r2 should no longer route traffic to the 10.4.0.0/24 network.",
    "Delete the static routing entry for subnet 10.2.0.0/24 configured on router r3.",

    # =========================
    # ip_forward_enable
    # =========================
    "Enable IP forwarding on router r1.",
    "Make router r2 forward packets between its interfaces.",
    "Enable net.ipv4.ip_forward on router r3.",

    # =========================
    # icmp_ping
    # =========================
    "Ping 10.3.0.1 from router r1.",
    "Check connectivity from r2 to the host at 10.4.0.1.",
    "Send ICMP echo requests from router r3 to IP address 10.2.0.1.",

    # =========================
    # firewall_drop_icmp_src
    # =========================
    "Block ICMP traffic from the 10.1.0.0/24 subnet on router r1.",
    "Prevent hosts in network 10.2.0.0/24 from sending ping requests through router r2.",
    "Install a firewall rule on router r3 to drop ICMP packets originating from 10.3.0.0/24."
]

intents_para_testar = [
    # =========================
    # static_route_add (6)
    # =========================
    "Set up a static route on router r1 for the 10.3.0.0/24 prefix using r0 as the next hop.",
    "On r1, add a static routing entry so that traffic for subnet 10.3.0.0/24 is forwarded through r0.",
    "Configure router r2 with a static route to reach the 10.4.0.0/24 network via r0.",
    "Create a static route in r2 pointing the 10.4.0.0/24 subnet to gateway r0.",
    "Provision a static route on r3 so packets destined to 10.2.0.0/24 go through router r0.",
    "Define a static routing rule on router r3 for prefix 10.2.0.0/24 using r0.",

    # =========================
    # static_route_del (6)
    # =========================
    "Remove the static route to the 10.3.0.0/24 network from router r1.",
    "On r1, delete any static routing entry associated with subnet 10.3.0.0/24.",
    "Ensure router r2 no longer has a route configured for the 10.4.0.0/24 prefix.",
    "Clear the static routing rule on r2 that forwards traffic to 10.4.0.0/24.",
    "Delete the static route entry on router r3 for the 10.2.0.0/24 subnet.",
    "Make sure r3 no longer routes traffic towards the 10.2.0.0/24 network.",

    # =========================
    # ip_forward_enable (6)
    # =========================
    "Enable IPv4 packet forwarding on router r1.",
    "Turn on IP forwarding functionality on r1.",
    "Activate packet forwarding between interfaces on router r2.",
    "Ensure that router r2 is configured to forward IP packets.",
    "Set the net.ipv4.ip_forward parameter to 1 on router r3.",
    "Switch on IP forwarding support at the kernel level on r3.",

    # =========================
    # icmp_ping (6)
    # =========================
    "From router r1, send ICMP echo requests to the address 10.3.0.1.",
    "Test connectivity from r1 by pinging host 10.3.0.1.",
    "Verify reachability from router r2 to IP 10.4.0.1 using ICMP.",
    "On r2, run a ping test towards the host located at 10.4.0.1.",
    "Initiate an ICMP echo test from router r3 to 10.2.0.1.",
    "Check network connectivity by pinging 10.2.0.1 from r3.",

    # =========================
    # firewall_drop_icmp_src (6)
    # =========================
    "Block ICMP packets originating from the 10.1.0.0/24 subnet on router r1.",
    "On r1, install a rule to drop ICMP traffic sourced from 10.1.0.0/24.",
    "Prevent ICMP echo requests from the 10.2.0.0/24 network from passing through router r2.",
    "Configure r2 to deny ICMP traffic coming from subnet 10.2.0.0/24.",
    "Drop ICMP packets originating in the 10.3.0.0/24 subnet on router r3.",
    "Set up a firewall rule on r3 to filter out ICMP traffic from 10.3.0.0/24."
]


def to_jsonable(obj):
    """
    Converte objetos comuns do pipeline (Pydantic, dataclass, list/dict, etc.)
    para algo serializável em JSON, sem depender de regex.
    """
    if obj is None:
        return None

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    # dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)

    # dict / list
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    # fallback: tenta serializar como string
    return str(obj)


def rodar_bateria_de_testes():
    import time
    app = build_graph()
    resultados = []
    comandos_execucao = []

    print(f"Iniciando bateria de {len(intents_para_testar)} testes...")

    for i, texto_intencao in enumerate(intents_para_testar):
        print(f"[{i+1}/{len(intents_para_testar)}] Testando: {texto_intencao[:50]}...")

        config = {"configurable": {"thread_id": f"teste_{i+1}"}}
        estado_inicial = {
            "user_intent_text": texto_intencao,
            "needs_human": False,
            "topology": {},        # aqui você guarda o "context slice" em algum momento
            "entities": [],        # aqui você guarda entidades extraídas
            "plan": {"steps": [], "warnings": [], "needs_human": True, "dry_run": True},
            "requirements": [],
            "cli_commands": None,
        }

        try:
            inicio = time.time()
            final_state = app.invoke(estado_inicial, config)
            fim = time.time()

            # --- Captura: Intent / Entidades / Contexto / Plano ---
            intent_obj = final_state.get("intent")
            entities_obj = final_state.get("entities")  # <- entidades extraídas/validadas

            # context slice pode estar em chaves diferentes dependendo do seu grafo
            # (mantive fallback pra não quebrar o log caso o nome mude)
            context_slice_obj = (
                final_state.get("context_slice")
                or final_state.get("topology")      # muitas vezes você usa "topology" pra carregar o slice final
                or final_state.get("topo_context")  # se você separou isso em outro campo
            )

            plan_obj = final_state.get("plan")

            resultado_teste = {
                "id": i + 1,
                "intent_original": texto_intencao,
                "classificacao": to_jsonable(intent_obj),
                "entidades_extraidas": to_jsonable(entities_obj),
                "context_slice": to_jsonable(context_slice_obj),
                "plan": to_jsonable(plan_obj),
            }
            resultados.append(resultado_teste)

            # --- Captura: Comandos CLI ---
            exec_result = final_state.get("cli_commands")
            if exec_result is not None:
                exec_result = to_jsonable(exec_result)
                comandos_execucao.append({
                    "id_teste": i + 1,
                    "intent": texto_intencao,
                    "status": exec_result.get("status"),
                    "commands": exec_result.get("commands")
                })
            else:
                comandos_execucao.append({
                    "id_teste": i + 1,
                    "intent": texto_intencao,
                    "cli": None
                })

            print(f"[OK] Teste {i+1} finalizado. Tempo: {fim - inicio:.2f}s")

        except Exception as e:
            print(f"Erro no teste {i+1}: {e}")
            resultados.append({"id": i + 1, "intent_original": texto_intencao, "erro": str(e)})

    # --- SALVAMENTO DOS ARQUIVOS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs_teste", exist_ok=True)

    arquivo_log = f"logs_teste/resultado_llm_{timestamp}.json"
    with open(arquivo_log, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    arquivo_comandos = f"logs_teste/comandos_cli_{timestamp}.json"
    with open(arquivo_comandos, "w", encoding="utf-8") as f:
        json.dump(comandos_execucao, f, indent=4, ensure_ascii=False)

    print(f"\n[SUCESSO] Log de processamento salvo em: {arquivo_log}")
    print(f"[SUCESSO] Comandos CLI gerados salvos em: {arquivo_comandos}")


if __name__ == "__main__":
    rodar_bateria_de_testes()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from main import IBNState  # Certifique-se que o IBNState está no seu main.py
from no_grafo import (
    node_intent, 
    node_entities, 
    node_requirements, 
    node_anonymize, 
    node_context, 
    node_plan, 
    node_decide_exec, 
    node_execute,
    node_command_profile,
    node_verify
)

def build_graph():
    # 1. Inicializa o Grafo com o estado definido no seu main.py
    workflow = StateGraph(IBNState)

    # 2. Adiciona os Nós (Nodes)
    # Cada nó é uma função que você definiu no no_grafo.py
    workflow.add_node("intent", node_intent)
    workflow.add_node("context", node_context)
    workflow.add_node("command_profile", node_command_profile)
    workflow.add_node("entities", node_entities)
    workflow.add_node("requirements", node_requirements)
    workflow.add_node("anonymize", node_anonymize)
    workflow.add_node("plan", node_plan)
    workflow.add_node("decide_exec", node_decide_exec)
    workflow.add_node("execute", node_execute)
    workflow.add_node("verify", node_verify)

    # 3. Define as Bordas (Edges) e o Fluxo
    workflow.set_entry_point("intent")
    
    workflow.add_edge("intent", "entities")
    workflow.add_edge("entities", "anonymize")
    workflow.add_edge("anonymize", "context")
    workflow.add_edge("context", "requirements")

    def router_needs_human(state: IBNState):
        if state.get("needs_human"):
            return "end"

        topo = state.get("topology") or {}
        devices = (topo.get("devices") or {})
        if isinstance(devices, dict) and len(devices) == 0:
            return "end"

        return "continue"

    workflow.add_conditional_edges(
        "requirements",
        router_needs_human,
        {
            "end": END,
            "continue": "command_profile"
        }
    )

    workflow.add_edge("command_profile", "plan")
    workflow.add_edge("plan", "decide_exec")
    workflow.add_edge("decide_exec", "execute")
    workflow.add_edge("execute", "verify")
    workflow.add_edge("verify", END)

    # 4. Configura a Persistência (Memória do Grafo)
    # Isso cria o arquivo sqlite que armazena o histórico dos seus testes
    conn = sqlite3.connect("ibn_checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    app = build_graph()
    print("Grafo de Intent-Based Networking compilado com sucesso!")
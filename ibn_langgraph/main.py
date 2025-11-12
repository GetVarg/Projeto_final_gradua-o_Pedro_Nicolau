from __future__ import annotations
from typing import TypedDict, List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import re

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from uuid import uuid4
from langchain_ollama import ChatOllama
OLLAMA_MODEL = "llama3.2:3b"
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


class Intent(BaseModel):
    name: str
    confidence: float = 0.0
    rationale: Optional[str] = None

class Entity(BaseModel):
    type: str
    value: str
    meta: Dict = Field(default_factory=dict)

class Requirement(BaseModel): 
    key: str
    ok: bool
    details: Optional[str] = None

class IBNState(TypedDict, total=False):
    # entrada
    user_intent_text: str
    timestamp: str

    # cache/topologia
    topology: Dict

    # processamento
    intent: Intent
    entities: List[Dict]
    requirements: List[Dict]
    anonymization_map: Dict[str, str]
    json_context: Dict
    plan: ExecPlan
    exec_result: Dict
    verification: Dict

    # controle
    needs_human: bool
    error: Optional[str]

class ExecPlan(BaseModel):
    steps: List[Dict] = []
    warnings: List[str] = []
    needs_human: bool = False
    dry_run: bool = True


def node_intent(state: IBNState) -> IBNState:
    text = state["user_intent_text"]
    system = """
    You are an Intent Classifier for an Intent-Based Networking (IBN) middleware.
    Classify the text into one of the following categories:
        - configuration | monitoring | control_security | diagnostic | removal_cleanup

    Respond ONLY in valid JSON:
    {
      "category": "string",
      "name": "slug_identifier",
      "confidence": 0.0,
      "rationale": "short explanation"
    }
    """

    user = f"Classify the following intent:\n\"\"\"{text}\"\"\""
            
    try:
        resp = llm.invoke([{"role": "system", "content": system},
                            {"role": "user", "content": user}])
        intent = json.loads(resp.content)
        state["intent"] = intent
    except Exception as e:
        state["intent"] = {"category": "unknown", "name": "error", "confidence": 0.0, "rationale": str(e)}
        state["needs_human"] = True
    return state

def node_plan(state: IBNState) -> IBNState:
    text = state["user_intent_text"]
    topo = state["topology"]

    system = """
        You are an Intent-Based Networking (IBN) assistant.
        Your role is to generate a reliable promp and to generate a deterministic and 
        safe NETWORK CONFIGURATION PLAN based on a structured Intent 
        object and the provided Network Context.

        Rules:
        - Respond STRICTLY in valid JSON — no text, explanations, or comments outside the JSON block.
        - NEVER invent devices, interfaces, IPs, or routes: use only data present in the NetworkContext or NetTable.
        - All generated commands must be idempotent and minimal (avoid redundant or unsafe actions).
        - Follow any guardrails or policies when they are provided.
        - If any essential data is missing, ambiguous, or could affect operational safety, 
        set "needs_human": true and include a clear warning message.
        - Always prefer safety and human oversight over automation.
        - Follow any guardrails or policies when they are provided.
        - Extract the DetectedEntities from the input text/context.
        - Each step in the plan must include: "device", "action", and a list of "commands".

        Expected JSON output:
            "PromptTemplate": {
                "Role": "You are an intent-based network assistant.",
                "Intent": { ...structured intent object... },
                "NetworkContext": { ...JSON topology/context... },
                "Instructions": "Summary of rules and goals for this execution."
            },
            "NetTable": {
                "Device.Interface": { "ip": "string", "mask": "string", "neighbor": "string" }
            },
            "AnonymizationTable": {
                "real_value": "anonymized_id"
            },
            "DetectedEntities": {
                "routers": [array of strings],
                "interfaces": [array of strings],
                "subnets": [array of strings]
            },
            "Plan": {
                "steps": [
                    { "device": "string", "action": "configure|probe|diag|cleanup", "commands": ["string"] }
                ],
                "warnings": ["string"],
                "needs_human": false
            }	

    """

    intent_obj = state.get("intent", {})

    user = f"""
        Intent:
        {json.dumps(intent_obj, indent=2)}

        NetworkContext:
        {json.dumps(topo, indent=2)}

        Instructions:
        Generate the PLAN in the JSON format described above.
        """
    
    try:
        resp = llm.invoke([{"role": "system", "content": system},
                            {"role": "user", "content": user}])
        raw = resp.content

        match = re.search(r"\{.*\}\s*$", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)

        plan = data.get("Plan", {"steps": [], "warnings": ["no plan returned"], "needs_human": True})
        state["plan"] = plan

    except Exception as e:
        state["plan"] = {"steps": [], "warnings": [f"LLM error: {e}"], "needs_human": True}
    return state

DATA_PATH = Path(__file__).resolve().parents[1] / "topologia.json"

def node_load_topology(state: IBNState) -> IBNState:
    if "topology" not in state:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Topologia.json nao encontrada em {DATA_PATH}")
        state["topology"] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return state

def validate_static_policy(topo: Dict) -> bool:
    for name, dev in topo.get("devices", {}).items():
        if dev.get("type") == "router" and name != "r0":
            ifaces = dev.get("interfaces", {})
            if not any("r0" in v.get("peer", "") for v in ifaces.values()):
                return False
    return True

def node_requirements(state: IBNState) -> IBNState:
    reqs = []
    ok = True
    if state["intent"].category == "configuration" and "static" in state["intent"].name:
        ok = validate_static_policy(state["topology"])
    state["requirements"] = [{"key": "static_policy_checjk", "ok": ok}]
    return state

def node_execute(state: IBNState) -> IBNState:
    result = {"status": "DRY_RUN", "applied": [], "skipped": []}
    plan = state.get("plan", {})

    for step in plan.get("steps", []):
        if plan.get("needs_human", False):
            result["skipped"].append(step)
        else:
            # Aqui entraria integração real com Mininet, Netmiko, etc.
            result["applied"].append(step)

    state["exec_result"] = result
    return state

# =====================================
# Construção do grafo
# =====================================
def build_graph():
    g = StateGraph(IBNState)
    g.add_node("load_topology", node_load_topology)
    g.add_node("intent", node_intent)
    g.add_node("plan", node_plan)
    g.add_node("execute", node_execute)

    g.set_entry_point("load_topology")
    g.add_edge("load_topology", "intent")
    g.add_edge("intent", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)

    # checkpointer = SqliteSaver.from_conn_string("ibn_checkpoints.sqlite")
    # return g.compile(checkpointer=checkpointer)
    return g

# =====================================
# Execução principal
# =====================================
if __name__ == "__main__":
    graph = build_graph()

    with SqliteSaver.from_conn_string("ibn_checkpoints.sqlite") as cp:

        app = graph.compile(checkpointer=cp)

        user_text = "Configure L3 connectivity between r1 and r2 via r0 using static routes."
        result = app.invoke({
            "user_intent_text": user_text,
            "timestamp": datetime.now().isoformat()},
            config={"configurable": {"thread_id": f"run-{uuid4()}"}}   
        )

        print("\n=== FINAL RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

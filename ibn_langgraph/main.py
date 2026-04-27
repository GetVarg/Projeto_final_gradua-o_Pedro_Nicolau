from __future__ import annotations
from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import re

from pydantic import BaseModel, Field

class Intent(BaseModel):
    name: str
    confidence: float = 0.0
    category: str = "config"
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
    topology_full: Dict
    slice_topology: Dict

    # processamento
    intent: Intent
    entities: List[Dict]
    entity_selectors: List[Dict]
    requirements: List[Dict]
    anonymization_map: Dict[str, str]
    plan: ExecPlan
    exec_result: Dict
    verification: Dict

    # perfil de comandos / ambiente
    cli_commands: Optional[Dict[str, Any]]

    work: Dict[str, Any]              # root_intent, subintents, cursor, results, etc.
    active_subintent_id: Optional[str]
    active_subintent_text: Optional[str]
    plan_items: List[Dict[str, Any]]
    plan_steps: List[Dict[str, Any]]
    warnings: List[str] 

    # controle
    needs_human: bool
    error: Optional[str]

class ExecPlan(BaseModel):
    steps: List[Dict] = []
    warnings: List[str] = []
    needs_human: bool = False
    dry_run: bool = True

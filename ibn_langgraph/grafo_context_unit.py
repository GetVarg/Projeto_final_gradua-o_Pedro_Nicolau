import json
import time
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END

from main import IBNState
import no_grafo_tcc as tcc
from no_grafo_tcc import node_context


TOPOLOGY_PATH = "dataset/topologias_convertidas/gabriel/10/0.json"
BENCHMARK_REPEATS = 3
BENCHMARK_TEMPERATURE = tcc.temperature


CONTEXT_UNIT_CASES = [
    {
        "id": "C1",
        "label": "ipv6_forwarding_and_route_interface",
        "subintent": {
            "id": "S1",
            "text": "Enable IPv6 forwarding on router 9",
            "intent_frame": {
                "goal": "enable_ipv6_forwarding",
                "arguments": {
                    "target": "router 9",
                    "forwarding_protocol": "IPv6",
                },
                "notes": {},
            },
        },
    },
    {
        "id": "C2",
        "label": "ipv6_route_through_interface",
        "subintent": {
            "id": "S2",
            "text": "Add an IPv6 route to 3000::/64 through interface 9-eth0",
            "intent_frame": {
                "goal": "add_ipv6_route",
                "arguments": {
                    "destination_prefix": "3000::/64",
                    "next_hop_interface": "9-eth0",
                },
                "notes": {},
            },
        },
    },
    {
        "id": "C3",
        "label": "indirect_next_hop_from_neighbor_interface",
        "subintent": {
            "id": "S3",
            "text": "Configure a static route on router 0 to 172.16.9.0/24 using the IP address of the neighbor connected to interface 0-eth2 as next-hop.",
            "intent_frame": {
                "goal": "configure_static_route",
                "arguments": {
                    "router": "0",
                    "destination_network": "172.16.9.0/24",
                    "next_hop_source": "neighbor_connected_to_interface",
                    "interface": "0-eth2",
                },
                "notes": {
                    "routing_type": "static",
                },
            },
        },
    },
    {
        "id": "C4",
        "label": "bgp_peer_group_binding",
        "subintent": {
            "id": "S4",
            "text": "Configure a BGP peer-group named IBGP on router 0 and bind neighbor 10.0.1.2 to it.",
            "intent_frame": {
                "goal": "configure_bgp_peer_group",
                "arguments": {
                    "router": "0",
                    "peer_group_name": "IBGP",
                    "neighbor": "10.0.1.2",
                },
                "notes": {},
            },
        },
    },
    {
        "id": "C5",
        "label": "unknown_interface_edge_case",
        "subintent": {
            "id": "S5",
            "text": "Add an IPv6 route to 3000::/64 through interface 9-eth999.",
            "intent_frame": {
                "goal": "add_ipv6_route",
                "arguments": {
                    "destination_prefix": "3000::/64",
                    "next_hop_interface": "9-eth999",
                },
                "notes": {},
            },
        },
    },
    {
        "id": "C6",
        "label": "empty_arguments_edge_case",
        "subintent": {
            "id": "S6",
            "text": "Enable IP forwarding.",
            "intent_frame": {
                "goal": "enable_ip_forwarding",
                "arguments": {},
                "notes": {},
            },
        },
    },
]


def load_experiment_topology(path: str) -> dict:
    topo_path = Path(path)
    if not topo_path.exists():
        raise FileNotFoundError(f"Topology file not found: {path}")

    with topo_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Topology file must contain a JSON object: {path}")

    return data


def timed_node(node_name, fn):
    def _wrapped(state):
        start = time.perf_counter()
        out = fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timing = out.get("_timing") or {"nodes": []}
        timing["nodes"].append({"node": node_name, "ms": round(elapsed_ms, 3)})
        out["_timing"] = timing
        return out

    return _wrapped


def build_context_unit_graph():
    workflow = StateGraph(IBNState)
    workflow.add_node("context", timed_node("context", node_context))
    workflow.set_entry_point("context")
    workflow.add_edge("context", END)
    return workflow.compile()


def _slice_counts(slice_topology: dict) -> dict:
    devices = (slice_topology or {}).get("devices") or {}
    networks = (slice_topology or {}).get("networks") or {}
    iface_count = 0
    for meta in devices.values():
        iface_count += len((meta.get("interfaces") or {}))
    return {
        "devices": len(devices),
        "networks": len(networks),
        "interfaces": iface_count,
    }


def print_context_case(case_index: int, case: dict, result: dict) -> None:
    print("\n" + "=" * 100)
    print(f"[CONTEXT][CASE {case_index}] {case['label']}")
    print("=" * 100)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _discretize_output_view(case: dict) -> dict:
    subintent = case.get("subintent") or {}
    frame = subintent.get("intent_frame") or {}
    raw_arguments = frame.get("arguments") or {}

    if isinstance(raw_arguments, dict) and (
        isinstance(raw_arguments.get("topology_arguments"), dict)
        or isinstance(raw_arguments.get("semantic_only_arguments"), dict)
    ):
        topology_arguments = raw_arguments.get("topology_arguments") or {}
    else:
        topology_arguments = raw_arguments if isinstance(raw_arguments, dict) else {}

    return {
        "subintents": [
            {
                "id": subintent.get("id", "S1"),
                "text": subintent.get("text", ""),
                "atomic": True,
                "depends_on": [],
                "requires_topology_context": bool(topology_arguments),
            }
        ],
        "confidence": 1.0,
        "needs_human": False,
    }


def _empty_case_output(case_index: int, case: dict) -> dict:
    subintent = case.get("subintent") or {}
    return {
        "id": case_index,
        "intent": subintent.get("text", ""),
        "node_discretize_intent": _discretize_output_view(case),
        "node_context": {
            "subintent_id": subintent.get("id", "S1"),
            "requested_topology_entities": {},
            "matched_topology_entities": [],
            "unmatched_topology_arguments": [],
            "slice_topology": {
                "devices": [],
                "interfaces": [],
                "networks": [],
            },
            "confidence": 0.0,
            "needs_human": True,
        },
    }


def _format_case_output(case_index: int, case: dict, out: dict | None) -> dict:
    result = _empty_case_output(case_index, case)
    context = (out or {}).get("context") or {}
    slice_topology = context.get("slice_topology") or {}
    slice_devices = slice_topology.get("devices") or {}
    slice_networks = slice_topology.get("networks") or {}
    interface_names = []
    for dev_meta in slice_devices.values():
        for if_name in ((dev_meta or {}).get("interfaces") or {}).keys():
            if if_name not in interface_names:
                interface_names.append(if_name)
    result["node_context"] = {
        "subintent_id": context.get("subintent_id") or (case.get("subintent") or {}).get("id", "S1"),
        "requested_topology_entities": context.get("requested_topology_entities") or {},
        "matched_topology_entities": context.get("matched_topology_entities") or [],
        "unmatched_topology_arguments": context.get("unmatched_topology_arguments") or [],
        "slice_topology": {
            "devices": sorted(slice_devices.keys()),
            "interfaces": interface_names,
            "networks": sorted(slice_networks.keys()),
        } if slice_devices or slice_networks else {
            "devices": [],
            "interfaces": [],
            "networks": [],
        },
        "confidence": float(context.get("confidence", 0.0) or 0.0),
        "needs_human": bool(context.get("needs_human", (out or {}).get("needs_human", False))),
    }
    return result


def run_context_unit_suite(app, cases: list[dict], model_name: str, topology: dict, run_label: str = "r1") -> dict:
    tcc.llm = tcc.HFRouterChat(model=model_name, temperature=BENCHMARK_TEMPERATURE)
    tcc.temperature = BENCHMARK_TEMPERATURE

    suite_start = time.perf_counter()
    results = []

    for i, case in enumerate(cases, start=1):
        state = {
            "topology_full": topology,
            "work": {
                "subintent": case["subintent"],
            },
        }

        start = time.perf_counter()
        try:
            out = app.invoke(
                state,
                config={
                    "configurable": {"thread_id": f"{model_name}-{run_label}-context-{case['id']}"},
                    "recursion_limit": 20,
                },
            )
            elapsed = time.perf_counter() - start
        except Exception as exc:
            elapsed = time.perf_counter() - start
            result = _empty_case_output(i, case)
            result["node_context"]["needs_human"] = True
            result["node_context"]["error"] = str(exc)
            results.append(result)
            print_context_case(i, case, result)
            continue

        result = _format_case_output(i, case, out)

        results.append(result)
        print_context_case(i, case, result)

    suite_elapsed = time.perf_counter() - suite_start
    return {
        "mode": "context_unit",
        "run_label": run_label,
        "temperature": BENCHMARK_TEMPERATURE,
        "model": model_name,
        "suite_elapsed_sec": round(suite_elapsed, 3),
        "num_cases": len(cases),
        "results": results,
    }


def _safe_model_tag(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_context_unit_report(report: dict, base_dir: str = "outputs") -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _safe_model_tag(report["model"])

    model_dir = Path(base_dir) / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    json_path = model_dir / f"context_unit_{ts}.json"
    txt_path = model_dir / f"context_unit_{ts}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"mode={report.get('mode')}\n")
        f.write(f"run_label={report.get('run_label')}\n")
        f.write(f"temperature={report.get('temperature')}\n")
        f.write(f"model={report.get('model')}\n")
        f.write(f"suite_elapsed_sec={report.get('suite_elapsed_sec')}\n")
        f.write(f"num_cases={report.get('num_cases')}\n\n")

        for result in report.get("results", []):
            f.write("=" * 100 + "\n")
            f.write(f"[CONTEXT][CASE {result.get('id')}] {result.get('intent')}\n")
            f.write("=" * 100 + "\n")
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
            f.write("\n")

    return str(json_path), str(txt_path)


if __name__ == "__main__":
    app = build_context_unit_graph()
    topology = load_experiment_topology(TOPOLOGY_PATH)

    test_models = [
        "meta-llama/Llama-3.1-8B-Instruct:novita",
    ]

    print("\n" + "=" * 100)
    print("[CONTEXT UNIT CONFIG]")
    print("=" * 100)
    print(f"cases={len(CONTEXT_UNIT_CASES)}")
    print(f"repeats={BENCHMARK_REPEATS}")
    print(f"temperature={BENCHMARK_TEMPERATURE}")
    print(f"models={len(test_models)}")

    all_reports = []

    for model_name in test_models:
        for repeat_idx in range(BENCHMARK_REPEATS):
            run_label = f"r{repeat_idx + 1}"

            print("\n" + "=" * 100)
            print(f"[CONTEXT UNIT] Running model: {model_name} | run={run_label}")
            print("=" * 100)

            report = run_context_unit_suite(
                app,
                CONTEXT_UNIT_CASES,
                model_name,
                topology,
                run_label=run_label,
            )
            json_path, txt_path = save_context_unit_report(report)
            all_reports.append(report)

            print("\n" + "#" * 100)
            print(f"[CONTEXT UNIT SUMMARY] model={report['model']} | run={report['run_label']}")
            print(f"[CONTEXT UNIT SUMMARY] suite_elapsed_sec={report['suite_elapsed_sec']}")
            print(f"[CONTEXT UNIT SUMMARY] json_saved_to={json_path}")
            print(f"[CONTEXT UNIT SUMMARY] txt_saved_to={txt_path}")
            print("#" * 100)

    print("\n" + "=" * 100)
    print("[CONTEXT UNIT] FINAL SUMMARY")
    print("=" * 100)
    for report in all_reports:
        print(
            f"- mode=context_unit | model={report['model']} | run={report.get('run_label')} | "
            f"suite_elapsed_sec={report['suite_elapsed_sec']}"
        )

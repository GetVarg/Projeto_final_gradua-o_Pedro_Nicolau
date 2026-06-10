import grafo_final as g


def main() -> None:
    app = g.build_graph_debug()
    test_intents = [
        case["intent"]
        for case in (g.GOLDEN_COMMANDS_BY_INTENT.values() or g.BENCHMARK_DIAGNOSTIC_CASES)
    ]

    print(f"[R2 PIPELINE ONLY] intents={len(test_intents)}")
    print(f"[R2 PIPELINE ONLY] discretize_model={g.LLAMA_8B_MODEL}")
    print(f"[R2 PIPELINE ONLY] downstream_model={g.LLAMA_3B_MODEL}")

    pipeline_report = g.run_intent_suite(
        app,
        test_intents,
        g.LLAMA_8B_MODEL,
        run_label="r2",
        downstream_model_name=g.LLAMA_3B_MODEL,
    )
    pipeline_report["profile"] = "discretize_8b_downstream_1b"

    pipeline_json_path = g.save_benchmark_report(pipeline_report)
    pipeline_txt_path = g.save_text_report(pipeline_report)
    argument_resolver_log_path = g.save_argument_resolver_log(pipeline_report)
    discretize_json_path = g.save_benchmark_report(
        g.build_discretize_only_report(pipeline_report)
    )

    print(f"[PIPELINE R2] json={pipeline_json_path}")
    print(f"[PIPELINE R2] txt={pipeline_txt_path}")
    print(f"[PIPELINE R2] argument_resolver_log={argument_resolver_log_path}")
    print(f"[PIPELINE R2] discretize_only={discretize_json_path}")


if __name__ == "__main__":
    main()

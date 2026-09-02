from src.iclr27_phase74r.gates import blocked_status, compute_gates


def test_denominator_mismatch_blocks_even_when_order_is_reproducible():
    gates = compute_gates(input_ok=True, model_order={"model_order_exactly_reproducible": True, "model_key_set_equals_metadata": False, "count": 82}, prefix_ok=True, asset_identity={"schema_complete": True, "duplicates_preserved": True}, branch_fixture={"mapped_images": 1, "q0_candidate_count": 1, "classification": "UNIQUE_MAPPING"}, reliability={"null_before_replay": True, "no_zero_for_unreplayed": True}, fragmentation={"ambiguous_overlap": True, "physical_fragmentation": True}, metamorphic={"category_shuffle": True, "event_label_swap": True, "physical_id_renumber": True, "future_append": True, "source_before_target": True, "repeat_determinism": True, "atomic_crash": True, "static_antihardcode": True}, reproducibility=True, artifact_format={"json_parse": True, "jsonl_parse": True, "timeline_json_array": True}, resource={"preflight_recorded": True, "postflight_recorded": True, "no_external_kill": True, "no_duplicate_supervisor": True}, metadata_count=152, metadata_key_count=152)
    assert not gates["EVALUATOR_DENOMINATOR"]["pass"]
    assert blocked_status(gates) == "PHASE74R_BLOCKED_DENOMINATOR"

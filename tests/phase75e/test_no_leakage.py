from pathlib import Path


def test_phase75e_contract_has_no_forbidden_model_inputs():
    text = Path("configs/iclr27_phase75e/phase75e_rank8.json").read_text()
    assert "forbidden_inference_inputs" in text
    assert "category" in text and "physical_id" in text and "future" in text

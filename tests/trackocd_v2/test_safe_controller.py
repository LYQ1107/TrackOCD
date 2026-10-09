import numpy as np

from src.trackocd_v2.methods.safe_controller import SafePersistentController


def test_safe_controller_defers_immature_and_zero_features():
    controller = SafePersistentController(known_prototypes={1: np.array([1.0, 0.0])}, min_observations=2)
    zero = controller.step(np.zeros(2), observations=4)
    short = controller.step(np.array([0.0, 1.0]), observations=1)
    assert zero["kind"] == "DEFER"
    assert short["kind"] == "DEFER"
    assert controller._anonymous == {}


def test_safe_controller_reuses_only_a_committed_anonymous_state():
    controller = SafePersistentController(
        known_prototypes={1: np.array([1.0, 0.0])},
        temperature=0.20,
        tau_known=0.99,
        tau_existing=0.20,
        tau_new=0.20,
        min_observations=1,
    )
    first = controller.step(np.array([0.0, 1.0]), observations=4)
    second = controller.step(np.array([0.0, 1.0]), observations=4)
    assert first["kind"] == "NEW"
    assert second["kind"] == "EXISTING"
    assert second["token"] == first["token"]

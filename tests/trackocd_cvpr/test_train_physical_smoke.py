import pytest
from scripts.trackocd_cvpr.smoke_train_physical_inputs import eligible_uuid


def test_no_foreign_GPU_even_when_free_memory_high():
    q = 'GPU-busy, 10, 40000, 0\nGPU-idle, 0, 40000, 0\nGPU-unknown, 0, 40000, [N/A]'
    assert eligible_uuid(q, 'GPU-busy', 12 * 1024**3) == 'GPU-idle'


@pytest.mark.parametrize('q', ['GPU-x, 0, 40000, [N/A]', 'GPU-x, 0, 40000, 20', 'GPU-x, 0, 1000, 0'])
def test_unknown_busy_or_insufficient_GPU_is_resource_wait(q):
    with pytest.raises(RuntimeError, match='RESOURCE_WAIT'):
        eligible_uuid(q, '', 12 * 1024**3)

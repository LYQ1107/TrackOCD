from scripts.trackocd_v2.frontend_execution_preflight import _task_owned


def test_task_owned_only_matches_trackocd_paths():
    assert _task_owned("python /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/scripts/trackocd_v2/foo.py")
    assert not _task_owned("python /data2/other_project/run.py")
    assert not _task_owned(None)

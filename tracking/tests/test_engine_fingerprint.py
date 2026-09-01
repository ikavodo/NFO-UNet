"""A TensorRT engine is NOT portable, and the failure mode is a cryptic deserialize crash.

An engine is compiled for one compute capability, one TensorRT version, one CUDA version. Copy the
.engine from this laptop (sm_120, TRT 11.2) to the Zotac and it will not load - and what TRT prints
is a serialization-mismatch error that says nothing about which of those four things moved. So the
export writes a sidecar fingerprint and this compares it against the live environment, to turn that
crash into a sentence naming the field that changed.

This is the whole reason the export cannot be "prepared for the Zotac" here: only the .pt and the
export command travel. The engine has to be built on the machine that will run it.
"""
from tracking.stream.export_trt import engine_is_stale

ENV = {'compute_cap': '12.0', 'trt': '11.2.1.2', 'cuda': '13.0',
       'imgsz': 640, 'half': True, 'pt_sha': 'abc123'}


def test_identical_environment_is_not_stale():
    assert engine_is_stale(dict(ENV), ENV) is None


def test_missing_fingerprint_is_stale():
    reason = engine_is_stale(None, ENV)
    assert reason and 'fingerprint' in reason.lower()


def test_different_gpu_architecture_names_the_compute_capability():
    reason = engine_is_stale({**ENV, 'compute_cap': '8.6'}, ENV)
    assert reason and '8.6' in reason and '12.0' in reason


def test_different_tensorrt_version_is_stale():
    reason = engine_is_stale({**ENV, 'trt': '10.7.0'}, ENV)
    assert reason and '10.7.0' in reason


def test_changed_source_checkpoint_is_stale():
    # the .pt was replaced or retrained: the engine is now built from weights that no longer exist
    reason = engine_is_stale({**ENV, 'pt_sha': 'deadbeef'}, ENV)
    assert reason and 'checkpoint' in reason.lower()


def test_requesting_a_different_input_size_is_stale():
    # a TRT engine has its input resolution baked in, so imgsz is not a runtime argument
    reason = engine_is_stale({**ENV, 'imgsz': 960}, ENV)
    assert reason and '960' in reason


def test_every_mismatched_field_is_reported_not_just_the_first():
    reason = engine_is_stale({**ENV, 'compute_cap': '8.6', 'trt': '10.7.0'}, ENV)
    assert '8.6' in reason and '10.7.0' in reason

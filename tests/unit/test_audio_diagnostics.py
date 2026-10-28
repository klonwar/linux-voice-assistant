"""Tests for bounded wake-word pipeline diagnostics."""

import logging
import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from linux_voice_assistant.audio_diagnostics import (
    DIAGNOSTIC_BUCKET_COUNT,
    AudioDiagnostics,
    micro_wake_word_activated,
    monitor_audio,
    process_micro_wake_word,
)


def test_micro_wake_word_activation_matches_existing_comparison():
    assert not micro_wake_word_activated(None, 0.9)
    assert not micro_wake_word_activated(0.89, 0.9)
    assert not micro_wake_word_activated(0.9, 0.9)
    assert micro_wake_word_activated(0.91, 0.9)
    assert micro_wake_word_activated(float("nan"), 0.9)


def test_micro_probability_path_invokes_model_once_and_returns_same_score():
    feature_frame = np.array([1])

    class FakeModel:
        calls = 0

        def process_streaming_prob(self, features):
            self.calls += 1
            assert np.array_equal(features, feature_frame)
            return 0.91

    model = FakeModel()

    probability, activated = process_micro_wake_word(model, feature_frame, 0.9)

    assert probability == pytest.approx(0.91)
    assert activated is True
    assert model.calls == 1


@pytest.mark.parametrize(
    ("scenario", "expected_activations", "expected_wakeup_calls", "block_count"),
    [
        ("accepted", {"wakeup_called": 1}, 1, 1),
        ("muted", {"muted": 1}, 0, 1),
        ("refractory", {"wakeup_called": 1, "refractory": 1}, 1, 2),
    ],
)
def test_process_audio_keeps_audio_flow_and_records_activation(
    monkeypatch,
    request,
    scenario,
    expected_activations,
    expected_wakeup_calls,
    block_count,
):
    import importlib

    existing_modules = {name for name in sys.modules if name.startswith("linux_voice_assistant.")}

    def restore_imported_modules():
        for name in list(sys.modules):
            if name.startswith("linux_voice_assistant.") and name not in existing_modules:
                sys.modules.pop(name, None)

    request.addfinalizer(restore_imported_modules)
    mpv_module = ModuleType("mpv")
    mpv_module.MPV = object
    webrtc_module = ModuleType("webrtc_noise_gain")
    webrtc_module.AudioProcessor = object
    monkeypatch.setitem(sys.modules, "mpv", mpv_module)
    monkeypatch.setitem(sys.modules, "webrtc_noise_gain", webrtc_module)
    main_module = importlib.import_module("linux_voice_assistant.__main__")

    class EndAudio(BaseException):
        pass

    class FakeWakeWord:
        def __init__(self, model_id, probability):
            self.id = model_id
            self.probability = probability
            self.calls = []

        def process_streaming_prob(self, features):
            self.calls.append(features.copy())
            return self.probability

    class FakeFeatures:
        def process_streaming(self, audio):
            assert isinstance(audio, bytes)
            return [np.array([1.0], dtype=np.float32)]

    class FakeRecorder:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def record(self, block_size):
            self.calls += 1
            if self.calls > block_count:
                raise EndAudio
            return np.full((block_size, 1), 0.25, dtype=np.float32)

    class FakeMicrophone:
        name = "test microphone"

        def __init__(self):
            self.recorder_instance = FakeRecorder()

        def recorder(self, **_kwargs):
            return self.recorder_instance

    class FakeModel(FakeWakeWord):
        pass

    monkeypatch.setattr(main_module, "MicroWakeWord", FakeModel)
    monkeypatch.setattr(main_module, "MicroWakeWordFeatures", FakeFeatures)

    from tests.unit.conftest import make_state

    wake_word = FakeModel("okay_nabu", 0.95)
    stop_word = FakeModel("stop", 0.0)
    satellite = SimpleNamespace(
        _is_streaming_audio=False,
        _pipeline_active=False,
        handle_audio=Mock(),
        send_messages=Mock(),
        wakeup=Mock(),
        state=None,
    )
    state = make_state(
        wake_words={wake_word.id: wake_word},
        active_wake_words={wake_word.id},
        stop_word=stop_word,
        audio_input_channels=1,
        wake_word_1_threshold=0.9,
        satellite=satellite,
    )
    state.wake_words_changed = True
    state.muted = scenario == "muted"
    state.available_wake_words = {wake_word.id: SimpleNamespace(probability_cutoff=0.9)}
    satellite.state = state
    diagnostics = AudioDiagnostics()

    with pytest.raises(EndAudio):
        main_module.process_audio(state, FakeMicrophone(), 16, diagnostics)

    assert satellite.handle_audio.call_count == block_count
    assert len(wake_word.calls) == block_count
    assert all(np.array_equal(item, np.array([1.0], dtype=np.float32)) for item in wake_word.calls)
    assert satellite.wakeup.call_count == expected_wakeup_calls
    if expected_wakeup_calls:
        satellite.wakeup.assert_called_with(wake_word)
    report = diagnostics.consume_interval(True, state)
    bucket = report["ten_second_buckets"][0]
    assert bucket["models"][wake_word.id]["probability_last"] == pytest.approx(0.95)
    assert bucket["models"][wake_word.id]["activations"] == expected_activations


def test_capture_and_processed_audio_are_aggregated_without_samples():
    diagnostics = AudioDiagnostics()
    diagnostics.record_capture(np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32))
    diagnostics.record_processed_audio(np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes())

    report = diagnostics.consume_interval(True, SimpleNamespace(muted=False, satellite=None))

    assert report["counts"]["capture_blocks"] == 1
    assert report["counts"]["processed_blocks"] == 1
    assert report["ten_second_buckets"][0]["mic"]["rms_mean"] == pytest.approx(0.61237, abs=1e-5)
    assert report["ten_second_buckets"][0]["processed"]["peak_max"] == pytest.approx(0.999969, abs=1e-6)
    assert "samples_data" not in report
    assert all("pcm" not in str(bucket).lower() for bucket in report["ten_second_buckets"])


def test_model_reconfiguration_removes_inactive_model_ids():
    diagnostics = AudioDiagnostics()
    diagnostics.configure_model(SimpleNamespace(id="old_model"), "microWakeWord", 0.9)
    diagnostics.clear_models()
    diagnostics.configure_model(SimpleNamespace(id="okay_nabu"), "microWakeWord", 0.9)

    report = diagnostics.consume_interval(True, SimpleNamespace(muted=False, satellite=None))

    assert list(report["models"]) == ["okay_nabu"]


def test_nonfinite_audio_and_model_outputs_are_counted_not_serialized():
    diagnostics = AudioDiagnostics()
    diagnostics.record_capture(np.array([0.0, float("nan"), float("inf")], dtype=np.float32))
    diagnostics.record_probability("okay_nabu", float("nan"), 0.9)
    diagnostics.record_probability("okay_nabu", None, 0.9)

    report = diagnostics.consume_interval(True, SimpleNamespace(muted=False, satellite=None))
    bucket = report["ten_second_buckets"][0]

    assert report["counts"]["mic_nonfinite_samples"] == 2
    assert report["counts"]["nonfinite_results"] == 1
    assert report["counts"]["none_results"] == 1
    assert bucket["models"]["okay_nabu"]["probability_max"] is None
    assert math.isfinite(report["interval_seconds"])


def test_summary_resets_interval_aggregates_and_bounds_bucket_count(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("linux_voice_assistant.audio_diagnostics.time.monotonic", lambda: clock[0])
    diagnostics = AudioDiagnostics()

    for _ in range(DIAGNOSTIC_BUCKET_COUNT + 5):
        clock[0] += 10.0
        diagnostics.record_capture(np.array([0.25], dtype=np.float32))

    first = diagnostics.consume_interval(True, SimpleNamespace(muted=False, satellite=None))
    second = diagnostics.consume_interval(True, SimpleNamespace(muted=False, satellite=None))

    assert len(first["ten_second_buckets"]) == DIAGNOSTIC_BUCKET_COUNT
    assert first["counts"]["capture_blocks"] == DIAGNOSTIC_BUCKET_COUNT + 5
    assert second["counts"]["capture_blocks"] == 0
    assert second["ten_second_buckets"] == []


def test_health_snapshot_identifies_blocked_stage_and_last_microphone_block(monkeypatch):
    diagnostics = AudioDiagnostics()
    clock = [100.0]
    monkeypatch.setattr("linux_voice_assistant.audio_diagnostics.time.monotonic", lambda: clock[0])
    diagnostics.set_stage("microphone.record")
    diagnostics.record_capture(np.array([0.1], dtype=np.float32))
    clock[0] += 12.0

    health = diagnostics.health_snapshot()

    assert health["stage"] == "microphone.record"
    assert health["stage_age_seconds"] == pytest.approx(12.0)
    assert health["last_block_age_seconds"] == pytest.approx(12.0)


def test_watchdog_logs_a_worker_stall_once(monkeypatch, caplog):
    diagnostics = AudioDiagnostics()
    diagnostics.set_stage("microphone.record")
    stage_started_at = diagnostics.health_snapshot()["stage_started_at"]
    monkeypatch.setattr("linux_voice_assistant.audio_diagnostics.time.monotonic", lambda: stage_started_at + 11)
    monkeypatch.setattr("linux_voice_assistant.audio_diagnostics.DIAGNOSTIC_INTERVAL_SECONDS", 300)

    checks = [0]

    def wait_once(timeout):
        assert timeout == 1.0
        checks[0] += 1
        return checks[0] > 1

    event = SimpleNamespace(wait=wait_once)
    worker = SimpleNamespace(is_alive=lambda: True)

    with caplog.at_level(logging.ERROR):
        monitor_audio(
            diagnostics,
            worker,
            SimpleNamespace(muted=False, satellite=None),
            event,
        )

    assert sum("processing stage stalled" in record.message for record in caplog.records) == 1


def test_watchdog_logs_recovery_and_waits_for_five_minute_summary(monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr("linux_voice_assistant.audio_diagnostics.time.monotonic", lambda: clock[0])
    diagnostics = AudioDiagnostics()
    diagnostics.set_stage("microphone.record")
    monkeypatch.setattr("linux_voice_assistant.audio_diagnostics.DIAGNOSTIC_INTERVAL_SECONDS", 300)
    stage_started_at = diagnostics.health_snapshot()["stage_started_at"]
    checks = [0]

    def wait_until_summary(timeout):
        assert timeout == 1.0
        checks[0] += 1
        if checks[0] == 1:
            clock[0] = stage_started_at + 11
            return False
        if checks[0] == 2:
            diagnostics.set_stage("micro_wake_word_inference:okay_nabu")
            clock[0] += 1
            return False
        if clock[0] >= stage_started_at + 300:
            return True
        diagnostics.set_stage("micro_wake_word_inference:okay_nabu")
        clock[0] += 1
        return False

    event = SimpleNamespace(wait=wait_until_summary)
    worker = SimpleNamespace(is_alive=lambda: True)

    with caplog.at_level(logging.INFO):
        monitor_audio(diagnostics, worker, SimpleNamespace(muted=False, satellite=None), event)

    assert sum("processing stage stalled" in record.message for record in caplog.records) == 1
    assert sum("processing stage recovered" in record.message for record in caplog.records) == 1
    assert sum("Audio diagnostics summary:" in record.message for record in caplog.records) == 1


def test_repeated_processing_error_is_logged_once_until_recovery():
    diagnostics = AudioDiagnostics()
    error = RuntimeError("recorder failed")

    assert diagnostics.record_error("microphone.record", error)
    assert not diagnostics.record_error("microphone.record", error)
    diagnostics.stage_completed()
    assert diagnostics.record_error("microphone.record", error)

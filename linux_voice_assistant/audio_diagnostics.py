"""Low-volume diagnostics for the microphone and wake-word pipeline."""

import importlib.metadata
import json
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

DIAGNOSTIC_INTERVAL_SECONDS = 300.0
DIAGNOSTIC_BUCKET_SECONDS = 10.0
DIAGNOSTIC_BUCKET_COUNT = int(DIAGNOSTIC_INTERVAL_SECONDS / DIAGNOSTIC_BUCKET_SECONDS)
AUDIO_STAGE_STALL_SECONDS = 10.0
_LOGGER = logging.getLogger(__name__)


def micro_wake_word_activated(probability: Optional[float], threshold: float) -> bool:
    """Apply the same probability comparison as MicroWakeWord.process_streaming."""
    # Preserve the wrapper's existing behavior for non-finite model results.
    return probability is not None and not probability <= threshold


def process_micro_wake_word(model: Any, features: np.ndarray, threshold: float) -> tuple[Optional[float], bool]:
    """Run one probability inference and return its score and activation result."""
    probability = model.process_streaming_prob(features)
    return probability, micro_wake_word_activated(probability, threshold)


class AudioDiagnostics:
    """Keep bounded audio and model aggregates for periodic watchdog reports."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._period_started_at = self._started_at
        self._stage = "starting"
        self._stage_started_at = self._started_at
        self._last_progress_at = self._started_at
        self._last_block_at: Optional[float] = None
        self._last_completed_stage: Optional[str] = None
        self._last_error: Optional[dict[str, Any]] = None
        self._active_error_signature: Optional[tuple[str, str, str]] = None
        self._models: dict[str, dict[str, Any]] = {}
        self._audio_configuration: dict[str, Any] = {}
        self._buckets: deque[dict[str, Any]] = deque(maxlen=DIAGNOSTIC_BUCKET_COUNT)
        self._counts = self._new_counts()
        self._bucket_index = -1

        try:
            self._micro_wake_word_version = importlib.metadata.version("pymicro-wakeword")
        except importlib.metadata.PackageNotFoundError:
            self._micro_wake_word_version = "unknown"

    def set_stage(self, stage: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._stage = stage
            self._stage_started_at = now

    def stage_completed(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_completed_stage = self._stage
            self._last_progress_at = now
            self._active_error_signature = None

    def configure_model(self, model: Any, model_type: str, threshold: float) -> None:
        model_id = str(getattr(model, "id", "unknown"))
        info: dict[str, Any] = {
            "type": model_type,
            "threshold": self._serialize_number(float(threshold)),
        }
        if model_type == "microWakeWord":
            interpreter_present = getattr(model, "interpreter", None) is not None
            model_present = getattr(model, "model", None) is not None
            info["interpreter_present"] = interpreter_present
            info["model_present"] = model_present
            info["loaded"] = interpreter_present and model_present
        else:
            info["loaded"] = True

        with self._lock:
            self._models[model_id] = info

    def clear_models(self) -> None:
        """Discard the previous active model set before it is reconfigured."""
        with self._lock:
            self._models.clear()

    def configure_audio(self, configuration: dict[str, Any]) -> None:
        with self._lock:
            self._audio_configuration = dict(configuration)

    def record_capture(self, samples: np.ndarray) -> None:
        stats = self._sample_stats(samples, float_samples=True)
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            bucket["capture_blocks"] += 1
            self._update_audio_stats(bucket["mic"], stats)
            self._counts["capture_blocks"] += 1
            if self._last_block_at is not None:
                interval = now - self._last_block_at
                bucket["block_interval_last_seconds"] = round(interval, 4)
                bucket["block_interval_min_seconds"] = min(bucket["block_interval_min_seconds"], interval)
                bucket["block_interval_max_seconds"] = max(bucket["block_interval_max_seconds"], interval)
                self._counts["capture_interval_last_seconds"] = round(interval, 4)
                self._counts["capture_interval_max_seconds"] = max(self._counts["capture_interval_max_seconds"], interval)
            self._counts["mic_nonfinite_samples"] += stats["nonfinite"]
            self._counts["mic_clipped_samples"] += stats["clipped"]
            if stats["peak"] == 0:
                self._counts["mic_zero_blocks"] += 1
            self._last_block_at = now
            self._last_progress_at = now

    def record_processed_audio(self, audio: bytes) -> None:
        stats = self._sample_stats(np.frombuffer(audio, dtype="<i2"), float_samples=False)
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            bucket["processed_blocks"] += 1
            self._update_audio_stats(bucket["processed"], stats)
            self._counts["processed_blocks"] += 1
            if stats["samples"] == 0 or stats["peak"] == 0:
                self._counts["processed_zero_blocks"] += 1
            self._counts["processed_nonfinite_samples"] += stats["nonfinite"]
            self._counts["processed_clipped_samples"] += stats["clipped"]

    def record_empty_processed_audio(self) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            bucket["processed_empty_blocks"] += 1
            self._counts["processed_empty_blocks"] += 1

    def record_features(self, features: np.ndarray) -> None:
        stats = self._feature_stats(features)
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            bucket["feature_frames"] += 1
            self._update_range(bucket["features"], stats)
            self._counts["feature_frames"] += 1
            self._counts["feature_nonfinite_values"] += stats["nonfinite"]

    def record_probability(self, model_id: str, probability: Optional[float], threshold: float) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            model_stats = bucket["models"].setdefault(model_id, self._new_model_stats())
            model_stats["probability_calls"] += 1
            self._counts["probability_calls"] += 1

            if probability is None:
                model_stats["none_results"] += 1
                self._counts["none_results"] += 1
                return

            value = float(probability)
            if not math.isfinite(value):
                model_stats["nonfinite_results"] += 1
                self._counts["nonfinite_results"] += 1
                return

            model_stats["probability_count"] += 1
            model_stats["probability_min"] = min(model_stats["probability_min"], value)
            model_stats["probability_max"] = max(model_stats["probability_max"], value)
            model_stats["probability_sum"] += value
            model_stats["probability_last"] = value
            model_stats["threshold"] = self._serialize_number(float(threshold))
            if value > threshold:
                model_stats["above_threshold"] += 1

    def record_activation(self, model_id: str, outcome: str) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            activations = bucket["models"].setdefault(model_id, self._new_model_stats())["activations"]
            activations[outcome] = activations.get(outcome, 0) + 1

    def record_error(self, stage: str, error: BaseException) -> bool:
        now = time.monotonic()
        message = str(error)[:240]
        signature = (stage, type(error).__name__, message)
        with self._lock:
            self._last_error = {
                "stage": stage,
                "type": type(error).__name__,
                "message": message,
            }
            self._counts["processing_errors"] += 1
            self._last_progress_at = now
            should_log = signature != self._active_error_signature
            self._active_error_signature = signature
            return should_log

    def health_snapshot(self, now: Optional[float] = None) -> dict[str, Any]:
        snapshot_at = time.monotonic() if now is None else now
        with self._lock:
            return {
                "stage": self._stage,
                "stage_started_at": self._stage_started_at,
                "stage_age_seconds": max(0.0, snapshot_at - self._stage_started_at),
                "last_progress_age_seconds": max(0.0, snapshot_at - self._last_progress_at),
                "last_block_age_seconds": None if self._last_block_at is None else max(0.0, snapshot_at - self._last_block_at),
                "last_completed_stage": self._last_completed_stage,
                "last_error": self._last_error,
            }

    def consume_interval(self, worker_alive: bool, state: Any) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            counts = dict(self._counts)
            models = self._models
            buckets = [self._serialize_bucket(bucket) for bucket in self._buckets]
            report = {
                "pymicro_wakeword_version": self._micro_wake_word_version,
                "interval_seconds": round(now - self._period_started_at, 2),
                "uptime_seconds": round(now - self._started_at, 2),
                "worker_alive": worker_alive,
                "stage": self._stage,
                "stage_age_seconds": round(max(0.0, now - self._stage_started_at), 2),
                "last_progress_age_seconds": round(max(0.0, now - self._last_progress_at), 2),
                "last_block_age_seconds": None if self._last_block_at is None else round(max(0.0, now - self._last_block_at), 2),
                "last_completed_stage": self._last_completed_stage,
                "audio_configuration": dict(self._audio_configuration),
                "muted": bool(state.muted),
                "pipeline_active": getattr(state.satellite, "_pipeline_active", None),
                "streaming_audio": getattr(state.satellite, "_is_streaming_audio", None),
                "counts": counts,
                "models": {model_id: dict(info) for model_id, info in models.items()},
                "ten_second_buckets": buckets,
                "last_error": self._last_error,
            }
            self._period_started_at = now
            self._counts = self._new_counts()
            self._buckets.clear()
            self._bucket_index = -1
            return report

    def _current_bucket(self, now: float) -> dict[str, Any]:
        index = min(
            DIAGNOSTIC_BUCKET_COUNT - 1,
            max(0, int((now - self._period_started_at) / DIAGNOSTIC_BUCKET_SECONDS)),
        )
        while self._bucket_index < index:
            self._bucket_index += 1
            self._buckets.append(self._new_bucket(self._bucket_index))
        return self._buckets[-1]

    @staticmethod
    def _new_counts() -> dict[str, Any]:
        return {
            "capture_blocks": 0,
            "capture_interval_last_seconds": None,
            "capture_interval_max_seconds": 0.0,
            "mic_zero_blocks": 0,
            "mic_nonfinite_samples": 0,
            "mic_clipped_samples": 0,
            "processed_blocks": 0,
            "processed_empty_blocks": 0,
            "processed_zero_blocks": 0,
            "processed_nonfinite_samples": 0,
            "processed_clipped_samples": 0,
            "feature_frames": 0,
            "feature_nonfinite_values": 0,
            "probability_calls": 0,
            "none_results": 0,
            "nonfinite_results": 0,
            "processing_errors": 0,
        }

    @staticmethod
    def _new_bucket(index: int) -> dict[str, Any]:
        return {
            "offset_seconds": index * int(DIAGNOSTIC_BUCKET_SECONDS),
            "capture_blocks": 0,
            "block_interval_last_seconds": None,
            "block_interval_min_seconds": math.inf,
            "block_interval_max_seconds": 0.0,
            "processed_blocks": 0,
            "processed_empty_blocks": 0,
            "feature_frames": 0,
            "mic": AudioDiagnostics._new_audio_stats(),
            "processed": AudioDiagnostics._new_audio_stats(),
            "features": AudioDiagnostics._new_range(),
            "models": {},
        }

    @staticmethod
    def _new_audio_stats() -> dict[str, Any]:
        return {
            "blocks": 0,
            "samples": 0,
            "rms_sum": 0.0,
            "rms_max": 0.0,
            "peak_max": 0.0,
            "zero_blocks": 0,
            "nonfinite": 0,
            "clipped": 0,
        }

    @staticmethod
    def _new_range() -> dict[str, Any]:
        return {"count": 0, "min": None, "max": None}

    @staticmethod
    def _new_model_stats() -> dict[str, Any]:
        return {
            "probability_calls": 0,
            "probability_count": 0,
            "none_results": 0,
            "nonfinite_results": 0,
            "probability_min": math.inf,
            "probability_max": -math.inf,
            "probability_sum": 0.0,
            "probability_last": None,
            "threshold": None,
            "above_threshold": 0,
            "activations": {},
        }

    @staticmethod
    def _sample_stats(samples: np.ndarray, float_samples: bool) -> dict[str, Any]:
        values = np.asarray(samples).reshape(-1)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        nonfinite = int(values.size - finite_values.size)
        if finite_values.size:
            numeric = finite_values.astype(np.float64, copy=False)
            scale = 1.0 if float_samples else 32768.0
            normalized = numeric / scale
            rms = float(np.sqrt(np.mean(np.square(normalized))))
            peak = float(np.max(np.abs(normalized)))
            clipped = int(np.count_nonzero(np.abs(normalized) >= 0.999))
        else:
            rms = 0.0
            peak = 0.0
            clipped = 0
        return {
            "samples": int(values.size),
            "rms": rms,
            "peak": peak,
            "nonfinite": nonfinite,
            "clipped": clipped,
        }

    @staticmethod
    def _feature_stats(features: np.ndarray) -> dict[str, Any]:
        values = np.asarray(features).reshape(-1)
        finite = values[np.isfinite(values)]
        return {
            "count": int(values.size),
            "nonfinite": int(values.size - finite.size),
            "min": None if not finite.size else float(np.min(finite)),
            "max": None if not finite.size else float(np.max(finite)),
        }

    @staticmethod
    def _update_audio_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
        target["blocks"] += 1
        target["samples"] += source["samples"]
        target["rms_sum"] += source["rms"]
        target["rms_max"] = max(target["rms_max"], source["rms"])
        target["peak_max"] = max(target["peak_max"], source["peak"])
        target["nonfinite"] += source["nonfinite"]
        target["clipped"] += source["clipped"]
        if source["peak"] == 0:
            target["zero_blocks"] += 1

    @staticmethod
    def _update_range(target: dict[str, Any], source: dict[str, Any]) -> None:
        if source["min"] is None:
            return
        target["count"] += source["count"]
        target["min"] = source["min"] if target["min"] is None else min(target["min"], source["min"])
        target["max"] = source["max"] if target["max"] is None else max(target["max"], source["max"])

    @staticmethod
    def _serialize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in bucket.items() if key not in {"mic", "processed", "features", "models"}}
        if not math.isfinite(result["block_interval_min_seconds"]):
            result["block_interval_min_seconds"] = None
        else:
            result["block_interval_min_seconds"] = round(result["block_interval_min_seconds"], 4)
        result["mic"] = AudioDiagnostics._serialize_audio_stats(bucket["mic"])
        result["processed"] = AudioDiagnostics._serialize_audio_stats(bucket["processed"])
        result["features"] = dict(bucket["features"])
        result["models"] = {}
        for model_id, stats in bucket["models"].items():
            count = stats["probability_count"]
            result["models"][model_id] = {
                "probability_calls": stats["probability_calls"],
                "probability_count": count,
                "none_results": stats["none_results"],
                "nonfinite_results": stats["nonfinite_results"],
                "probability_min": None if not count else round(stats["probability_min"], 6),
                "probability_max": None if not count else round(stats["probability_max"], 6),
                "probability_mean": None if not count else round(stats["probability_sum"] / count, 6),
                "probability_last": None if stats["probability_last"] is None else round(stats["probability_last"], 6),
                "threshold": stats["threshold"],
                "above_threshold": stats["above_threshold"],
                "activations": dict(stats["activations"]),
            }
        return result

    @staticmethod
    def _serialize_audio_stats(stats: dict[str, Any]) -> dict[str, Any]:
        samples = stats["samples"]
        blocks = stats["blocks"]
        return {
            "blocks": blocks,
            "samples": samples,
            "rms_mean": None if not blocks else round(stats["rms_sum"] / blocks, 6),
            "rms_max": round(stats["rms_max"], 6),
            "peak_max": round(stats["peak_max"], 6),
            "zero_blocks": stats["zero_blocks"],
            "nonfinite_samples": stats["nonfinite"],
            "clipped_samples": stats["clipped"],
        }

    @staticmethod
    def _serialize_number(value: float) -> float | str:
        return value if math.isfinite(value) else repr(value)


def monitor_audio(
    diagnostics: AudioDiagnostics,
    worker: threading.Thread,
    state: Any,
    stop_event: threading.Event,
) -> None:
    """Report audio worker stalls and periodic bounded diagnostic summaries."""
    next_summary_at = time.monotonic() + DIAGNOSTIC_INTERVAL_SECONDS
    warned_stage: Optional[tuple[str, float]] = None
    worker_exit_reported = False

    while not stop_event.wait(1.0):
        now = time.monotonic()
        health = diagnostics.health_snapshot(now)

        if not worker.is_alive():
            if not worker_exit_reported:
                _LOGGER.error(
                    "Audio diagnostics: audio worker exited (last_stage=%s, last_block_age=%s, last_error=%s)",
                    health["stage"],
                    health["last_block_age_seconds"],
                    health["last_error"],
                )
                worker_exit_reported = True
        elif health["stage_age_seconds"] >= AUDIO_STAGE_STALL_SECONDS:
            stage_token = (health["stage"], health["stage_started_at"])
            if stage_token != warned_stage:
                _LOGGER.error(
                    "Audio diagnostics: processing stage stalled (stage=%s, stage_age=%.1fs, last_progress_age=%.1fs, last_block_age=%s)",
                    health["stage"],
                    health["stage_age_seconds"],
                    health["last_progress_age_seconds"],
                    health["last_block_age_seconds"],
                )
                warned_stage = stage_token
        elif warned_stage is not None:
            _LOGGER.info(
                "Audio diagnostics: processing stage recovered (previous_stage=%s, current_stage=%s)",
                warned_stage[0],
                health["stage"],
            )
            warned_stage = None

        if now >= next_summary_at:
            try:
                report = diagnostics.consume_interval(worker.is_alive(), state)
                _LOGGER.info("Audio diagnostics summary: %s", json.dumps(report, separators=(",", ":"), allow_nan=False))
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Audio diagnostics: failed to create periodic summary")
            next_summary_at = now + DIAGNOSTIC_INTERVAL_SECONDS

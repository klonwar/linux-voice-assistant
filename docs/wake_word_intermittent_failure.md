# Intermittent wake-word failure: investigation plan

## Goal

Find why the assistant can stop reacting to wake words while the LVA process remains running, then validate one targeted fix at a time on the deployed assistant. The microphone and peripheral hardware are not currently suspected. No application logs or live process are available during the initial code investigation.

Mute/unmute is a recovery workaround reported by the operator. In the current code it does not recreate the microphone recorder, feature processors, or wake-word models. Muting stops TTS playback and may invoke a playback callback that clears `_pipeline_active`; this makes a stuck pipeline gate one hypothesis, not a confirmed root cause.

## Working agreement

1. Add low-volume diagnostic logging without changing thresholds or recovery behavior.
2. The operator deploys the diagnostic build from the fork, waits for the failure, and saves logs covering before, during, and after recovery.
3. Review those logs against the hypotheses below and agree on a specific fix.
4. Test candidate fixes one at a time on the real assistant. Keep each run identifiable by commit/build and record whether the issue recurs and whether the candidate restores recognition.
5. Do not combine a recovery mechanism with diagnostic changes until the failure path is understood. Do not log raw microphone audio.

## Phase 1: diagnostic-only change

Add an independent watchdog and bounded, in-memory aggregates at these boundaries. The watchdog reports every five minutes and checks the audio worker once per second. It emits one warning when a processing stage makes no progress for ten seconds and one recovery event when that stage advances again. Each five-minute report contains at most thirty ten-second aggregate buckets; the interval aggregates are cleared after reporting.

### Audio capture and preprocessing (`linux_voice_assistant/__main__.py`, `linux_voice_assistant/webrtc.py`)

- Log recorder start, sample rate, channel count, block size, and any recorder exit or exception.
- Track successful microphone blocks and normalized RMS/peak, exact-zero, non-finite, and clipped-sample counts.
- Track the same aggregate signal metrics after WebRTC processing and count empty outputs.
- Track feature-frame count and finite-value range from MicroWakeWord feature extraction.
- Report the active processing stage, stage age, last successful microphone-block age, and audio-worker liveness from the separate watchdog.
- Do not log audio samples, encoded audio, transcripts, or other microphone content.

### Wake-word processing (`linux_voice_assistant/__main__.py`)

- Report the `pymicro-wakeword` version, active model IDs/types, loaded interpreter/model state, and thresholds.
- Record probability call/result counts, None and non-finite outputs, min/max/mean/last finite probability, and threshold crossings for both engines. Use MicroWakeWord's public probability API once per feature frame; do not run inference a second time for diagnostics.
- Record activation and suppression outcomes, with the score and threshold on activation.
- When an activation is produced, log whether it was suppressed because of mute, refractory period, or `_pipeline_active`; include the relevant state and elapsed refractory time.
- Record exceptions at feature/model boundaries with stack traces and identify the model/step. Preserve existing exception handling and behavior in this diagnostic-only phase.

### Pipeline and mute lifecycle (`linux_voice_assistant/satellite.py`, `linux_voice_assistant/mpv_player.py`)

- Log state transitions for `_pipeline_active`, `_is_streaming_audio`, and `state.muted`, with the reason/event that changed each value.
- Log the incoming voice-pipeline event type and the resulting state, especially RUN_START, RUN_END, STT_END, STT_VAD_END, TTS_START, and ERROR.
- Log when `wakeup()` is called and whether it starts a pipeline or returns early; identify the guard that rejected it.
- Log mute/unmute entry, TTS player state before/after stop/play, and which completion callback (if any) runs.
- Log callback entry/exit and state changes in wake-up sound completion and `_tts_finished()`.

### Logging constraints

- Keep normal periodic logs to one compact summary every five minutes. Only actual stalls, worker exit/recovery, model changes, and errors generate event logs.
- Buckets contain aggregate metrics only and are overwritten/cleared after each report. Do not retain a per-block series, PCM samples, waveforms, encoded audio, transcripts, or serialized audio-derived time series.
- A five-minute summary that shows low model probabilities can localize failure to signal/model input, but cannot prove what words were spoken or distinguish semantic model rejection from an acoustically degraded signal without recording audio.
- Do not change timeouts, reset models, restart threads, alter thresholds, or otherwise change recognition behavior in this phase.

## Phase 2: real-device observation

The operator deploys the diagnostic-only build from the fork. When recognition fails:

1. Do not immediately mute/unmute; first preserve a short log window before and during the failure.
2. Record approximate failure start time, whether the assistant is idle or handling a pipeline/TTS response, and the active wake-word model(s).
3. Capture logs through one mute/unmute recovery and note the exact recovery time.
4. Save the build/commit identifier and relevant configuration (model type, active wake words, WebRTC AGC/NS settings, input block size). Do not include secrets or raw audio.
5. Share the logs and observations for analysis before choosing a behavioral fix.

If operational constraints require recovery before logs are saved, use mute/unmute and record that the pre-recovery evidence was lost; do not treat the recovery itself as proof of a cause.

## Phase 3: interpret evidence

Use the last progressing boundary and state transitions to distinguish these hypotheses:

| Evidence during failure | Candidate failure area |
| --- | --- |
| Microphone block counter stops advancing | Recorder read or audio-capture thread stalled/exited |
| Capture advances, but WebRTC output stops | WebRTC processor or its buffering path |
| Audio/WebRTC advances, but feature frames stop | Feature extraction processor stalled or not yielding frames |
| Feature frames and inference continue, but scores become abnormal or frozen | Wake-word model streaming state or its input |
| Model activation occurs, but `wakeup()` is rejected with `_pipeline_active=True` | Pipeline completion/state transition failed to clear the gate |
| Activation is rejected because muted/refractory state is wrong | Mute or refractory state path |
| Capture and features progress but no activation occurs | Compare the ten-second signal and probability buckets. Low/no scores despite energetic input localize the failure to the feature/model-input path; healthy expected scores with no accepted wake-up point to recognition gating. The logs cannot verify speech content. |

An absent log line alone is not enough to identify a failure. Confirm that the relevant periodic heartbeat was expected and that the logging process itself remained alive.

## Phase 4: sequential experiments

After reviewing the captured evidence, select only the experiment that targets the supported hypothesis. Keep diagnostics in place for comparison and change one behavior at a time.

1. **Pipeline gate experiment:** if activations are observed but blocked by a stale `_pipeline_active`, implement a narrowly scoped recovery for a demonstrably incomplete pipeline. Validate that normal long-running responses and TTS are not interrupted.
2. **Feature-state experiment:** if capture continues but feature extraction stops or becomes stale, test resetting/recreating only the relevant feature processor, using the installed library's supported lifecycle API.
3. **Wake-model-state experiment:** if feature frames continue but model scores indicate stale inference state, test recreating the affected model and its dependent feature state. Avoid resetting unrelated models.
4. **Audio-thread experiment:** if capture or processing stops advancing, test a bounded restart of the specific failed stage, with clear failure logging and protection against restart loops.

For each experiment, record: hypothesis, minimal code change, build/commit, observation duration, whether the failure recurred, whether the recovery trigger fired, and any side effects. If a candidate does not change the observed failure path or the issue recurs unchanged, revert/retire that hypothesis before trying another.

## Current code references

- Audio capture and wake-word loop: `linux_voice_assistant/__main__.py`, `process_audio()`.
- Mute side effects and pipeline events: `linux_voice_assistant/satellite.py`, `_set_muted()` and `handle_voice_event()`.
- Wake-up gate and completion callbacks: `linux_voice_assistant/satellite.py`, `wakeup()`, `_on_wakeup_sound_finished()`, `_tts_finished()`.
- TTS stop/callback behavior: `linux_voice_assistant/mpv_player.py` and `linux_voice_assistant/player/libmpv.py`.
- Audio stack background: [Audio debugging](debugging.md).

## Verification boundaries

Unit tests and static checks can verify logging does not alter control flow and that state transitions are reported. They cannot establish that microphone hardware, the deployed audio stack, or the intermittent failure is fixed. Each behavioral experiment requires observation on the real assistant. No hardware test has been performed as part of writing this plan.

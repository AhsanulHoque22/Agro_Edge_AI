"""
Continuous runtime scheduler for AgroEdge edge decision loop.

Features:
  - fixed-interval execution
  - retry with exponential backoff for fetch/publish failures
  - local JSONL cycle logging for observability and offline debugging
  - graceful shutdown on SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from edge_inference.inference_engine import DecisionResult
from edge_inference.runtime_loop import RuntimeContext, run_one_cycle


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0


@dataclass
class SchedulerConfig:
    interval_seconds: float = 900.0
    publish_log: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    max_cycles: int | None = None


class RuntimeScheduler:
    """Executes the runtime decision cycle continuously at a fixed interval."""

    def __init__(
        self,
        config: SchedulerConfig,
        runtime_context: RuntimeContext,
        node_id: str,
        engine,
        fetch_telemetry_fn: Callable[[], dict[str, Any]],
        publish_log_fn: Callable[[dict[str, Any]], int] | None,
        local_log_path: Path,
    ) -> None:
        self.config = config
        self.runtime_context = runtime_context
        self.node_id = node_id
        self.engine = engine
        self.fetch_telemetry_fn = fetch_telemetry_fn
        self.publish_log_fn = publish_log_fn
        self.local_log_path = local_log_path
        self._stop_requested = False

    def install_signal_handlers(self) -> None:
        """Install graceful shutdown handlers for Ctrl+C and terminate."""

        def _handler(signum, _frame) -> None:  # type: ignore[no-untyped-def]
            self._stop_requested = True
            print(f"\nReceived signal {signum}. Stopping after current cycle...")

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _sleep_with_interrupt(self, seconds: float) -> None:
        """Sleep in short steps to respond quickly to stop requests."""
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            if self._stop_requested:
                return
            time.sleep(min(0.25, end - time.time()))

    def _with_retry(self, operation_name: str, fn: Callable[[], Any]) -> Any:
        """Execute fn with retry/backoff according to scheduler retry policy."""
        policy = self.config.retry_policy
        last_exc: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - intentional boundary
                last_exc = exc
                if attempt >= policy.max_attempts:
                    break
                delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2 ** (attempt - 1)))
                print(
                    f"{operation_name} failed on attempt {attempt}/{policy.max_attempts}: {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                self._sleep_with_interrupt(delay)
                if self._stop_requested:
                    break
        raise RuntimeError(f"{operation_name} failed after {policy.max_attempts} attempts") from last_exc

    def _append_local_log(self, row: dict[str, Any]) -> None:
        self.local_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _build_log_row(
        self,
        cycle_index: int,
        telemetry: dict[str, Any] | None,
        feature_payload: dict[str, Any] | None,
        decision: DecisionResult | None,
        action_payload: dict[str, Any] | None,
        status: str,
        error: str | None,
        publish_entry_id: int | None,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        return {
            "cycle_index": cycle_index,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "error": error,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "node_id": self.node_id,
            "telemetry": telemetry,
            "feature_payload": feature_payload,
            "decision": (
                None
                if decision is None
                else {
                    "should_irrigate": decision.should_irrigate,
                    "approved_duration_minutes": decision.approved_duration_minutes,
                    "blocked_reason": decision.blocked_reason,
                    "model_probability": decision.model_prediction.irrigation_probability,
                    "model_predicted_duration": decision.model_prediction.irrigation_duration_minutes,
                    "model_version": decision.model_version,
                    "model_probability_bin": _probability_bin(
                        decision.model_prediction.irrigation_probability
                    ),
                }
            ),
            "action_payload": action_payload,
            "publish_entry_id": publish_entry_id,
            "monitoring": (telemetry or {}).get("_runtime_monitoring", {}),
        }

    def run(self) -> None:
        """
        Run scheduler loop until stop requested or max_cycles reached.
        """
        self.install_signal_handlers()
        cycle = 0
        print(
            f"Starting runtime scheduler: interval={self.config.interval_seconds}s "
            f"publish_log={self.config.publish_log} max_cycles={self.config.max_cycles}"
        )
        while not self._stop_requested:
            if self.config.max_cycles is not None and cycle >= self.config.max_cycles:
                print(f"Reached max_cycles={self.config.max_cycles}. Stopping scheduler.")
                break

            cycle += 1
            started = time.time()
            telemetry: dict[str, Any] | None = None
            feature_payload: dict[str, Any] | None = None
            decision: DecisionResult | None = None
            action_payload: dict[str, Any] | None = None
            publish_entry_id: int | None = None
            status = "ok"
            error: str | None = None

            try:
                telemetry = self._with_retry("telemetry_fetch", self.fetch_telemetry_fn)
                feature_payload, decision, action_payload = run_one_cycle(
                    engine=self.engine,
                    telemetry_row=telemetry,
                    context=self.runtime_context,
                    node_id=self.node_id,
                )

                if self.config.publish_log and self.publish_log_fn is not None:
                    publish_entry_id = self._with_retry(
                        "log_publish", lambda: self.publish_log_fn(action_payload)
                    )

                print(
                    f"[cycle {cycle}] should_irrigate={decision.should_irrigate} "
                    f"duration={decision.approved_duration_minutes:.2f} "
                    f"blocked_reason={decision.blocked_reason}"
                )
            except Exception as exc:  # noqa: BLE001 - log and continue loop
                status = "error"
                error = str(exc)
                print(f"[cycle {cycle}] ERROR: {error}")

            elapsed = time.time() - started
            self._append_local_log(
                self._build_log_row(
                    cycle_index=cycle,
                    telemetry=telemetry,
                    feature_payload=feature_payload,
                    decision=decision,
                    action_payload=action_payload,
                    status=status,
                    error=error,
                    publish_entry_id=publish_entry_id,
                    elapsed_seconds=elapsed,
                )
            )

            if self._stop_requested:
                break
            sleep_time = max(0.0, self.config.interval_seconds - elapsed)
            self._sleep_with_interrupt(sleep_time)


def _probability_bin(probability: float) -> str:
    p = float(probability)
    if p < 0.2:
        return "0.0-0.2"
    if p < 0.4:
        return "0.2-0.4"
    if p < 0.6:
        return "0.4-0.6"
    if p < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"

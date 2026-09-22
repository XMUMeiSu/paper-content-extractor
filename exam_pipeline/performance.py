"""Run-local performance telemetry for visual model requests."""
from __future__ import annotations

import copy
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence


class PerformanceCollector:
    """Collect bounded request facts without coupling pipeline stages together."""

    def __init__(self) -> None:
        self._events = []
        self._document_id: Optional[str] = None
        self._lock = threading.Lock()

    def set_document(self, document_id: Optional[str]) -> None:
        self._document_id = document_id

    def record(self, *, stage: str, provider: str, operation: str,
               duration_seconds: float, status: str = "SUCCESS",
               paths: Sequence[Path] = (), attempts: int = 1,
               retry_reasons: Sequence[str] = (), cache_hit: bool = False,
               usage: Optional[Dict[str, Any]] = None,
               error_type: Optional[str] = None, **context: Any) -> None:
        image_paths = [Path(path) for path in paths]
        image_bytes = 0
        for path in image_paths:
            try:
                image_bytes += path.stat().st_size
            except OSError:
                pass
        event = {
            "document_id": self._document_id,
            "stage": stage,
            "provider": provider,
            "operation": operation,
            "duration_seconds": round(max(0.0, float(duration_seconds)), 4),
            "image_count": len(image_paths),
            "image_bytes": int(image_bytes),
            "attempts": max(1, int(attempts or 1)),
            "retry_reasons": list(retry_reasons or ()),
            "cache_hit": bool(cache_hit),
            "usage": copy.deepcopy(usage) if isinstance(usage, dict) else usage,
            "status": status,
            "error_type": error_type,
        }
        event.update({key: value for key, value in context.items() if value is not None})
        with self._lock:
            self._events.append(event)

    def instrument(self, request: Callable, *, stage: str, provider: str,
                   operation: str) -> Callable:
        """Wrap a ``(prompt, images, schema)`` model adapter."""
        def measured(prompt, paths, schema):
            started = time.monotonic()
            try:
                result = request(prompt, paths, schema)
            except Exception as exc:
                audit = getattr(exc, "response_audit", {}) or {}
                self.record(
                    stage=stage, provider=provider, operation=operation,
                    duration_seconds=time.monotonic() - started, paths=paths,
                    attempts=audit.get("attempts", 1),
                    retry_reasons=audit.get("retry_reasons", ()),
                    usage=audit.get("usage"), status="FAILED",
                    error_type=type(exc).__name__,
                )
                raise
            audit = getattr(result, "response_audit", {}) or {}
            self.record(
                stage=stage, provider=provider, operation=operation,
                duration_seconds=time.monotonic() - started, paths=paths,
                attempts=audit.get("attempts", 1),
                retry_reasons=audit.get("retry_reasons", ()),
                usage=audit.get("usage"), status="SUCCESS",
            )
            return result
        return measured

    def summary(self, document_id: Optional[str] = None, include_events: bool = True) -> Dict[str, Any]:
        with self._lock:
            events = [copy.deepcopy(event) for event in self._events
                      if document_id is None or event.get("document_id") == document_id]
        operations = defaultdict(lambda: {
            "request_count": 0, "duration_seconds": 0.0, "image_count": 0,
            "image_bytes": 0, "attempts": 0, "failures": 0, "cache_hits": 0,
        })
        for event in events:
            bucket = operations[event["operation"]]
            bucket["request_count"] += 1
            bucket["duration_seconds"] += event["duration_seconds"]
            bucket["image_count"] += event["image_count"]
            bucket["image_bytes"] += event["image_bytes"]
            bucket["attempts"] += event["attempts"]
            bucket["failures"] += event["status"] != "SUCCESS"
            bucket["cache_hits"] += bool(event["cache_hit"])
        aggregates = {}
        for name, bucket in sorted(operations.items()):
            bucket["duration_seconds"] = round(bucket["duration_seconds"], 4)
            bucket["cache_hit_rate"] = round(
                bucket["cache_hits"] / max(1, bucket["request_count"]), 4)
            aggregates[name] = bucket
        response = {
            "request_count": len(events),
            # This is a sum of operation durations. Nested model calls
            # can overlap, so document wall time remains authoritative in the
            # manifest and phase trace.
            "cumulative_operation_seconds": round(
                sum(event["duration_seconds"] for event in events), 4),
            "failures": sum(event["status"] != "SUCCESS" for event in events),
            "operations": aggregates,
        }
        if include_events:
            response["events"] = events
        return response


_ACTIVE_COLLECTOR: Optional[PerformanceCollector] = None


def set_active_collector(collector: Optional[PerformanceCollector]) -> None:
    global _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector


def get_active_collector() -> Optional[PerformanceCollector]:
    return _ACTIVE_COLLECTOR


__all__ = ["PerformanceCollector", "get_active_collector", "set_active_collector"]

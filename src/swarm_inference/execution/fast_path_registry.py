"""Fast-path discovery, exactness admission, caching, and measured selection."""

from __future__ import annotations

from typing import Literal

from swarm_inference.engines.interfaces import ExecutionDevice
from swarm_inference.execution.fast_path import (
    FastPathMeasurement,
    NativeFastPath,
    PreparedFastPath,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.runtime.performance_profiles import (
    FastPathProfile,
    FastPathProfileKey,
    FastPathProfileStore,
)


class FastPathAdmissionError(RuntimeError):
    pass


class NativeFastPathRegistry:
    def __init__(self, paths: tuple[NativeFastPath, ...] = ()) -> None:
        self._paths: dict[str, NativeFastPath] = {}
        for path in paths:
            self.register(path)

    def register(self, path: NativeFastPath, *, replace: bool = False) -> None:
        if path.fast_path_id in self._paths and not replace:
            raise ValueError(f"fast path {path.fast_path_id!r} is already registered")
        self._paths[path.fast_path_id] = path

    def get(self, fast_path_id: str) -> NativeFastPath:
        try:
            return self._paths[fast_path_id]
        except KeyError as exc:
            raise KeyError(f"fast path {fast_path_id!r} is not registered") from exc

    @staticmethod
    def _select(
        measurements: tuple[FastPathMeasurement, ...],
        objective: Literal["speed", "throughput", "capacity", "balanced"],
    ) -> FastPathMeasurement:
        eligible = [item for item in measurements if item.eligible]
        if not eligible:
            failures = "; ".join(
                f"{item.candidate_mode}: {item.failure_reason or 'exactness failed'}"
                for item in measurements
            )
            raise FastPathAdmissionError(f"no exact fast-path candidate is eligible; {failures}")
        if objective == "capacity":
            return min(
                eligible,
                key=lambda item: (item.memory_bytes, -item.decode_tokens_s, item.candidate_mode),
            )
        if objective == "speed":
            return max(
                eligible,
                key=lambda item: (item.decode_tokens_s, -item.ttft_ms, item.candidate_mode),
            )
        if objective == "throughput":
            return max(eligible, key=lambda item: (item.decode_tokens_s, item.candidate_mode))
        return max(
            eligible,
            key=lambda item: (
                item.decode_tokens_s / max(1.0, item.ttft_ms),
                -item.memory_bytes,
                item.candidate_mode,
            ),
        )

    def admit(
        self,
        *,
        fast_path_id: str,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
        profile_key: FastPathProfileKey,
        profile_store: FastPathProfileStore,
        objective: Literal["speed", "throughput", "capacity", "balanced"],
        benchmark_kwargs: dict[str, object] | None = None,
        prepare_kwargs: dict[str, object] | None = None,
    ) -> PreparedFastPath:
        path = self.get(fast_path_id)
        report = path.probe(model, device)
        if not report.supported:
            raise FastPathAdmissionError(f"{report.status.value}: {report.reason}")
        cached = profile_store.get(profile_key)
        if cached is None:
            measurements = tuple(
                path.benchmark_candidates(model, device, **(benchmark_kwargs or {}))
            )
            if not measurements:
                raise FastPathAdmissionError("fast-path benchmark returned no candidates")
            if len({item.candidate_mode for item in measurements}) != len(measurements):
                raise FastPathAdmissionError("fast-path benchmark returned duplicate candidates")
            for measurement in measurements:
                if measurement.fast_path_id != fast_path_id:
                    raise FastPathAdmissionError(
                        "fast-path benchmark returned evidence for a different implementation"
                    )
                if (
                    measurement.batch_bucket != profile_key.batch_bucket
                    or measurement.context_bucket != profile_key.context_bucket
                ):
                    raise FastPathAdmissionError(
                        "fast-path benchmark evidence does not match the requested shape buckets"
                    )
            try:
                selected = self._select(measurements, objective)
            except FastPathAdmissionError as exc:
                profile_store.put(
                    FastPathProfile(
                        key=profile_key,
                        measurements=measurements,
                        selected_candidate=None,
                        exactness_result="failed",
                        failure_reason=str(exc),
                    )
                )
                raise
            profile_store.put(
                FastPathProfile(
                    key=profile_key,
                    measurements=measurements,
                    selected_candidate=selected.candidate_mode,
                    exactness_result="passed",
                )
            )
        else:
            if cached.exactness_result != "passed":
                raise FastPathAdmissionError(
                    cached.failure_reason or "cached fast-path exactness admission failed"
                )
            selected = self._select(cached.measurements, objective)
        return path.prepare(
            model,
            device,
            selected,
            profile_fingerprint=profile_key.fingerprint,
            **(prepare_kwargs or {}),
        )


__all__ = ["FastPathAdmissionError", "NativeFastPathRegistry"]

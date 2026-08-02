"""Manifest, challenge, duplicate-result, reputation, and quarantine controls."""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_010.codecs import numerical_error
from swarm_inference.experiments.experiment_010.schemas import (
    DeterminismMode,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    WorkerManifest,
)
from swarm_inference.experiments.experiment_010.worker import verify_worker_signature


@dataclass(slots=True)
class WorkerReputation:
    worker_id: str
    requests_completed: int = 0
    timeouts: int = 0
    incorrect_results: int = 0
    challenge_requests: int = 0
    challenge_failures: int = 0
    duplicate_checks: int = 0
    duplicate_disagreements: int = 0
    latency_ns: list[int] = field(default_factory=list)
    confidence_score: float = 0.5
    quarantined: bool = False
    quarantine_reason: str | None = None
    updated_at_ns: int = 0

    def update_confidence(self) -> None:
        successes = self.requests_completed - self.incorrect_results - self.challenge_failures
        failures = (
            self.timeouts
            + self.incorrect_results
            + 2 * self.challenge_failures
            + self.duplicate_disagreements
        )
        self.confidence_score = (max(successes, 0) + 1) / (max(successes, 0) + failures + 2)
        self.updated_at_ns = time.time_ns()

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        if self.latency_ns:
            result["median_latency_ns"] = int(np.median(self.latency_ns))
            result["p95_latency_ns"] = int(np.percentile(self.latency_ns, 95))
        else:
            result["median_latency_ns"] = None
            result["p95_latency_ns"] = None
        return result


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    accepted: bool
    reasons: tuple[str, ...]
    numerical: dict[str, Any] | None
    detection_latency_ns: int
    verification_compute_ns: int
    verification_network_bytes: int


class TrustController:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        quantization_fingerprint: str,
        model_fingerprint: str,
        sampled_duplicate_fraction: float = 0.01,
        quarantine_failures: int = 2,
        seed: int = 1010,
    ) -> None:
        if not 0 <= sampled_duplicate_fraction <= 1:
            raise ValueError("sampled duplicate fraction must be between zero and one")
        if quarantine_failures <= 0:
            raise ValueError("quarantine threshold must be positive")
        self.model_id = model_id
        self.model_revision = model_revision
        self.quantization_fingerprint = quantization_fingerprint
        self.model_fingerprint = model_fingerprint
        self.sampled_duplicate_fraction = sampled_duplicate_fraction
        self.quarantine_failures = quarantine_failures
        self.random = random.Random(seed)
        self.manifests: dict[str, WorkerManifest] = {}
        self.secrets: dict[str, bytes] = {}
        self.reputations: dict[str, WorkerReputation] = {}
        self.history: list[dict[str, Any]] = []

    def register(self, manifest: WorkerManifest, *, signature_secret: bytes) -> None:
        mismatches = []
        for label, expected, observed in (
            ("model_id", self.model_id, manifest.model_id),
            ("model_revision", self.model_revision, manifest.model_revision),
            (
                "quantization_fingerprint",
                self.quantization_fingerprint,
                manifest.quantization_fingerprint,
            ),
            ("model_fingerprint", self.model_fingerprint, manifest.model_fingerprint),
        ):
            if observed != expected:
                mismatches.append(f"{label}: expected {expected!r}, received {observed!r}")
        if mismatches:
            raise ValueError("worker manifest identity mismatch: " + "; ".join(mismatches))
        if not manifest.bridge_version or not manifest.tensor_hashes:
            raise ValueError("worker manifest must prove bridge version and tensor hashes")
        self.manifests[manifest.worker_id] = manifest
        self.secrets[manifest.worker_id] = signature_secret
        self.reputations.setdefault(manifest.worker_id, WorkerReputation(manifest.worker_id))
        self._record(manifest.worker_id, "registered", {"pid": manifest.process_id})

    def should_duplicate(self, request: ExpertExecutionRequest) -> bool:
        if request.challenge:
            return True
        # Use a stable request-local decision so resume/replay does not change
        # which requests are verified.
        digest = hashlib.sha256(request.request_id.encode("utf-8")).digest()
        sample = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return sample < self.sampled_duplicate_fraction

    def verify(
        self,
        request: ExpertExecutionRequest,
        response: ExpertExecutionResponse,
        result: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        latency_ns: int | None = None,
        verification_network_bytes: int = 0,
    ) -> VerificationDecision:
        started = time.perf_counter_ns()
        reasons: list[str] = []
        reputation = self.reputations.setdefault(
            response.worker_id, WorkerReputation(response.worker_id)
        )
        manifest = self.manifests.get(response.worker_id)
        secret = self.secrets.get(response.worker_id)
        if manifest is None or secret is None:
            reasons.append("unregistered_worker")
        else:
            if response.model_revision != self.model_revision:
                reasons.append("wrong_model_revision")
            if response.integrity.model_fingerprint != self.model_fingerprint:
                reasons.append("wrong_model_fingerprint")
            if request.layer_id != response.layer_id:
                reasons.append("wrong_layer")
            if request.request_id != response.request_id:
                reasons.append("stale_or_wrong_request")
            if not verify_worker_signature(response, result, secret=secret):
                reasons.append("result_hash_or_signature_mismatch")
            owned = set(manifest.owned_experts.get(str(request.layer_id), []))
            if not set(request.expert_ids) <= owned:
                reasons.append("worker_did_not_own_requested_expert")
        numerical: dict[str, Any] | None = None
        if reference is not None:
            numerical = numerical_error(reference, result)
            if request.determinism_mode == DeterminismMode.EXACT:
                if not numerical["exact"]:
                    reasons.append("exact_result_mismatch")
            elif numerical["relative_l2_error"] > 1e-2 or numerical["cosine_similarity"] < 0.999:
                reasons.append("quality_contract_exceeded")
        accepted = not reasons
        reputation.requests_completed += 1
        if latency_ns is not None:
            reputation.latency_ns.append(latency_ns)
        if request.challenge:
            reputation.challenge_requests += 1
            if not accepted:
                reputation.challenge_failures += 1
        if not accepted:
            reputation.incorrect_results += 1
        self._apply_quarantine(reputation)
        reputation.update_confidence()
        elapsed = time.perf_counter_ns() - started
        self._record(
            response.worker_id,
            "verified" if accepted else "rejected",
            {"request_id": request.request_id, "reasons": reasons},
        )
        return VerificationDecision(
            accepted=accepted,
            reasons=tuple(reasons),
            numerical=numerical,
            detection_latency_ns=elapsed,
            verification_compute_ns=elapsed,
            verification_network_bytes=verification_network_bytes,
        )

    def compare_duplicate(
        self,
        worker_a: str,
        result_a: np.ndarray,
        worker_b: str,
        result_b: np.ndarray,
        *,
        exact: bool,
    ) -> VerificationDecision:
        started = time.perf_counter_ns()
        numerical = numerical_error(result_a, result_b)
        accepted = bool(
            numerical["exact"]
            if exact
            else numerical["relative_l2_error"] <= 1e-2 and numerical["cosine_similarity"] >= 0.999
        )
        for worker_id in (worker_a, worker_b):
            reputation = self.reputations.setdefault(worker_id, WorkerReputation(worker_id))
            reputation.duplicate_checks += 1
            if not accepted:
                reputation.duplicate_disagreements += 1
            self._apply_quarantine(reputation)
            reputation.update_confidence()
        elapsed = time.perf_counter_ns() - started
        self._record(
            worker_a,
            "duplicate_agreement" if accepted else "duplicate_disagreement",
            {"peer": worker_b, "numerical": numerical},
        )
        return VerificationDecision(
            accepted=accepted,
            reasons=() if accepted else ("duplicate_result_disagreement",),
            numerical=numerical,
            detection_latency_ns=elapsed,
            verification_compute_ns=elapsed,
            verification_network_bytes=int(result_a.nbytes + result_b.nbytes),
        )

    def timeout(self, worker_id: str) -> None:
        reputation = self.reputations.setdefault(worker_id, WorkerReputation(worker_id))
        reputation.timeouts += 1
        self._apply_quarantine(reputation)
        reputation.update_confidence()
        self._record(worker_id, "timeout", {})

    def _apply_quarantine(self, reputation: WorkerReputation) -> None:
        integrity_failures = (
            reputation.incorrect_results
            + reputation.challenge_failures
            + reputation.duplicate_disagreements
        )
        if integrity_failures >= self.quarantine_failures:
            reputation.quarantined = True
            reputation.quarantine_reason = (
                f"integrity failure threshold {self.quarantine_failures} reached"
            )

    def _record(self, worker_id: str, event: str, details: dict[str, Any]) -> None:
        self.history.append(
            {
                "timestamp_ns": time.time_ns(),
                "worker_id": worker_id,
                "event": event,
                "details": details,
            }
        )

    def eligible(self, worker_id: str, *, minimum_confidence: float = 0.5) -> bool:
        reputation = self.reputations.get(worker_id)
        return bool(
            reputation
            and not reputation.quarantined
            and math.isfinite(reputation.confidence_score)
            and reputation.confidence_score >= minimum_confidence
        )


def reconcile_expert_ownership(
    manifests: list[WorkerManifest],
    *,
    expected: set[tuple[int, int]],
    allow_replication: bool = False,
) -> dict[str, Any]:
    owners: dict[tuple[int, int], list[str]] = {}
    content_hashes: dict[tuple[int, int], set[str]] = {}
    per_worker: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        owned: set[tuple[int, int]] = set()
        for layer, experts in manifest.owned_experts.items():
            for expert in experts:
                key = (int(layer), int(expert))
                owned.add(key)
                owners.setdefault(key, []).append(manifest.worker_id)
                tensor_hash = manifest.tensor_hashes.get(f"{key[0]}:{key[1]}", "")
                content_hashes.setdefault(key, set()).add(tensor_hash)
        per_worker[manifest.worker_id] = {
            "owned_experts": [list(item) for item in sorted(owned)],
            "resident_tensor_bytes": manifest.resident_tensor_bytes,
            "expert_bytes": manifest.expert_bytes,
            "cache_bytes": manifest.cache_bytes,
            "peak_rss_bytes": manifest.peak_rss_bytes,
        }
    observed = set(owners)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = {key: value for key, value in owners.items() if len(value) > 1}
    hash_conflicts = {
        key: sorted(values) for key, values in content_hashes.items() if len(values - {""}) > 1
    }
    return {
        "complete": not missing and not unexpected,
        "disjoint": not duplicates,
        "allow_replication": allow_replication,
        "valid": bool(
            not missing
            and not unexpected
            and not hash_conflicts
            and (allow_replication or not duplicates)
        ),
        "missing": [list(item) for item in missing],
        "unexpected": [list(item) for item in unexpected],
        "duplicates": {f"{key[0]}:{key[1]}": value for key, value in duplicates.items()},
        "hash_conflicts": {f"{key[0]}:{key[1]}": value for key, value in hash_conflicts.items()},
        "per_worker": per_worker,
    }


def reconcile_microshard_ownership(
    manifests: list[WorkerManifest],
    *,
    expected_widths: dict[tuple[int, int], int],
) -> dict[str, Any]:
    ranges: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    for manifest in manifests:
        for descriptor in manifest.owned_microshards:
            key = (int(descriptor["layer_id"]), int(descriptor["expert_id"]))
            ranges.setdefault(key, []).append(
                (
                    int(descriptor["hidden_start"]),
                    int(descriptor["hidden_end"]),
                    manifest.worker_id,
                )
            )
    missing_experts = sorted(set(expected_widths) - set(ranges))
    unexpected_experts = sorted(set(ranges) - set(expected_widths))
    errors = []
    coverage: dict[str, Any] = {}
    for key, logical_width in sorted(expected_widths.items()):
        owned = sorted(ranges.get(key, []))
        cursor = 0
        for start, end, worker in owned:
            if start != cursor:
                errors.append(f"{key[0]}:{key[1]} expected start {cursor}, got {start} on {worker}")
            if end <= start or end > logical_width:
                errors.append(f"{key[0]}:{key[1]} invalid range {start}:{end} on {worker}")
            cursor = end
        if cursor != logical_width:
            errors.append(f"{key[0]}:{key[1]} coverage ends at {cursor}, expected {logical_width}")
        coverage[f"{key[0]}:{key[1]}"] = [
            {"hidden_start": start, "hidden_end": end, "worker_id": worker}
            for start, end, worker in owned
        ]
    return {
        "valid": not missing_experts and not unexpected_experts and not errors,
        "missing_experts": [list(item) for item in missing_experts],
        "unexpected_experts": [list(item) for item in unexpected_experts],
        "errors": errors,
        "coverage": coverage,
        "per_worker": {
            manifest.worker_id: {
                "expert_bytes": manifest.expert_bytes,
                "resident_tensor_bytes": manifest.resident_tensor_bytes,
                "cache_bytes": manifest.cache_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "owned_microshards": manifest.owned_microshards,
            }
            for manifest in manifests
        },
    }

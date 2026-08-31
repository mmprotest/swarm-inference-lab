"""Audited E025-only Vast.ai lifecycle and short-run offer selection."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import (
    CONSUMER_GPU_NAMES,
    HEADLINE_BACKBONE_GPU_NAMES,
    PREFERRED_SMALL_GPU_NAMES,
    PROFESSIONAL_GPU_MARKERS,
)
from .io import append_jsonl, atomic_write_json, canonical_sha256, read_json, utc_now

LEDGER_SCHEMA_VERSION = "experiment-025-vast-ledger-v1"
OFFER_SCHEMA_VERSION = "experiment-025-vast-offers-v1"


def _json_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        for index, character in enumerate(stripped):
            if character in "[{":
                try:
                    return json.loads(stripped[index:])
                except json.JSONDecodeError:
                    continue
        raise


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("offers", "instances", "results", "data"):
            child = value.get(key)
            if isinstance(child, list):
                return [dict(row) for row in child if isinstance(row, dict)]
        return [dict(value)]
    return []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _gpu_allowed(name: str) -> bool:
    normalized = name.upper()
    return (
        any(candidate.upper() in normalized for candidate in CONSUMER_GPU_NAMES)
        and not any(marker in normalized for marker in PROFESSIONAL_GPU_MARKERS)
    )


@dataclass(frozen=True, slots=True)
class Offer:
    offer_id: int
    machine_id: int
    gpu_name: str
    gpu_count: int
    gpu_ram_gib: float
    reliability: float
    verified: bool
    rentable: bool
    dph_total: float
    storage_cost_per_gb_month: float
    inet_down_mbps: float
    inet_up_mbps: float
    inet_down_cost_per_gb: float
    inet_up_cost_per_gb: float
    disk_bw_mbps: float
    disk_space_gb: float
    direct_port_count: int
    static_ip: bool
    driver_version: str
    cuda_max_version: float
    raw: dict[str, Any]

    @property
    def gpu_rental_rate_per_hour(self) -> float:
        return _number(self.raw.get("dph_base"), self.dph_total)

    def effective_rate_per_hour(self, disk_gb: float) -> float:
        return self.gpu_rental_rate_per_hour + (
            self.storage_cost_per_gb_month * disk_gb / (30.0 * 24.0)
        )

    @classmethod
    def from_raw(cls, row: dict[str, Any]) -> Offer:
        name = str(row.get("gpu_name", row.get("gpu", ""))).replace("_", " ")
        raw_gpu_ram = _number(row.get("gpu_ram"))
        # Vast 1.5.4 emits raw gpu_ram in MiB even though its query language uses
        # GiB. Keep the normalized record unit explicit and independently tested.
        gpu_ram_gib = raw_gpu_ram / 1024.0 if raw_gpu_ram > 1024.0 else raw_gpu_ram
        verification = str(row.get("verification", "")).strip().lower()
        verified = (
            verification == "verified"
            or _integer(row.get("vericode"), 0) == 1
            or row.get("verified") is True
        ) and row.get("is_vm_deverified") is not True
        return cls(
            offer_id=_integer(row.get("id", row.get("ask_contract_id"))),
            machine_id=_integer(row.get("machine_id")),
            gpu_name=name,
            gpu_count=_integer(row.get("num_gpus"), 0),
            gpu_ram_gib=gpu_ram_gib,
            reliability=_number(row.get("reliability2", row.get("reliability"))),
            verified=verified,
            rentable=bool(row.get("rentable", True)),
            dph_total=_number(row.get("dph_total", row.get("dph"))),
            storage_cost_per_gb_month=_number(row.get("storage_cost")),
            inet_down_mbps=_number(row.get("inet_down")),
            inet_up_mbps=_number(row.get("inet_up")),
            inet_down_cost_per_gb=_number(row.get("inet_down_cost")),
            inet_up_cost_per_gb=_number(row.get("inet_up_cost")),
            disk_bw_mbps=_number(row.get("disk_bw")),
            disk_space_gb=_number(row.get("disk_space")),
            direct_port_count=_integer(row.get("direct_port_count"), 0),
            static_ip=bool(row.get("static_ip")),
            driver_version=str(row.get("driver_version", row.get("driver_vers", ""))),
            cuda_max_version=_number(row.get("cuda_max_good", row.get("cuda_vers"))),
            raw=dict(row),
        )

    def qualifies(
        self,
        role: str,
        *,
        disk_gb: float,
        allow_multi_gpu: bool = False,
    ) -> bool:
        preferred = (
            PREFERRED_SMALL_GPU_NAMES
            if role == "SUB_LAYER_WORKER"
            else HEADLINE_BACKBONE_GPU_NAMES
        )
        minimum_vram = 7.5 if role == "SUB_LAYER_WORKER" else 23.0
        return (
            self.offer_id >= 0
            and self.machine_id >= 0
            and (self.gpu_count >= 1 if allow_multi_gpu else self.gpu_count == 1)
            and self.rentable
            and self.verified
            and self.reliability >= 0.98
            and _gpu_allowed(self.gpu_name)
            and any(name.upper() in self.gpu_name.upper() for name in preferred)
            and self.gpu_ram_gib >= minimum_vram
            and self.disk_space_gb >= disk_gb
            and self.inet_down_mbps > 0
            and self.disk_bw_mbps > 0
            and self.direct_port_count >= 1
            and self.cuda_max_version >= 13.0
        )

    def acquisition_score(
        self,
        *,
        required_download_bytes: int,
        disk_gb: float,
        bootstrap_fixed_seconds: float = 90.0,
        history: Mapping[str, Any] | None = None,
        scoring_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        transfer_seconds = required_download_bytes * 8 / (self.inet_down_mbps * 1e6)
        bootstrap_seconds = bootstrap_fixed_seconds + transfer_seconds
        active_cost = self.gpu_rental_rate_per_hour * bootstrap_seconds / 3600.0
        storage_cost = self.storage_cost_per_gb_month * disk_gb * bootstrap_seconds / (
            30.0 * 24.0 * 3600.0
        )
        transfer_gb = required_download_bytes / 1e9
        ingress_cost = self.inet_down_cost_per_gb * transfer_gb
        reliability_penalty = max(0.0, 1.0 - self.reliability) * 2.0
        slow_disk_penalty = max(0.0, 500.0 - self.disk_bw_mbps) / 5000.0
        total = (
            active_cost
            + storage_cost
            + ingress_cost
            + reliability_penalty
            + slow_disk_penalty
        )
        baseline: dict[str, Any] = {
            "expected_bootstrap_seconds": bootstrap_seconds,
            "expected_active_cost_usd": active_cost,
            "expected_storage_cost_usd": storage_cost,
            "expected_ingress_cost_usd": ingress_cost,
            "reliability_penalty": reliability_penalty,
            "slow_disk_penalty": slow_disk_penalty,
            "short_run_acquisition_score": total,
        }
        if scoring_policy is None:
            return baseline

        machine = dict(history or {})
        prior_weight = float(scoring_policy["provider_reliability_prior_weight"])
        attempt_count = int(machine.get("attempt_count", 0))
        ready_success_count = int(machine.get("ready_success_count", 0))
        provider_probability = min(0.9999, max(0.01, self.reliability))
        ready_probability = (
            ready_success_count + prior_weight * provider_probability
        ) / max(1.0, attempt_count + prior_weight)

        mapping_prior = float(scoring_policy["public_mapping_prior_probability"])
        mapping_prior_weight = float(
            scoring_policy["public_mapping_prior_weight"]
        )
        mapping_observations = int(
            machine.get("public_mapping_observation_count", attempt_count)
        )
        mapping_successes = int(
            machine.get("public_mapping_success_count", ready_success_count)
        )
        mapping_probability = (
            mapping_successes + mapping_prior_weight * mapping_prior
        ) / max(1.0, mapping_observations + mapping_prior_weight)

        healthy_prior = float(scoring_policy["post_ready_health_prior_probability"])
        healthy_prior_weight = float(
            scoring_policy["post_ready_health_prior_weight"]
        )
        healthy_observations = int(
            machine.get("post_ready_health_observation_count", ready_success_count)
        )
        healthy_successes = int(machine.get("ready_healthy_count", 0))
        healthy_probability = (
            healthy_successes + healthy_prior_weight * healthy_prior
        ) / max(1.0, healthy_observations + healthy_prior_weight)

        probability_floor = float(scoring_policy["probability_floor"])
        healthy_ready_probability = min(
            0.9999,
            max(
                probability_floor,
                ready_probability * mapping_probability * healthy_probability,
            ),
        )

        observed_ready_seconds = machine.get("median_time_to_ready_seconds")
        global_ready_seconds = float(
            scoring_policy["global_median_ready_seconds"]
        )
        history_time_prior_weight = float(
            scoring_policy["history_time_prior_weight"]
        )
        if observed_ready_seconds is not None and ready_success_count > 0:
            expected_ready_seconds = (
                float(observed_ready_seconds) * ready_success_count
                + global_ready_seconds * history_time_prior_weight
            ) / (ready_success_count + history_time_prior_weight)
        else:
            expected_ready_seconds = max(bootstrap_seconds, global_ready_seconds)

        observed_download_mbps = machine.get("median_download_throughput_mbps")
        if observed_download_mbps is not None and float(observed_download_mbps) > 0:
            empirical_transfer_seconds = (
                required_download_bytes * 8 / (float(observed_download_mbps) * 1e6)
            )
            expected_ready_seconds = max(
                bootstrap_fixed_seconds + empirical_transfer_seconds,
                expected_ready_seconds,
            )

        expected_active_cost = (
            self.gpu_rental_rate_per_hour * expected_ready_seconds / 3600.0
        )
        expected_storage_cost = (
            self.storage_cost_per_gb_month
            * disk_gb
            * expected_ready_seconds
            / (30.0 * 24.0 * 3600.0)
        )
        expected_direct_cost = expected_active_cost + expected_storage_cost + ingress_cost
        expected_cost_to_healthy = expected_direct_cost / healthy_ready_probability
        fleet_delay_cost = (
            expected_ready_seconds
            / 60.0
            / healthy_ready_probability
            * float(scoring_policy["fleet_delay_cost_usd_per_minute"])
        )
        repeated_failure_penalty = float(
            scoring_policy["repeated_attributable_failure_penalty_usd"]
        ) * int(machine.get("attributable_failure_count", 0))
        historical_success_credit = min(
            float(scoring_policy["maximum_historical_success_credit_usd"]),
            float(scoring_policy["historical_success_credit_usd"])
            * int(machine.get("ready_healthy_count", 0)),
        )
        score = max(
            0.0,
            expected_cost_to_healthy
            + fleet_delay_cost
            + repeated_failure_penalty
            - historical_success_credit,
        )
        if bool(machine.get("hard_excluded")):
            score += float(scoring_policy["hard_exclusion_penalty_usd"])
        return {
            **baseline,
            "score_formula_version": str(scoring_policy["formula_version"]),
            "history_attempt_count": attempt_count,
            "history_ready_success_count": ready_success_count,
            "history_ready_healthy_count": int(
                machine.get("ready_healthy_count", 0)
            ),
            "history_attributable_failure_count": int(
                machine.get("attributable_failure_count", 0)
            ),
            "empirical_probability_ready": ready_probability,
            "empirical_probability_public_mapping": mapping_probability,
            "empirical_probability_post_ready_health": healthy_probability,
            "empirical_probability_healthy_ready": healthy_ready_probability,
            "expected_ready_seconds": expected_ready_seconds,
            "expected_active_cost_usd": expected_active_cost,
            "expected_storage_cost_usd": expected_storage_cost,
            "expected_direct_attempt_cost_usd": expected_direct_cost,
            "expected_cost_to_healthy_worker_usd": expected_cost_to_healthy,
            "fleet_tail_delay_cost_usd": fleet_delay_cost,
            "repeated_attributable_failure_penalty_usd": repeated_failure_penalty,
            "historical_success_credit_usd": historical_success_credit,
            "hard_excluded_by_history": bool(machine.get("hard_excluded")),
            "short_run_acquisition_score": score,
        }


class AppendOnlyLifecycleLedger:
    """Hash-chained, fsynced ledger shared with the independent watchdog."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path.expanduser().resolve()
        self.run_id = run_id
        self.lock = threading.Lock()

    @contextmanager
    def _process_lock(self) -> Any:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("E025 lifecycle ledger contains a non-object")
                rows.append(value)
        previous = "0" * 64
        for sequence, row in enumerate(rows, start=1):
            if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise ValueError("E025 lifecycle ledger schema differs")
            if int(row.get("sequence", 0)) != sequence:
                raise ValueError("E025 lifecycle ledger sequence is discontinuous")
            if row.get("previous_entry_sha256") != previous:
                raise ValueError("E025 lifecycle ledger hash chain differs")
            observed = str(row.get("entry_sha256", ""))
            body = dict(row)
            body.pop("entry_sha256", None)
            if canonical_sha256(body) != observed:
                raise ValueError("E025 lifecycle ledger entry digest differs")
            previous = observed
        return rows

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        with self.lock, self._process_lock():
            rows = self.entries()
            entry = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "sequence": len(rows) + 1,
                "run_id": self.run_id,
                "timestamp": utc_now(),
                "event": event,
                "previous_entry_sha256": (
                    str(rows[-1]["entry_sha256"]) if rows else "0" * 64
                ),
                **fields,
            }
            entry["entry_sha256"] = canonical_sha256(entry)
            append_jsonl(self.path, entry)
            return entry

    def instance_ids(self) -> list[int]:
        return sorted(
            {
                int(row["instance_id"])
                for row in self.entries()
                if row.get("event") == "CREATE_CONFIRMED"
                and row.get("instance_id") is not None
            }
        )


class VastClient:
    def __init__(
        self,
        *,
        executable: str = "vastai",
        ledger: AppendOnlyLifecycleLedger | None = None,
    ) -> None:
        self.executable = executable
        self.ledger = ledger

    def _run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )

    def version(self) -> str:
        result = self._run(["--version"], timeout_seconds=30)
        if result.returncode != 0:
            raise RuntimeError(f"Vast CLI version failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def search_offers(
        self,
        *,
        gpu_names: tuple[str, ...],
        storage_gb: float,
        limit: int = 1000,
        single_gpu_only: bool = True,
    ) -> list[Offer]:
        names = ",".join(f'"{name}"' for name in gpu_names)
        gpu_count = " num_gpus=1" if single_gpu_only else ""
        query = (
            f"gpu_name in [{names}]{gpu_count} rentable=True verified=True "
            "reliability>=0.98 direct_port_count>=1 cuda_vers>=13.0"
        )
        result = self._run(
            [
                "search",
                "offers",
                query,
                "--on-demand",
                "--limit",
                str(limit),
                "--storage",
                str(storage_gb),
                "--raw",
            ],
            timeout_seconds=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Vast offer search failed: {result.stderr[-1000:]}")
        return [Offer.from_raw(row) for row in _rows(_json_output(result.stdout))]

    def show_instances(self) -> list[dict[str, Any]]:
        result = self._run(
            ["show", "instances", "--raw", "--all"],
            timeout_seconds=90,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Vast instance query failed: {result.stderr[-1000:]}")
        return _rows(_json_output(result.stdout))

    def user_budget(self) -> dict[str, Any]:
        """Return only spend controls; discard user identity and SSH fields."""

        result = self._run(["show", "user", "--raw"], timeout_seconds=90)
        if result.returncode != 0:
            raise RuntimeError(f"Vast user budget query failed: {result.stderr[-1000:]}")
        value = _json_output(result.stdout)
        if not isinstance(value, dict):
            raise RuntimeError("Vast user budget query returned an unexpected payload")
        credit = _number(value.get("credit"))
        balance = _number(value.get("balance"))
        return {
            "credit_usd": credit,
            "balance_usd": balance,
            "conservative_available_usd": max(0.0, credit + balance),
            "identity_fields_persisted": False,
            "api_key_persisted": False,
        }

    def logs(self, instance_id: int, *, tail: int = 2000) -> str:
        result = self._run(
            ["logs", str(instance_id), "--tail", str(tail)],
            timeout_seconds=90,
        )
        if self.ledger is not None:
            self.ledger.append(
                "LOGS_RETRIEVED" if result.returncode == 0 else "LOGS_RETRIEVAL_FAILED",
                command_category="logs",
                offer_id=None,
                instance_id=instance_id,
                returncode=result.returncode,
                log_bytes=len(result.stdout.encode("utf-8", errors="replace")),
                stderr_tail=result.stderr[-1000:],
            )
        if result.returncode != 0:
            raise RuntimeError(f"Vast log retrieval failed for {instance_id}")
        return result.stdout

    def create_instance(
        self,
        *,
        run_id: str,
        role: str,
        index: int,
        offer: Offer,
        image: str,
        disk_gb: int,
        env_options: str,
        watchdog_receipt: Path,
        go_receipt: Path,
    ) -> int:
        if self.ledger is None:
            raise RuntimeError("E025 mutations require an append-only lifecycle ledger")
        if not watchdog_receipt.is_file() or not go_receipt.is_file():
            raise RuntimeError(
                "E025 create is not armed by matching watchdog and GO receipts"
            )
        watchdog = read_json(watchdog_receipt)
        go = read_json(go_receipt)
        if (
            watchdog.get("status") != "RUNNING"
            or watchdog.get("run_id") != run_id
            or go.get("status") != "GO"
            or go.get("run_id") != run_id
        ):
            raise RuntimeError("E025 create is not armed by matching watchdog and GO receipts")
        label = f"e025-{run_id}-{role.lower()}-{index:03d}"
        self.ledger.append(
            "CREATE_REQUESTED",
            command_category="create instance",
            offer_id=offer.offer_id,
            instance_id=None,
            machine_id=offer.machine_id,
            gpu_model=offer.gpu_name,
            gpu_count=offer.gpu_count,
            advertised_vram_gib=offer.gpu_ram_gib,
            advertised_reliability=offer.reliability,
            advertised_internet_down_mbps=offer.inet_down_mbps,
            advertised_internet_up_mbps=offer.inet_up_mbps,
            advertised_disk_bandwidth_mbps=offer.disk_bw_mbps,
            active_rental_rate_usd_per_hour=offer.gpu_rental_rate_per_hour,
            storage_rate_usd_per_gb_month=offer.storage_cost_per_gb_month,
            internet_ingress_rate_usd_per_gb=offer.inet_down_cost_per_gb,
            internet_egress_rate_usd_per_gb=offer.inet_up_cost_per_gb,
            requested_disk_gb=disk_gb,
            instance_label=label,
            role=role,
            index=index,
            secrets_logged=False,
        )
        result = self._run(
            [
                "create",
                "instance",
                str(offer.offer_id),
                "--raw",
                "--image",
                image,
                "--disk",
                str(disk_gb),
                "--label",
                label,
                "--env",
                env_options,
                "--cancel-unavail",
                "--args",
                "--port",
                "42525",
            ],
            timeout_seconds=120,
        )
        if result.returncode != 0:
            self.ledger.append(
                "CREATE_FAILED",
                command_category="create instance",
                offer_id=offer.offer_id,
                instance_id=None,
                machine_id=offer.machine_id,
                instance_label=label,
                returncode=result.returncode,
                stderr_tail=result.stderr[-1000:],
                secrets_logged=False,
            )
            raise RuntimeError(f"Vast create failed for offer {offer.offer_id}")
        value = _json_output(result.stdout)
        instance_id = (
            _integer(value.get("new_contract", value.get("id")))
            if isinstance(value, dict) and value.get("success") is True
            else -1
        )
        if instance_id < 0:
            matching = [
                row
                for row in self.show_instances()
                if str(row.get("label", "")) == label
            ]
            recovered_ids = {
                _integer(row.get("id", row.get("instance_id"))) for row in matching
            }
            recovered_ids.discard(-1)
            if len(recovered_ids) != 1:
                self.ledger.append(
                    "CREATE_ID_LOST",
                    command_category="create instance",
                    offer_id=offer.offer_id,
                    instance_id=None,
                    machine_id=offer.machine_id,
                    instance_label=label,
                    final_status="EMERGENCY_LABEL_CLEANUP_REQUIRED",
                    matched_instance_ids=sorted(recovered_ids),
                )
                raise RuntimeError("Vast create succeeded without one recoverable instance ID")
            instance_id = recovered_ids.pop()
        self.ledger.append(
            "CREATE_CONFIRMED",
            command_category="create instance",
            offer_id=offer.offer_id,
            instance_id=instance_id,
            machine_id=offer.machine_id,
            gpu_model=offer.gpu_name,
            gpu_count=offer.gpu_count,
            advertised_vram_gib=offer.gpu_ram_gib,
            advertised_reliability=offer.reliability,
            advertised_internet_down_mbps=offer.inet_down_mbps,
            advertised_internet_up_mbps=offer.inet_up_mbps,
            advertised_disk_bandwidth_mbps=offer.disk_bw_mbps,
            active_rental_rate_usd_per_hour=offer.gpu_rental_rate_per_hour,
            storage_rate_usd_per_gb_month=offer.storage_cost_per_gb_month,
            internet_ingress_rate_usd_per_gb=offer.inet_down_cost_per_gb,
            internet_egress_rate_usd_per_gb=offer.inet_up_cost_per_gb,
            requested_disk_gb=disk_gb,
            instance_label=label,
            creation_time=utc_now(),
            running_time=None,
            worker_ready_time=None,
            destroy_request_time=None,
            destroy_confirmed_time=None,
            final_status="CREATED",
            raw_response_redacted={"success": True, "new_contract": instance_id},
            secrets_logged=False,
        )
        return instance_id

    def destroy_instance(self, instance_id: int, *, reason: str) -> bool:
        if self.ledger is None:
            raise RuntimeError("E025 destruction requires its lifecycle ledger")
        self.ledger.append(
            "DESTROY_REQUESTED",
            command_category="destroy instance",
            offer_id=None,
            instance_id=instance_id,
            destroy_request_time=utc_now(),
            reason=reason,
        )
        result = self._run(
            ["destroy", "instance", str(instance_id), "--yes", "--raw"],
            timeout_seconds=90,
        )
        success = result.returncode == 0
        self.ledger.append(
            "DESTROY_CONFIRMED" if success else "DESTROY_FAILED",
            command_category="destroy instance",
            offer_id=None,
            instance_id=instance_id,
            destroy_confirmed_time=utc_now() if success else None,
            final_status="DESTROYED" if success else "DESTROY_FAILED",
            returncode=result.returncode,
            stderr_tail=result.stderr[-1000:],
            reason=reason,
        )
        return success


def rank_offers_for_workers(
    offers: list[Offer],
    workers: list[dict[str, Any]],
    *,
    disk_gb: int,
    alternates_per_worker: int = 3,
) -> dict[str, Any]:
    candidates = [offer for offer in offers if _gpu_allowed(offer.gpu_name)]
    used_offer_ids: set[int] = set()
    used_machine_ids: set[int] = set()
    assignments: list[dict[str, Any]] = []
    ordered_workers = sorted(
        workers,
        key=lambda row: (
            0 if row["role"] == "SUB_LAYER_WORKER" else 1,
            -int(row["download_bytes_cold_cache"]),
            str(row["worker_id"]),
        ),
    )
    for worker in ordered_workers:
        role = str(worker["role"])
        scored: list[tuple[float, Offer, dict[str, float]]] = []
        for offer in candidates:
            if (
                offer.offer_id in used_offer_ids
                or offer.machine_id in used_machine_ids
                or not offer.qualifies(role, disk_gb=disk_gb)
            ):
                continue
            score = offer.acquisition_score(
                required_download_bytes=int(worker["download_bytes_cold_cache"]),
                disk_gb=disk_gb,
            )
            bootstrap_limit = (
                8 * 60
                if role == "SUB_LAYER_WORKER"
                else (4 * 60 if role == "SUB_LAYER_PARENT" else 6 * 60)
            )
            if float(score["expected_bootstrap_seconds"]) > bootstrap_limit:
                continue
            scored.append(
                (
                    float(score["short_run_acquisition_score"]),
                    offer,
                    score,
                )
            )
        scored.sort(key=lambda row: (row[0], -row[1].reliability, row[1].offer_id))
        if not scored:
            raise RuntimeError(f"no qualifying live Vast offer for {worker['worker_id']}")
        _, selected, score = scored[0]
        used_offer_ids.add(selected.offer_id)
        used_machine_ids.add(selected.machine_id)
        alternates = [
            {
                "offer": asdict(offer),
                "score": alternate_score,
            }
            for _, offer, alternate_score in scored[1 : 1 + alternates_per_worker]
        ]
        assignments.append(
            {
                **worker,
                "selected_offer": asdict(selected),
                "selection_score": score,
                "alternates": alternates,
            }
        )
    assignments.sort(key=lambda row: str(row["worker_id"]))
    return {
        "schema_version": "experiment-025-fleet-plan-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "on_demand_only": True,
        "consumer_only": True,
        "disk_gb_per_instance": disk_gb,
        "worker_count": len(assignments),
        "unique_selected_offer_count": len(
            {int(row["selected_offer"]["offer_id"]) for row in assignments}
        ),
        "unique_selected_machine_count": len(
            {int(row["selected_offer"]["machine_id"]) for row in assignments}
        ),
        "sub_layer_machine_ids": sorted(
            int(row["selected_offer"]["machine_id"])
            for row in assignments
            if row["role"] == "SUB_LAYER_WORKER"
        ),
        "total_active_rental_rate_usd_per_hour": sum(
            Offer(**row["selected_offer"]).gpu_rental_rate_per_hour
            for row in assignments
        ),
        "total_expected_ingress_bytes": sum(
            int(row["download_bytes_cold_cache"]) for row in assignments
        ),
        "workers": assignments,
    }


def _temporary_disk_bytes(worker: dict[str, Any]) -> int:
    value = int(worker.get("temporary_disk_bytes", 0))
    if value > 0:
        return value
    download = int(worker.get("download_bytes_cold_cache", 0))
    assigned = int(worker.get("assigned_tensor_bytes", 0))
    return max(download + assigned, 40 * 1024**3)


def _group_disk_gb(workers: list[dict[str, Any]], minimum_disk_gb: int) -> int:
    temporary = sum(_temporary_disk_bytes(worker) for worker in workers)
    return max(minimum_disk_gb, math.ceil(temporary / 1024**3) + 10)


def _backbone_name(name: str) -> bool:
    normalized = name.upper()
    return any(value.upper() in normalized for value in HEADLINE_BACKBONE_GPU_NAMES)


def rank_grouped_offers_for_workers(
    offers: list[Offer],
    workers: list[dict[str, Any]],
    *,
    minimum_disk_gb: int,
    alternates_per_group: int = 3,
    reserved_alternate_machines: int = 8,
    excluded_machine_ids: set[int] | None = None,
    acquisition_history: Mapping[int, Mapping[str, Any]] | None = None,
    scoring_policy: Mapping[str, Any] | None = None,
    maximum_candidate_ready_seconds: float = 20 * 60,
) -> dict[str, Any]:
    """Pack one isolated worker process per GPU while retaining many machines."""

    fragments = sorted(
        (row for row in workers if row["role"] == "SUB_LAYER_WORKER"),
        key=lambda row: str(row["worker_id"]),
    )
    parents = [row for row in workers if row["role"] == "SUB_LAYER_PARENT"]
    backbone_workers = sorted(
        (row for row in workers if row["role"] == "BACKBONE_STAGE"),
        key=lambda row: str(row["worker_id"]),
    )
    if len(fragments) != 4 or len(parents) != 1 or len(backbone_workers) != 92:
        raise ValueError("E025 grouped planner requires 92 stages, one parent, and four fragments")
    excluded_machines = {int(value) for value in (excluded_machine_ids or set())}
    candidates = [
        offer
        for offer in offers
        if _gpu_allowed(offer.gpu_name) and offer.machine_id not in excluded_machines
    ]
    selected_groups: list[dict[str, Any]] = []
    used_offer_ids: set[int] = set()
    used_machine_ids: set[int] = set()

    def acquisition_score(
        offer: Offer, *, required_download_bytes: int, disk_gb: float
    ) -> dict[str, Any]:
        history = (
            acquisition_history.get(offer.machine_id)
            if acquisition_history is not None
            else None
        )
        return offer.acquisition_score(
            required_download_bytes=required_download_bytes,
            disk_gb=disk_gb,
            history=history,
            scoring_policy=scoring_policy,
        )

    def select_single(worker: dict[str, Any], group_id: str) -> None:
        role = str(worker["role"])
        scored: list[tuple[float, Offer, dict[str, float]]] = []
        for offer in candidates:
            if offer.offer_id in used_offer_ids or offer.machine_id in used_machine_ids:
                continue
            if not offer.qualifies(role, disk_gb=minimum_disk_gb):
                continue
            score = acquisition_score(
                offer,
                required_download_bytes=int(worker["download_bytes_cold_cache"]),
                disk_gb=minimum_disk_gb,
            )
            readiness_estimate = float(
                score.get("expected_ready_seconds", score["expected_bootstrap_seconds"])
            )
            if readiness_estimate > maximum_candidate_ready_seconds:
                continue
            scored.append(
                (float(score["short_run_acquisition_score"]), offer, score)
            )
        scored.sort(key=lambda row: (row[0], -row[1].reliability, row[1].offer_id))
        if not scored:
            raise RuntimeError(f"no distinct single-GPU offer for {worker['worker_id']}")
        _, offer, score = scored[0]
        used_offer_ids.add(offer.offer_id)
        used_machine_ids.add(offer.machine_id)
        selected_groups.append(
            {
                "instance_group_id": group_id,
                "role": role,
                "selected_offer": asdict(offer),
                "selection_score": score,
                "disk_gb": minimum_disk_gb,
                "workers": [
                    {
                        **worker,
                        "gpu_slot": 0,
                        "container_port": 42525,
                    }
                ],
            }
        )

    for index, fragment in enumerate(fragments):
        select_single(fragment, f"e025-sub-layer-group-{index:02d}")
    select_single(parents[0], "e025-layer-089-parent-group")

    backbone_offers = [
        offer
        for offer in candidates
        if offer.offer_id not in used_offer_ids
        and offer.machine_id not in used_machine_ids
        and _backbone_name(offer.gpu_name)
        and offer.qualifies(
            "BACKBONE_STAGE",
            disk_gb=minimum_disk_gb,
            allow_multi_gpu=True,
        )
        and 1 <= offer.gpu_count <= 8
        and offer.direct_port_count >= offer.gpu_count
    ]
    average_download = math.ceil(
        sum(int(row["download_bytes_cold_cache"]) for row in backbone_workers)
        / len(backbone_workers)
    )
    maximum_temporary = max(_temporary_disk_bytes(row) for row in backbone_workers)
    maximum_download = max(
        int(row["download_bytes_cold_cache"]) for row in backbone_workers
    )
    by_machine: dict[int, dict[int, tuple[float, Offer, int, dict[str, float]]]] = {}
    for offer in backbone_offers:
        capacity = int(offer.gpu_count)
        safe_disk_gb = max(
            minimum_disk_gb,
            math.ceil(capacity * maximum_temporary / 1024**3) + 10,
        )
        if offer.disk_space_gb < safe_disk_gb:
            continue
        score = acquisition_score(
            offer,
            required_download_bytes=capacity * average_download,
            disk_gb=safe_disk_gb,
        )
        worst_case_bootstrap = acquisition_score(
            offer,
            required_download_bytes=capacity * maximum_download,
            disk_gb=safe_disk_gb,
        )
        worst_case_ready = float(
            worst_case_bootstrap.get(
                "expected_ready_seconds",
                worst_case_bootstrap["expected_bootstrap_seconds"],
            )
        )
        if worst_case_ready > maximum_candidate_ready_seconds:
            continue
        architecture_penalty = 0.01 * capacity if "5090" in offer.gpu_name else 0.0
        total = float(score["short_run_acquisition_score"]) + architecture_penalty
        options = by_machine.setdefault(offer.machine_id, {})
        current = options.get(capacity)
        if current is None or total < current[0]:
            options[capacity] = (total, offer, safe_disk_gb, score)
    machine_options = sorted(by_machine.items())
    if not machine_options:
        raise RuntimeError("no qualifying grouped consumer backbone offers")
    maximum_machines = min(
        len(backbone_workers),
        max(1, len(machine_options) - reserved_alternate_machines),
        72,
    )
    states: dict[
        tuple[int, int],
        tuple[float, tuple[tuple[Offer, int, dict[str, float]], ...]],
    ] = {(0, 0): (0.0, ())}
    for _, options in machine_options:
        updated = dict(states)
        for (machine_count, gpu_count), (cost, choices) in states.items():
            if machine_count >= maximum_machines:
                continue
            for capacity, (option_cost, offer, safe_disk, score) in options.items():
                next_gpus = gpu_count + capacity
                if next_gpus > len(backbone_workers):
                    continue
                key = (machine_count + 1, next_gpus)
                candidate = (
                    cost + option_cost,
                    (*choices, (offer, safe_disk, score)),
                )
                if key not in updated or candidate[0] < updated[key][0]:
                    updated[key] = candidate
        states = updated
    selected_state = next(
        (
            states[(machine_count, len(backbone_workers))]
            for machine_count in range(maximum_machines, 0, -1)
            if (machine_count, len(backbone_workers)) in states
        ),
        None,
    )
    if selected_state is None:
        maximum_capacity = max((gpus for _, gpus in states), default=0)
        raise RuntimeError(
            "live grouped consumer fleet cannot supply 92 backbone GPUs; "
            f"maximum planned capacity was {maximum_capacity}"
        )
    _, selected_backbone = selected_state
    bins: list[dict[str, Any]] = [
        {
            "offer": offer,
            "safe_disk_gb": safe_disk,
            "selection_score": score,
            "capacity": offer.gpu_count,
            "assigned": [],
            "download_bytes": 0,
        }
        for offer, safe_disk, score in selected_backbone
    ]
    for worker in sorted(
        backbone_workers,
        key=lambda row: (-int(row["download_bytes_cold_cache"]), str(row["worker_id"])),
    ):
        available = [row for row in bins if len(row["assigned"]) < row["capacity"]]
        if not available:
            raise AssertionError("E025 grouped planner lost a backbone GPU slot")
        target = min(
            available,
            key=lambda row: (
                int(row["download_bytes"]),
                len(row["assigned"]),
                int(row["offer"].offer_id),
            ),
        )
        target["assigned"].append(worker)
        target["download_bytes"] += int(worker["download_bytes_cold_cache"])
    for index, row in enumerate(sorted(bins, key=lambda value: value["offer"].offer_id)):
        offer = row["offer"]
        assigned = sorted(row["assigned"], key=lambda value: str(value["worker_id"]))
        disk = _group_disk_gb(assigned, minimum_disk_gb)
        if len(assigned) != offer.gpu_count or disk > offer.disk_space_gb:
            raise RuntimeError("selected Vast bundle cannot safely hold its assigned workers")
        actual_score = acquisition_score(
            offer,
            required_download_bytes=sum(
                int(worker["download_bytes_cold_cache"]) for worker in assigned
            ),
            disk_gb=disk,
        )
        used_offer_ids.add(offer.offer_id)
        used_machine_ids.add(offer.machine_id)
        selected_groups.append(
            {
                "instance_group_id": f"e025-backbone-group-{index:03d}",
                "role": "BACKBONE_GROUP",
                "selected_offer": asdict(offer),
                "selection_score": actual_score,
                "disk_gb": disk,
                "workers": [
                    {
                        **worker,
                        "gpu_slot": slot,
                        "container_port": 42525 + slot,
                    }
                    for slot, worker in enumerate(assigned)
                ],
            }
        )

    for group in selected_groups:
        selected = Offer(**group["selected_offer"])
        capacity = len(group["workers"])
        role = str(group["workers"][0]["role"])
        alternate_rows: list[dict[str, Any]] = []
        for offer in candidates:
            if offer.offer_id in used_offer_ids or offer.machine_id in used_machine_ids:
                continue
            if offer.gpu_count < capacity or not offer.qualifies(
                role,
                disk_gb=int(group["disk_gb"]),
                allow_multi_gpu=offer.gpu_count > 1,
            ):
                continue
            if offer.direct_port_count < capacity:
                continue
            if (role != "SUB_LAYER_WORKER" and not _backbone_name(offer.gpu_name)):
                continue
            score = acquisition_score(
                offer,
                required_download_bytes=sum(
                    int(worker["download_bytes_cold_cache"])
                    for worker in group["workers"]
                ),
                disk_gb=int(group["disk_gb"]),
            )
            alternate_rows.append({"offer": asdict(offer), "score": score})
        alternate_rows.sort(
            key=lambda row: (
                float(row["score"]["short_run_acquisition_score"]),
                -float(row["offer"]["reliability"]),
                int(row["offer"]["offer_id"]),
            )
        )
        group["alternates"] = alternate_rows[:alternates_per_group]
        group["selected_gpu_slots"] = list(range(capacity))
        group["unused_rented_gpu_slots"] = []
        group["selected_machine_is_distinct"] = selected.machine_id in used_machine_ids

    selected_groups.sort(key=lambda row: str(row["instance_group_id"]))
    flattened: list[dict[str, Any]] = []
    for group in selected_groups:
        for worker in group["workers"]:
            flattened.append(
                {
                    **worker,
                    "instance_group_id": group["instance_group_id"],
                    "group_disk_gb": group["disk_gb"],
                    "selected_offer": group["selected_offer"],
                    "selection_score": group["selection_score"],
                    "alternates": group["alternates"],
                }
            )
    flattened.sort(key=lambda row: str(row["worker_id"]))
    gpu_names = sorted(
        {str(group["selected_offer"]["gpu_name"]) for group in selected_groups}
    )
    backbone_gpu_names = sorted(
        {
            str(group["selected_offer"]["gpu_name"])
            for group in selected_groups
            if group["role"] in {"BACKBONE_GROUP", "SUB_LAYER_PARENT"}
        }
    )
    maximum_bootstrap_seconds = max(
        float(row["selection_score"]["expected_bootstrap_seconds"])
        for row in selected_groups
    )
    maximum_expected_ready_seconds = max(
        float(
            row["selection_score"].get(
                "expected_ready_seconds",
                row["selection_score"]["expected_bootstrap_seconds"],
            )
        )
        for row in selected_groups
    )
    acquisition_feasible = maximum_expected_ready_seconds <= maximum_candidate_ready_seconds
    return {
        "schema_version": "experiment-025-grouped-fleet-plan-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if acquisition_feasible else "NO_GO",
        "on_demand_only": True,
        "consumer_only": True,
        "one_isolated_process_per_physical_gpu": True,
        "worker_count": len(flattened),
        "instance_group_count": len(selected_groups),
        "unique_selected_offer_count": len(selected_groups),
        "unique_selected_machine_count": len(selected_groups),
        "selected_gpu_models": gpu_names,
        "compatibility_canaries_required": [
            name for name in backbone_gpu_names if "3090" not in name
        ],
        "excluded_prior_failed_machine_ids": sorted(excluded_machines),
        "expected_acquisition_within_25_minute_deadline": acquisition_feasible,
        "sub_layer_machine_ids": sorted(
            int(row["selected_offer"]["machine_id"])
            for row in selected_groups
            if row["role"] == "SUB_LAYER_WORKER"
        ),
        "total_active_rental_rate_usd_per_hour": sum(
            Offer(**row["selected_offer"]).gpu_rental_rate_per_hour
            for row in selected_groups
        ),
        "total_effective_rate_including_requested_storage_usd_per_hour": sum(
            Offer(**row["selected_offer"]).effective_rate_per_hour(
                float(row["disk_gb"])
            )
            for row in selected_groups
        ),
        "total_expected_ingress_bytes": sum(
            int(row["download_bytes_cold_cache"]) for row in flattened
        ),
        "total_expected_ingress_cost_usd": sum(
            float(row["selection_score"]["expected_ingress_cost_usd"])
            for row in selected_groups
        ),
        "total_expected_bootstrap_active_cost_usd": sum(
            float(row["selection_score"]["expected_active_cost_usd"])
            for row in selected_groups
        ),
        "total_expected_bootstrap_storage_cost_usd": sum(
            float(row["selection_score"]["expected_storage_cost_usd"])
            for row in selected_groups
        ),
        "maximum_expected_bootstrap_seconds": maximum_bootstrap_seconds,
        "maximum_expected_ready_seconds": maximum_expected_ready_seconds,
        "maximum_candidate_ready_seconds": maximum_candidate_ready_seconds,
        "instance_groups": selected_groups,
        "workers": flattened,
    }


def snapshot_and_rank(
    *,
    output_path: Path,
    worker_requirements: list[dict[str, Any]],
    disk_gb: int,
    excluded_machine_ids: set[int] | None = None,
    acquisition_history: Mapping[int, Mapping[str, Any]] | None = None,
    scoring_policy: Mapping[str, Any] | None = None,
    maximum_candidate_ready_seconds: float = 20 * 60,
    stage_ttl_seconds: float = 45 * 60,
    executable: str = "vastai",
) -> dict[str, Any]:
    client = VastClient(executable=executable)
    backbone = client.search_offers(
        gpu_names=HEADLINE_BACKBONE_GPU_NAMES,
        storage_gb=disk_gb,
        single_gpu_only=False,
    )
    small = client.search_offers(
        gpu_names=PREFERRED_SMALL_GPU_NAMES,
        storage_gb=disk_gb,
    )
    offers = {offer.offer_id: offer for offer in [*backbone, *small]}
    plan = rank_grouped_offers_for_workers(
        list(offers.values()),
        worker_requirements,
        minimum_disk_gb=disk_gb,
        excluded_machine_ids=excluded_machine_ids,
        acquisition_history=acquisition_history,
        scoring_policy=scoring_policy,
        maximum_candidate_ready_seconds=maximum_candidate_ready_seconds,
    )
    budget = client.user_budget()
    maximum_full_stage_cost = (
        float(plan["total_effective_rate_including_requested_storage_usd_per_hour"])
        * stage_ttl_seconds
        / 3600.0
        + float(plan["total_expected_ingress_cost_usd"])
    )
    budget["maximum_stage_plus_expected_ingress_usd"] = (
        maximum_full_stage_cost
    )
    if stage_ttl_seconds == 45 * 60:
        budget["maximum_45_minute_stage_plus_expected_ingress_usd"] = (
            maximum_full_stage_cost
        )
    budget["stage_ttl_seconds"] = stage_ttl_seconds
    budget["required_safety_multiplier"] = 1.10
    budget["sufficient"] = (
        float(budget["conservative_available_usd"])
        >= maximum_full_stage_cost * 1.10
    )
    if budget["sufficient"] is not True:
        plan["status"] = "NO_GO"
    payload = {
        "schema_version": OFFER_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS" if plan["status"] == "PASS" else "NO_GO",
        "vast_cli_version": client.version(),
        "query_mode": "READ_ONLY_ON_DEMAND",
        "raw_offer_count": len(offers),
        "excluded_prior_failed_machine_ids": sorted(excluded_machine_ids or set()),
        "redacted_budget": budget,
        "offers": [asdict(offers[key]) for key in sorted(offers)],
        "fleet_plan": plan,
    }
    atomic_write_json(output_path, payload)
    return payload


def destroy_all_from_ledger(
    *,
    ledger_path: Path,
    run_id: str,
    reason: str,
    executable: str = "vastai",
    attempts: int = 3,
) -> dict[str, Any]:
    ledger = AppendOnlyLifecycleLedger(ledger_path, run_id)
    client = VastClient(executable=executable, ledger=ledger)
    known_ids = set(ledger.instance_ids())
    live_before = client.show_instances()
    label_prefix = f"e025-{run_id}-".lower()
    labeled_ids = {
        _integer(row.get("id", row.get("instance_id")))
        for row in live_before
        if str(row.get("label", "")).lower().startswith(label_prefix)
    }
    labeled_ids.discard(-1)
    recovered_ids = sorted(labeled_ids - known_ids)
    if recovered_ids:
        ledger.append(
            "LABELED_INSTANCES_RECOVERED_FOR_CLEANUP",
            command_category="show instances",
            instance_id=None,
            recovered_instance_ids=recovered_ids,
            final_status="CLEANUP_TARGET",
        )
    target_ids = sorted(known_ids | labeled_ids)
    failures: set[int] = set(target_ids)
    for attempt in range(1, attempts + 1):
        if not failures:
            break
        current = sorted(failures)
        failures.clear()
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(current)))) as pool:
            futures = {
                pool.submit(
                    client.destroy_instance,
                    instance_id,
                    reason=f"{reason}; attempt={attempt}",
                ): instance_id
                for instance_id in current
            }
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    if not future.result():
                        failures.add(instance_id)
                except BaseException:
                    failures.add(instance_id)
        if failures and attempt < attempts:
            time.sleep(2.0)
    live_rows = client.show_instances()
    survivor_rows = [
        row
        for row in live_rows
        if _integer(row.get("id", row.get("instance_id"))) in set(target_ids)
        or str(row.get("label", "")).lower().startswith(label_prefix)
    ]
    survivors = sorted(
        {
            _integer(row.get("id", row.get("instance_id")))
            for row in survivor_rows
            if _integer(row.get("id", row.get("instance_id"))) >= 0
        }
    )
    other_live_e025 = sorted(
        {
            _integer(row.get("id", row.get("instance_id")))
            for row in live_rows
            if str(row.get("label", "")).lower().startswith("e025-")
            and not str(row.get("label", "")).lower().startswith(label_prefix)
            and _integer(row.get("id", row.get("instance_id"))) >= 0
        }
    )
    zero_live_e025 = not survivors and not other_live_e025
    receipt = {
        "schema_version": "experiment-025-cleanup-verification-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if zero_live_e025 else "FAIL",
        "run_id": run_id,
        "target_instance_ids": target_ids,
        "label_recovered_instance_ids": recovered_ids,
        "label_prefix": label_prefix,
        "surviving_instance_ids": survivors,
        "other_live_e025_instance_ids": other_live_e025,
        "zero_live_e025_instances": zero_live_e025,
        "unrelated_instances_destroyed": False,
        "reason": reason,
    }
    ledger.append(
        "CLEANUP_VERIFIED" if zero_live_e025 else "CLEANUP_INCOMPLETE",
        command_category="show instances",
        instance_id=None,
        surviving_instance_ids=survivors,
        other_live_e025_instance_ids=other_live_e025,
        final_status=receipt["status"],
    )
    return receipt


def summarize_lifecycle_costs(ledger_path: Path, run_id: str) -> dict[str, Any]:
    """Derive a transparent elapsed-time cost estimate from immutable ledger facts."""

    rows = AppendOnlyLifecycleLedger(ledger_path, run_id).entries()
    creates = {
        int(row["instance_id"]): row
        for row in rows
        if row.get("event") == "CREATE_CONFIRMED"
        and row.get("instance_id") is not None
    }
    summaries: list[dict[str, Any]] = []
    for instance_id, create in sorted(creates.items()):
        destroy_rows = [
            row
            for row in rows
            if row.get("event") == "DESTROY_CONFIRMED"
            and int(row.get("instance_id", -1)) == instance_id
        ]
        created_at = str(create.get("creation_time") or create["timestamp"])
        destroyed_at = (
            str(destroy_rows[0].get("destroy_confirmed_time") or destroy_rows[0]["timestamp"])
            if destroy_rows
            else None
        )
        duration_seconds = (
            max(
                0.0,
                (
                    datetime.fromisoformat(destroyed_at)
                    - datetime.fromisoformat(created_at)
                ).total_seconds(),
            )
            if destroyed_at
            else None
        )
        active_rate = float(create.get("active_rental_rate_usd_per_hour", 0.0))
        storage_rate = float(create.get("storage_rate_usd_per_gb_month", 0.0))
        disk_gb = float(create.get("requested_disk_gb", 0.0))
        active_cost = (
            active_rate * duration_seconds / 3600.0
            if duration_seconds is not None
            else None
        )
        storage_cost = (
            storage_rate * disk_gb * duration_seconds / (30.0 * 24.0 * 3600.0)
            if duration_seconds is not None
            else None
        )
        summaries.append(
            {
                "instance_id": instance_id,
                "offer_id": create.get("offer_id"),
                "machine_id": create.get("machine_id"),
                "gpu_model": create.get("gpu_model"),
                "gpu_count": create.get("gpu_count"),
                "created_at_utc": created_at,
                "destroyed_at_utc": destroyed_at,
                "elapsed_seconds": duration_seconds,
                "active_rental_rate_usd_per_hour": active_rate,
                "requested_disk_gb": disk_gb,
                "storage_rate_usd_per_gb_month": storage_rate,
                "estimated_active_cost_usd": active_cost,
                "estimated_storage_cost_usd": storage_cost,
                "destroy_receipt_present": bool(destroy_rows),
            }
        )
    complete = bool(summaries) and all(
        row["elapsed_seconds"] is not None for row in summaries
    )
    return {
        "schema_version": "experiment-025-ledger-cost-summary-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if complete else "INCOMPLETE",
        "basis": (
            "ledger elapsed time multiplied by advertised active and storage rates; "
            "this is not represented as a provider invoice"
        ),
        "instance_count": len(summaries),
        "instances": summaries,
        "total_elapsed_instance_seconds": sum(
            float(row["elapsed_seconds"] or 0.0) for row in summaries
        ),
        "estimated_active_cost_usd": sum(
            float(row["estimated_active_cost_usd"] or 0.0) for row in summaries
        ),
        "estimated_storage_cost_usd": sum(
            float(row["estimated_storage_cost_usd"] or 0.0) for row in summaries
        ),
    }


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "OFFER_SCHEMA_VERSION",
    "AppendOnlyLifecycleLedger",
    "Offer",
    "VastClient",
    "destroy_all_from_ledger",
    "rank_grouped_offers_for_workers",
    "rank_offers_for_workers",
    "snapshot_and_rank",
    "summarize_lifecycle_costs",
]

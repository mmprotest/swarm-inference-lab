from __future__ import annotations

from swarm_inference.experiments.fanout_analysis import count_is_stable
from swarm_inference.experiments.fanout_resources import (
    aggregate_nvml_process_memory,
    parse_nvidia_gpu_query,
    parse_nvml_process_memory,
    parse_windows_gpu_process_memory,
)


def test_nvml_process_and_aggregate_memory_parsing() -> None:
    rows = parse_nvml_process_memory("100, 256\n200, 512\n")
    assert rows == [
        {
            "process_id": 100,
            "nvml_gpu_memory_bytes": 256 * 1024 * 1024,
            "gpu_process_memory_source": "nvidia-smi-compute-apps",
        },
        {
            "process_id": 200,
            "nvml_gpu_memory_bytes": 512 * 1024 * 1024,
            "gpu_process_memory_source": "nvidia-smi-compute-apps",
        },
    ]
    assert aggregate_nvml_process_memory(rows) == 768 * 1024 * 1024
    missing = parse_nvml_process_memory("100, N/A\n")
    assert aggregate_nvml_process_memory(missing) is None


def test_windows_pdh_process_memory_parsing_and_pid_filter() -> None:
    text = (
        '"(PDH-CSV 4.0)",'
        '"\\\\host\\GPU Process Memory(pid_100_luid_a_phys_0)\\Dedicated Usage",'
        '"\\\\host\\GPU Process Memory(pid_100_luid_b_phys_0)\\Dedicated Usage",'
        '"\\\\host\\GPU Process Memory(pid_200_luid_a_phys_0)\\Dedicated Usage"\n'
        '"date","1024.0","2048.0","4096.0"\n'
    )
    assert parse_windows_gpu_process_memory(text, process_ids=[100]) == [
        {
            "process_id": 100,
            "nvml_gpu_memory_bytes": None,
            "gpu_process_memory_bytes": 3072,
            "gpu_process_memory_source": "windows-pdh-dedicated-usage",
        }
    ]


def test_system_gpu_query_and_missing_resource_thresholds() -> None:
    parsed = parse_nvidia_gpu_query("1024, 3072, 4096, 90, 40, 300, 70, 2500, 14000, 10, 20\n")
    assert parsed["gpu_memory_total_mib"] == 4096
    rows = [
        {
            "phase": "warm",
            "passed": True,
            "exact_token_identity": True,
            "direct_data_plane": True,
            "clean_shutdown": True,
            "worker_crash": False,
            "oom": False,
            "request_timeout": False,
            "stale_cache": False,
            "peak_gpu_memory_fraction": None,
            "peak_system_memory_fraction": 0.5,
            "warm_output_tokens_per_second": 10,
        }
        for _ in range(3)
    ]
    assert not count_is_stable(
        rows,
        repeats=3,
        max_gpu_memory_fraction=0.95,
        max_system_memory_fraction=0.90,
    )

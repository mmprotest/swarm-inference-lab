from __future__ import annotations

import json
from pathlib import Path

from swarm_inference.experiments.experiment_014.compatibility import (
    build_cuda_operation_matrix,
)


def test_cuda_operation_matrix_fails_closed_and_counts_reuse(tmp_path: Path) -> None:
    kimi = tmp_path / "kimi_k3.c"
    cuda = tmp_path / "backend_cuda.cu"
    makefile = tmp_path / "Makefile"
    output = tmp_path / "k3-cuda-operation-matrix.json"
    kimi.write_text("static void w_matmul(void) {}\n", encoding="utf-8")
    cuda.write_text(
        "\n".join(
            (
                "coli_cuda_matmul",
                "coli_cuda_pipe_gemm",
                "coli_cuda_pipe_rmsnorm",
                "coli_cuda_pipe_router",
                "coli_cuda_attention_absorb_batch",
                "weighted_sum_rows",
                "sum_slots",
            )
        ),
        encoding="utf-8",
    )
    makefile.write_text("kimi_k3$(EXE): kimi_k3.c\n", encoding="utf-8")

    receipt = build_cuda_operation_matrix(kimi, cuda, makefile, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["status"] == "FAIL"
    assert payload["hypothesis"]["result"] == "FALSIFIED"
    assert payload["summary"]["critical_operation_classes"] == 11
    assert payload["summary"]["wiring_only_candidates"] == 5
    assert payload["summary"]["cuda_ready"] == 0
    assert {row["readiness"] for row in payload["operations"]} == {"NOT_CUDA_READY"}

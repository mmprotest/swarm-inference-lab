from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.local_dispatch import (
    run_local_native_dispatch,
)
from swarm_inference.experiments.experiment_025.rehearsal import run_local_rehearsal
from swarm_inference.experiments.experiment_025.secrets import (
    create_transport_material,
    load_transport_material,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the zero-spend E025 Kimi rehearsal")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\models\Kimi-K3"))
    parser.add_argument(
        "--cuda-library",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("artifacts/experiment-014/oracle-full-93-idot0"),
    )
    parser.add_argument(
        "--graph-certification",
        type=Path,
        default=Path(
            "artifacts/experiment-014/cuda/h014-038-regression-full-93-layer.json"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.run_root.resolve()
    run_id = root.name.removeprefix("experiment-025-")
    private_root = Path(".e025-private").resolve() / run_id
    material = (
        load_transport_material(private_root, run_id)
        if private_root.exists()
        else create_transport_material(private_root, run_id)
    )
    native_dispatch_path = root / "preflight" / "local-native-dispatch.json"
    native_dispatch = (
        json.loads(native_dispatch_path.read_text(encoding="utf-8"))
        if native_dispatch_path.is_file()
        else run_local_native_dispatch(
            checkpoint=arguments.checkpoint.resolve(),
            cuda_library=arguments.cuda_library.resolve(),
            placement_path=root / "preflight" / "physical-placement.json",
            oracle_trace=(arguments.oracle_root / "hidden-trace.f32").resolve(),
            oracle_routes=(arguments.oracle_root / "routes.txt").resolve(),
            credential_path=Path(material["credential_path"]),
            certificate=Path(material["certificate_path"]),
            private_key=Path(material["private_key_path"]),
            output_path=native_dispatch_path,
        )
    )
    if native_dispatch.get("status") != "PASS":
        print(json.dumps(native_dispatch, indent=2, sort_keys=True))
        return 2
    receipt = run_local_rehearsal(
        checkpoint=arguments.checkpoint.resolve(),
        cuda_library=arguments.cuda_library.resolve(),
        oracle_root=arguments.oracle_root.resolve(),
        graph_certification=arguments.graph_certification.resolve(),
        placement_path=root / "preflight" / "physical-placement.json",
        test_receipt_path=root / "preflight" / "tests.json",
        native_dispatch_path=native_dispatch_path,
        full_graph_output=root / "preflight" / "local-full-graph.json",
        sub_layer_output=root / "preflight" / "local-layer89-four-worker.json",
        output_path=root / "preflight" / "local-rehearsal.json",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

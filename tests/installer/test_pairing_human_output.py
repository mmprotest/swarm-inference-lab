from __future__ import annotations

import json
from pathlib import Path

from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.commands._common import redact_text
from swarm_inference.commands.cluster import _deliver_pairing, _emit_join_command
from swarm_inference.protocol.cluster import PairingCreateResponse

URI = "swarm://192.168.1.20:55000/join/opaque_single_use_data"


def test_human_pairing_prints_one_complete_copy_paste_command(capsys) -> None:
    _emit_join_command(URI, cluster_name="villani-home")
    output = capsys.readouterr().out
    assert output.count(URI) == 1
    assert output == (
        "Cluster ready: villani-home\n\n"
        "Run this command on the machine joining the cluster:\n\n"
        f'swarm node join "{URI}"\n'
    )


def test_json_pairing_remains_secret_free_and_uses_protected_file(
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "state")
    response = PairingCreateResponse(
        session_id="session-1",
        pairing_uri=URI,
        redacted_uri="swarm://192.168.1.20:55000/join/REDACTED",
        expires_at_unix_ns=2_000_000_000,
    )
    result = _deliver_pairing(
        state,
        response,
        json_output=True,
        pairing_output=None,
        force=False,
    )
    document = json.dumps(result.model_dump(mode="json"))
    assert URI not in document
    assert result.invitation_file is not None
    assert result.invitation_file.read_text(encoding="utf-8") == URI
    assert URI not in redact_text(f"join failed: {URI}")

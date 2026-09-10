from swarm_inference.experiments.experiment_026 import vast
from swarm_inference.experiments.experiment_026.io import write_once, append_event


def test_budget_charges_storage_network_and_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(vast, "ROOT", tmp_path)
    write_once(tmp_path / "leases/e026-a-test.json", {"hourly_usd": .25, "network_reserve_usd": .3,
                                                    "deadline_epoch": 10800})
    append_event(tmp_path / "ledger.jsonl", {"event": "CREATE_ATTEMPT", "label": "e026-a-test",
                                            "timestamp": "1970-01-01T00:00:00+00:00"})
    assert vast.estimate(3600) == {"estimated_upper_cost_usd": .55, "reserved_future_rental_usd": .5}
    append_event(tmp_path / "ledger.jsonl", {"event": "ABSENCE_CONFIRMED", "label": "e026-a-test",
                                            "timestamp": "1970-01-01T01:00:00+00:00"})
    assert vast.estimate(7200) == {"estimated_upper_cost_usd": .55, "reserved_future_rental_usd": 0.0}


def test_destroy_requires_owned_lease(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(vast, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside E026"):
        vast.destroy("unrelated-user-instance", 42, "test")


def test_append_only_extension_changes_effective_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(vast,"ROOT",tmp_path)
    lease={"hourly_usd":.25,"network_reserve_usd":.3,"deadline_epoch":10800}
    write_once(tmp_path/"leases/e026-a-test.json",lease)
    append_event(tmp_path/"ledger.jsonl",{"event":"CREATE_ATTEMPT","label":"e026-a-test",
                                          "timestamp":"1970-01-01T00:00:00+00:00"})
    append_event(tmp_path/"ledger.jsonl",{"event":"LEASE_EXTENDED","label":"e026-a-test",
                                          "new_deadline_epoch":14400})
    assert vast.effective_deadline("e026-a-test",lease)==14400
    assert vast.estimate(3600)["reserved_future_rental_usd"]==.75

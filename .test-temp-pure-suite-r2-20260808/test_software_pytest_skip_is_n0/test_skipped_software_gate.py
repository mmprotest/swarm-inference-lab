import pytest

@pytest.mark.skip(reason='fixture unavailable')
def test_software_gate():
    raise AssertionError('must not run')

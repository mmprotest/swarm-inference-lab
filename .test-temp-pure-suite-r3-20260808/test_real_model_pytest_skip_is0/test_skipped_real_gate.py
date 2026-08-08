import pytest

@pytest.mark.skip(reason='fixture unavailable')
def test_real_gate():
    raise AssertionError('must not run')

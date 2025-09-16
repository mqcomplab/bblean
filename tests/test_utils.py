import pytest

from bblean.bitbirch import _min_safe_uint  # type: ignore


def test_min_safe_uint() -> None:
    assert _min_safe_uint(254) == "uint8"
    assert _min_safe_uint(255) == "uint8"
    assert _min_safe_uint(256) == "uint16"
    assert _min_safe_uint(65534) == "uint16"
    assert _min_safe_uint(65535) == "uint16"
    assert _min_safe_uint(65536) == "uint32"
    assert _min_safe_uint(4294967295) == "uint32"
    assert _min_safe_uint(4294967296) == "uint64"
    assert _min_safe_uint(18446744073709551615) == "uint64"
    with pytest.raises(ValueError):
        _min_safe_uint(18446744073709551616)

"""Smoke test: the package imports and exposes a version string."""

import monogenet


def test_version_is_string() -> None:
    assert isinstance(monogenet.__version__, str)
    assert monogenet.__version__ == "0.0.1"

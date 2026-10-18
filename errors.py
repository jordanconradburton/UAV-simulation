"""
errors.py

The package's own exception types, in one module with no dependencies.
"""


class PlacementError(RuntimeError):
    """Sensors could not be placed at the requested spacing in the given region."""


class VerificationError(RuntimeError):
    """A shard was written but failed its own integrity check."""


class InjectionError(RuntimeError):
    """A source dataset or an intermediate result failed a correctness check."""

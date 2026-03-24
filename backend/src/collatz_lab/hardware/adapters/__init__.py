"""OS-specific hardware probes (display adapters, PCI, etc.)."""

from .display import probe_display_adapters

__all__ = ["probe_display_adapters"]

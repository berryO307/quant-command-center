"""
mdg_analysis — Python analysis layer for the C++ market data gateway.

Public API:
    from mdg_analysis import reader, metrics, plots, style
"""

from . import reader, metrics, plots, style

__all__ = ["reader", "metrics", "plots", "style"]
__version__ = "0.1.0"

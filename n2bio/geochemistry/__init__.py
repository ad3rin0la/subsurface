"""
n2bio.geochemistry
==================
Geochemical speciation sub-package for the N2Bio framework.

Exports
-------
NH3Speciation : NH₃/NH₄⁺ acid-base speciation model.
"""

from .speciation import NH3Speciation

__all__ = ["NH3Speciation"]

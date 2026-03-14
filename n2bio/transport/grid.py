"""
transport/grid.py
=================
1D Integral Finite Difference Method (IFDM) grid for TOUGHREACT-style
subsurface reactive transport simulations.

The IFDM discretises the domain into volume elements (cells).  Mass
balance equations are written per cell, with fluxes at the interfaces
between cells computed from Darcy's law.

Reference
---------
Narasimhan, T.N.; Witherspoon, P.A. (1976). Water Resour. Res. 12, 57-64.
Pruess, K. (1991). TOUGH2 – A General-Purpose Numerical Simulator for
    Multiphase Fluid and Heat Flow. LBL-29400.
"""

import numpy as np
from typing import List, Tuple, Optional


class IFDMGrid1D:
    """
    1D Integral Finite Difference grid.

    Nodes are at cell centres.  Connections (interfaces) link adjacent
    cells.  All geometric quantities are pre-computed at construction.

    Parameters
    ----------
    n_cells : int
        Number of grid cells.
    lengths : array-like
        Cell lengths [m].  Length n_cells.
    cross_area : float
        Cross-sectional area of the column [m²].  Uniform for 1D.

    Attributes
    ----------
    volumes : np.ndarray
        Cell pore volumes (= lengths * cross_area) [m³].
    centers : np.ndarray
        Positions of cell centres [m].
    connections : list of (int, int)
        Index pairs (i, j) for each interface between adjacent cells.
    interface_areas : np.ndarray
        Cross-sectional area at each interface [m²].
    d_ij : np.ndarray
        Centre-to-centre distances for each connection [m].
    """

    def __init__(self,
                 n_cells: int,
                 lengths,
                 cross_area: float = 1.0):
        self.n_cells     = n_cells
        self.lengths     = np.asarray(lengths, dtype=float)
        self.cross_area  = cross_area

        if len(self.lengths) != n_cells:
            raise ValueError(
                f"lengths array must have {n_cells} elements, "
                f"got {len(self.lengths)}"
            )

        # Cell volumes [m³]
        self.volumes = self.lengths * cross_area

        # Cell centres: cumulative sum minus half the current cell length
        self.centers = np.cumsum(self.lengths) - self.lengths / 2.0

        # Interface connections: (i, i+1) for each pair of adjacent cells
        self.connections: List[Tuple[int, int]] = [
            (i, i + 1) for i in range(n_cells - 1)
        ]

        # Interface areas (uniform for rectangular column)
        self.interface_areas = np.full(n_cells - 1, cross_area)

        # Centre-to-centre distances for each connection
        self.d_ij = np.diff(self.centers)

    # ------------------------------------------------------------------
    # Class constructors
    # ------------------------------------------------------------------

    @classmethod
    def uniform(cls, n_cells: int, domain_length: float,
                cross_area: float = 1.0) -> 'IFDMGrid1D':
        """
        Create a uniform grid with equal cell lengths.

        Parameters
        ----------
        n_cells : int       Number of cells.
        domain_length : float  Total domain length [m].
        cross_area : float  Cross-sectional area [m²].

        Returns
        -------
        IFDMGrid1D
        """
        lengths = np.full(n_cells, domain_length / n_cells)
        return cls(n_cells, lengths, cross_area)

    @classmethod
    def logarithmic(cls, n_cells: int, domain_length: float,
                    refinement: float = 3.0,
                    cross_area: float = 1.0) -> 'IFDMGrid1D':
        """
        Create a logarithmically-spaced grid (fine near x=0, coarse far).

        Useful for injection-well problems where fronts develop near x=0.

        Parameters
        ----------
        n_cells : int
        domain_length : float  [m]
        refinement : float     Ratio of last-to-first cell length.
        cross_area : float     [m²]

        Returns
        -------
        IFDMGrid1D
        """
        # Geometrically spaced cell lengths
        r = refinement ** (1.0 / (n_cells - 1))
        lengths_raw = np.array([r ** i for i in range(n_cells)])
        lengths = lengths_raw / lengths_raw.sum() * domain_length
        return cls(n_cells, lengths, cross_area)

    # ------------------------------------------------------------------
    # Transmissibility
    # ------------------------------------------------------------------

    def transmissibility(self, permeability: float,
                         viscosity: float,
                         i: int, j: int) -> float:
        """
        Darcy transmissibility between cells i and j [m³/(Pa·s)].

        T_ij = A_ij * k / (μ * d_ij)

        where d_ij is the distance between cell centres i and j, A_ij is
        the interface area, k is the absolute permeability, and μ is the
        dynamic viscosity.

        For harmonic-mean permeability between two cells:
            k_ij = 2*kᵢ*kⱼ / (kᵢ + kⱼ)
        Here we assume uniform permeability, so k_ij = permeability.

        Parameters
        ----------
        permeability : float  Absolute permeability [m²].
        viscosity : float     Dynamic viscosity [Pa·s].
        i : int               Index of first cell.
        j : int               Index of second cell (must be i±1).

        Returns
        -------
        float  Transmissibility [m³/(Pa·s)].
        """
        conn_idx = min(i, j)   # connection index
        A = self.interface_areas[conn_idx]
        d = self.d_ij[conn_idx]
        return A * permeability / (viscosity * d)

    def transmissibility_array(self, permeability: float,
                                viscosity: np.ndarray) -> np.ndarray:
        """
        Compute transmissibility for all connections.

        Parameters
        ----------
        permeability : float   Absolute permeability [m²] (uniform).
        viscosity : np.ndarray Viscosity per cell [Pa·s].

        Returns
        -------
        np.ndarray  Transmissibility per connection [m³/(Pa·s)].
        """
        T = np.zeros(len(self.connections))
        for idx, (i, j) in enumerate(self.connections):
            mu_avg = 0.5 * (viscosity[i] + viscosity[j])
            T[idx] = self.interface_areas[idx] * permeability / (mu_avg * self.d_ij[idx])
        return T

    # ------------------------------------------------------------------
    # Grid geometry helpers
    # ------------------------------------------------------------------

    @property
    def n_connections(self) -> int:
        """Number of cell interfaces."""
        return len(self.connections)

    @property
    def total_length(self) -> float:
        """Total domain length [m]."""
        return float(self.lengths.sum())

    @property
    def total_volume(self) -> float:
        """Total domain volume (bulk) [m³]."""
        return float(self.volumes.sum())

    def pore_volumes(self, porosity: float) -> np.ndarray:
        """
        Pore volume of each cell [m³].

        Parameters
        ----------
        porosity : float  Formation porosity [-].

        Returns
        -------
        np.ndarray  Pore volumes per cell.
        """
        return self.volumes * porosity

    def cell_index(self, x: float) -> int:
        """
        Return the cell index containing position x [m].

        Parameters
        ----------
        x : float  Position along the column [m].

        Returns
        -------
        int  Zero-based cell index.  Clamped to [0, n_cells-1].
        """
        edges = np.concatenate([[0.0], np.cumsum(self.lengths)])
        idx = np.searchsorted(edges, x, side='right') - 1
        return int(np.clip(idx, 0, self.n_cells - 1))

    def __repr__(self) -> str:
        return (f"IFDMGrid1D(n_cells={self.n_cells}, "
                f"domain={self.total_length:.2f} m, "
                f"cross_area={self.cross_area:.3f} m², "
                f"uniform={np.allclose(self.lengths, self.lengths[0])})")

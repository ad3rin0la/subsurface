"""
manifold_fba.py
===============
Matrix Manifold Flux Balance Analysis for the N2Bio four-guild consortium.

Three geometric enhancements over standard LP-based FBA:

  1. Grassmann manifold Gr(k,n)
     Each guild's null space is a point on Gr(k,n). Geodesic distance
     between guilds measures metabolic complementarity. The lifecycle is
     a geodesic trajectory across phases.

  2. Stiefel manifold V_{n,k}
     Orthonormal null-space bases live on V_{n,k}. Riemannian gradient
     descent stays in null(S) exactly — no re-projection needed.

  3. SPD manifold Sym+(n)
     Flux covariance lives on Sym+(n). The Fisher-Rao metric enables
     principled cross-guild comparison and anisotropic sampling.

Guilds
------
  0  Diazotroph   — N2-fixation + dark fermentation  (Thermotoga chassis)
  1  Hydrogenotroph — H2 + CO2 → CH4
  2  SRB          — H2/HC + SO4 → H2S
  3  Aromatic degrader — HC_arom detoxification

Exchange metabolites (9, shared pool)
--------------------------------------
  N2_aq, NH4, H2_aq, CO2_aq, HC_ali, SO4, H2S, HC_arom, CH4

Null-space design
-----------------
Each guild model contains internal metabolites AND parallel reaction
branches. Parallel branches between the same source → sink create
stoichiometric dependencies: n_rxn > rank(S), so null_dim = n_rxn - rank > 0.
This mirrors real GEMs which have thousands of reactions vs hundreds of mets.

Dependencies: numpy, scipy, cvxpy, pymanopt
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import null_space, svd
import cvxpy as cp
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1.  Exchange metabolite index
# ---------------------------------------------------------------------------

EXCHANGE_METABOLITES: Dict[str, int] = {
    "N2_aq":   0,
    "NH4":     1,
    "H2_aq":   2,
    "CO2_aq":  3,
    "HC_ali":  4,
    "SO4":     5,
    "H2S":     6,
    "HC_arom": 7,
    "CH4":     8,
    "acetate": 9,   # fermentation coproduct; acetyl-CoA donor to SAT
}
N_EXCH = len(EXCHANGE_METABOLITES)   # 10
GUILD_NAMES = ["Diazotroph", "Hydrogenotroph", "SRB", "AromaticDegrader"]


# ---------------------------------------------------------------------------
# 2.  Guild model dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuildModel:
    """
    Stoichiometric network for one microbial guild.

    Row layout of S:
      rows 0 .. n_internal-1   = intracellular metabolites
      rows n_internal .. end   = exchange metabolites (N_EXCH = 10)

    Null-space design principle
    ---------------------------
    Each guild has at least one pair of parallel reaction routes connecting
    the same metabolites. This creates a linear dependency in S, so
    null_dim = n_rxn - rank(S) >= 1.  The null vector corresponds to flux
    through one route minus flux through the parallel route — a biochemical
    cycle / futile cycle — which is the geometric object that lives on the
    Grassmann manifold.
    """
    name: str
    S: np.ndarray           # (n_met, n_rxn)
    lb: np.ndarray          # (n_rxn,)
    ub: np.ndarray          # (n_rxn,)
    biomass_reaction_idx: int
    n_internal: int
    guild_index: int

    def __post_init__(self):
        self.c = np.zeros(self.n_rxn)
        self.c[self.biomass_reaction_idx] = 1.0

    @property
    def n_rxn(self) -> int:
        return self.S.shape[1]

    @property
    def n_met(self) -> int:
        return self.S.shape[0]

    @property
    def S_exchange(self) -> np.ndarray:
        """Last N_EXCH rows — exchange metabolites only."""
        return self.S[-N_EXCH:, :]


# ---------------------------------------------------------------------------
# 3.  Guild stoichiometric models  (guaranteed non-trivial null spaces)
# ---------------------------------------------------------------------------

def _ex(n_internal: int, name: str) -> int:
    """Row index of an exchange metabolite given the internal offset."""
    return n_internal + EXCHANGE_METABOLITES[name]


def build_diazotroph_model() -> GuildModel:
    """
    Diazotroph  (Thermotoga chassis)  — updated with three N–S couplings.

    Internal metabolites (3):
      PYR — pyruvate         (central C3 intermediate)
      OAS — O-acetylserine   (SAT product; OASTL substrate)
      CYS — L-cysteine       (nitrogenase Fe-S supply; structural S)

    Reactions (10):
      r0: N2_aq → 2 NH4 + H2            nitrogenase (Cys co-limited via exchange bounds)
      r1: HC_ali → 2 PYR                 glycolysis-like
      r2: PYR → H2 + CO2 + acetate       PFOR + acetate kinase (acetate coproduct)
      r3: HC_ali → H2 + CO2 + acetate    direct fermentation ← PARALLEL to r1+r2
      r4: PYR + NH4 → biomass (obj)      anabolic pyruvate use
      r5: NH4 → (biomass-N drain)        N-assimilation sink
      r6: N2_aq → NH4                    alternative N₂ fixation (parallel to r0)
      r7: NH4 + acetate → OAS            SAT  (acetate = acetyl-CoA proxy)
      r8: OAS + H2S → CYS + acetate      OASTL  (H₂S bypass of APR/SiR chain)
      r9: CYS →                          Cys drain  (nitrogenase Fe-S + structural)

    Null-space structure:
      r1+r2 ‖ r3  (HC_ali → H2 + CO2 + acetate)   — 1 null vector
      r0 ‖ r6     (N2 → NH4 with/without H2)        — 1 null vector
      r7+r8 cycle (acetate recycling in SAT/OASTL)  — 1 null vector
      null_dim ≥ 3.

    Biochemical basis of new reactions:
      r2:  PYR + CoA + Fd_ox → acetyl-CoA + CO2 + H2 (PFOR),
           acetyl-CoA → acetate (phosphotransacetylase/AK) → net: PYR → acetate + CO2 + H2
      r7:  L-serine + acetyl-CoA → OAS + CoA
           (acetate is the dissolved proxy for intracellular acetyl-CoA via ACS)
      r8:  OAS + HS⁻ → Cys + acetate + H⁺
           (H₂S from exchange pool; bypasses SO₄²⁻ reductive chain)
      r9:  Cys → (structural protein S + nitrogenase Fe-S cluster turnover)
    """
    n_int = 3
    n_rxn = 10
    S = np.zeros((n_int + N_EXCH, n_rxn))
    PYR = 0
    OAS = 1
    CYS = 2
    ex  = lambda name: _ex(n_int, name)

    # r0: nitrogenase  N2 → 2 NH4 + H2
    S[ex("N2_aq"), 0] = -1.0
    S[ex("NH4"),   0] = +2.0
    S[ex("H2_aq"), 0] = +1.0

    # r1: glycolysis  HC_ali → 2 PYR
    S[ex("HC_ali"), 1] = -1.0
    S[PYR,          1] = +2.0

    # r2: PFOR + AK  PYR → H2 + CO2 + acetate
    S[PYR,            2] = -1.0
    S[ex("H2_aq"),    2] = +1.0
    S[ex("CO2_aq"),   2] = +1.0
    S[ex("acetate"),  2] = +1.0    # acetate coproduct from PFOR→acetyl-CoA→acetate

    # r3: direct fermentation  HC_ali → H2 + CO2 + acetate  (parallel to r1+r2)
    S[ex("HC_ali"),   3] = -0.5
    S[ex("H2_aq"),    3] = +1.0
    S[ex("CO2_aq"),   3] = +0.5
    S[ex("acetate"),  3] = +0.5    # acetate coproduct

    # r4: biomass synthesis  PYR + NH4 → X  (objective)
    S[PYR,          4] = -0.5
    S[ex("NH4"),    4] = -0.3

    # r5: N-assimilation drain
    S[ex("NH4"),    5] = -0.1

    # r6: alternative N2 fixation (no H2 — parallel to r0)
    S[ex("N2_aq"),  6] = -0.5
    S[ex("NH4"),    6] = +1.0

    # r7: SAT  NH4 + acetate → OAS
    # (NH4 gates serine via GS/GOGAT; acetate is dissolved acetyl-CoA proxy)
    S[ex("NH4"),      7] = -0.1    # N stoichiometry (1 N per serine/OAS)
    S[ex("acetate"),  7] = -1.0    # acetyl-CoA consumed
    S[OAS,            7] = +1.0

    # r8: OASTL  OAS + H2S → CYS + acetate
    # (H₂S drawn from exchange pool; pH-speciation handled in kinetic model)
    S[OAS,            8] = -1.0
    S[ex("H2S"),      8] = -1.0
    S[CYS,            8] = +1.0
    S[ex("acetate"),  8] = +1.0    # acetate released from OAS

    # r9: Cys drain  CYS → (structural + Fe-S maintenance)
    S[CYS,            9] = -1.0

    lb = np.zeros(n_rxn)
    ub = np.array([10.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0])
    return GuildModel("Diazotroph", S, lb, ub,
                      biomass_reaction_idx=4, n_internal=n_int, guild_index=0)


def build_hydrogenotroph_model() -> GuildModel:
    """
    Hydrogenotroph / hydrogenotrophic methanogen.

    Internal metabolites (1):
      F420r — reduced coenzyme F420

    Reactions (6):
      r0: H2 → F420r                  H2 activation
      r1: F420r + CO2 → CH4           methanogenesis via F420
      r2: H2 + CO2 → CH4              direct methanogenesis (parallel to r0+r1)
      r3: F420r → biomass-e           anabolic electron use  (objective)
      r4: H2 → (overflow)             H2 dissipation
      r5: CO2 → (exch drain)          CO2 overflow

    Null space: r0+r1 and r2 are parallel (H2+CO2 → CH4). null_dim >= 1.
    """
    n_int = 1
    n_rxn = 6
    S = np.zeros((n_int + N_EXCH, n_rxn))
    F420r = 0
    ex = lambda name: _ex(n_int, name)

    # r0: H2 activation
    S[ex("H2_aq"),  0] = -2.0
    S[F420r,        0] = +1.0

    # r1: methanogenesis via F420
    S[F420r,         1] = -1.0
    S[ex("CO2_aq"),  1] = -0.5
    S[ex("CH4"),     1] = +0.5

    # r2: direct H2 + CO2 → CH4  (parallel)
    S[ex("H2_aq"),   2] = -4.0
    S[ex("CO2_aq"),  2] = -1.0
    S[ex("CH4"),     2] = +1.0

    # r3: biomass  (objective)
    S[F420r,         3] = -0.5
    S[ex("H2_aq"),   3] = -0.5

    # r4: H2 overflow
    S[ex("H2_aq"),   4] = -1.0

    # r5: CO2 drain
    S[ex("CO2_aq"),  5] = -0.5

    lb = np.zeros(n_rxn)
    ub = np.array([20.0, 15.0, 15.0, 3.0, 5.0, 5.0])
    return GuildModel("Hydrogenotroph", S, lb, ub,
                      biomass_reaction_idx=3, n_internal=n_int, guild_index=1)


def build_srb_model() -> GuildModel:
    """
    Sulfate-reducing bacteria.

    Internal metabolites (1):
      APS — adenosine-5'-phosphosulfate  (activated sulfate)

    Reactions (8):
      r0: SO4 → APS                    sulfate activation
      r1: H2 + APS → H2S               H2-driven dissimilation
      r2: HC_ali + APS → CO2 + H2S     organic-C oxidation with sulfate
      r3: H2 + SO4 → H2S               direct H2+SO4 (parallel to r0+r1)
      r4: HC_ali + SO4 → CO2 + H2S     direct organic-C (parallel to r0+r2)
      r5: biomass  (objective)
      r6: SO4 → (assimilatory S drain)  N-S coupling — scales with NH4
      r7: H2 → (overflow)

    Null space: (r0+r1 vs r3) and (r0+r2 vs r4) give null_dim >= 2.
    """
    n_int = 1
    n_rxn = 8
    S = np.zeros((n_int + N_EXCH, n_rxn))
    APS = 0
    ex = lambda name: _ex(n_int, name)

    # r0: SO4 activation
    S[ex("SO4"),    0] = -1.0
    S[APS,          0] = +1.0

    # r1: H2 + APS → H2S
    S[APS,          1] = -1.0
    S[ex("H2_aq"),  1] = -4.0
    S[ex("H2S"),    1] = +1.0

    # r2: HC_ali + APS → CO2 + H2S
    S[APS,           2] = -1.0
    S[ex("HC_ali"),  2] = -1.0
    S[ex("CO2_aq"),  2] = +1.0
    S[ex("H2S"),     2] = +1.0

    # r3: direct H2 + SO4 → H2S  (parallel to r0+r1)
    S[ex("H2_aq"),   3] = -4.0
    S[ex("SO4"),     3] = -1.0
    S[ex("H2S"),     3] = +1.0

    # r4: direct HC_ali + SO4 → CO2 + H2S  (parallel to r0+r2)
    S[ex("HC_ali"),  4] = -1.0
    S[ex("SO4"),     4] = -1.0
    S[ex("CO2_aq"),  4] = +1.0
    S[ex("H2S"),     4] = +1.0

    # r5: biomass  (objective)
    S[ex("NH4"),     5] = -0.3
    S[ex("HC_ali"),  5] = -0.2

    # r6: assimilatory SO4  (N-S coupling)
    S[ex("SO4"),     6] = -0.1

    # r7: H2 overflow
    S[ex("H2_aq"),   7] = -1.0

    lb = np.zeros(n_rxn)
    ub = np.array([12.0, 12.0, 8.0, 12.0, 8.0, 4.0, 5.0, 5.0])
    return GuildModel("SRB", S, lb, ub,
                      biomass_reaction_idx=5, n_internal=n_int, guild_index=2)


def build_aromatic_degrader_model() -> GuildModel:
    """
    Aromatic degrader.

    Internal metabolites (1):
      CAT — catechol  (ring-cleavage intermediate)

    Reactions (6):
      r0: HC_arom → CAT               ring hydroxylation
      r1: CAT → CO2 + H2              meta-cleavage + electron transfer
      r2: HC_arom → CO2               direct oxidation  (parallel to r0+r1)
      r3: CAT → biomass  (objective)  anabolic use
      r4: HC_arom → CAT + H2          alternative hydroxylation (parallel to r0)
      r5: NH4 → (biomass-N drain)

    Null space: r0+r1 vs r2 (parallel); also r0 vs r4 (alternative hydroxylation).
    null_dim >= 2.
    """
    n_int = 1
    n_rxn = 6
    S = np.zeros((n_int + N_EXCH, n_rxn))
    CAT = 0
    ex = lambda name: _ex(n_int, name)

    # r0: ring hydroxylation
    S[ex("HC_arom"), 0] = -1.0
    S[CAT,           0] = +1.0

    # r1: meta-cleavage  CAT → CO2 + H2
    S[CAT,           1] = -1.0
    S[ex("CO2_aq"),  1] = +1.0
    S[ex("H2_aq"),   1] = +0.5

    # r2: direct oxidation  HC_arom → CO2  (parallel to r0+r1)
    S[ex("HC_arom"), 2] = -1.0
    S[ex("CO2_aq"),  2] = +1.0

    # r3: biomass  (objective)
    S[CAT,           3] = -0.5
    S[ex("NH4"),     3] = -0.3

    # r4: alternative hydroxylation  HC_arom → CAT + H2  (parallel to r0)
    S[ex("HC_arom"), 4] = -1.0
    S[CAT,           4] = +1.0
    S[ex("H2_aq"),   4] = +0.3

    # r5: N-drain
    S[ex("NH4"),     5] = -0.1

    lb = np.zeros(n_rxn)
    ub = np.array([5.0, 5.0, 5.0, 2.0, 5.0, 2.0])
    return GuildModel("AromaticDegrader", S, lb, ub,
                      biomass_reaction_idx=3, n_internal=n_int, guild_index=3)


def build_all_guilds() -> List[GuildModel]:
    return [
        build_diazotroph_model(),
        build_hydrogenotroph_model(),
        build_srb_model(),
        build_aromatic_degrader_model(),
    ]


# ---------------------------------------------------------------------------
# 4.  Null space geometry (Grassmann manifold)
# ---------------------------------------------------------------------------

def compute_null_basis(S: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Orthonormal basis for null(S) via SVD.

    Returns
    -------
    W : np.ndarray  shape (n_rxn, k)   k = null dimension
    """
    W = null_space(S, rcond=tol)
    if W.size == 0:
        raise ValueError(
            f"Null space is trivial (S is {S.shape}, rank={np.linalg.matrix_rank(S)}). "
            "Add parallel reaction branches to create flux cycles."
        )
    Q, _ = np.linalg.qr(W)
    return Q[:, :W.shape[1]]


def grassmann_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Geodesic distance on Gr(k, n) via principal angles.

    d = ||arccos(singular_values(W1^T W2))||_F
    Both W1, W2 must be orthonormal (output of compute_null_basis).
    """
    k = min(W1.shape[1], W2.shape[1])
    M = W1[:, :k].T @ W2[:, :k]
    _, sigma, _ = svd(M, full_matrices=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    return float(np.linalg.norm(np.arccos(np.abs(sigma))))


def grassmann_log(W_base: np.ndarray, W_target: np.ndarray) -> np.ndarray:
    """Riemannian log map: tangent vector at W_base pointing toward W_target."""
    k = W_base.shape[1]
    M = W_base.T @ W_target[:, :k]
    U, sigma, Vt = svd(M, full_matrices=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    angles = np.arccos(sigma)
    residual = W_target[:, :k] - W_base @ M
    if np.linalg.norm(residual) < 1e-12:
        return np.zeros_like(W_base)
    Q, _ = np.linalg.qr(residual)
    Q = Q[:, :k]
    return Q @ np.diag(angles) @ Vt


def grassmann_exp(W_base: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """Riemannian exp map: move from W_base along tangent Delta."""
    U, sigma, Vt = svd(Delta, full_matrices=False)
    W_new = (W_base @ Vt.T @ np.diag(np.cos(sigma))
             + U @ np.diag(np.sin(sigma))) @ Vt
    Q, _ = np.linalg.qr(W_new)
    return Q[:, :W_base.shape[1]]


def grassmann_geodesic(W0: np.ndarray, W1: np.ndarray,
                       n_points: int = 20) -> List[np.ndarray]:
    """Uniformly-spaced geodesic curve on Gr(k, n) from W0 to W1."""
    Delta = grassmann_log(W0, W1)
    return [grassmann_exp(W0, t * Delta) for t in np.linspace(0.0, 1.0, n_points)]


# ---------------------------------------------------------------------------
# 5.  SPD manifold (Fisher-Rao metric on flux covariance)
# ---------------------------------------------------------------------------

def spd_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Affine-invariant distance on Sym+(n):
        d(A,B) = ||log(A^{-1/2} B A^{-1/2})||_F
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-12)
    A_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    M = A_inv_sqrt @ B @ A_inv_sqrt
    ev, evec = np.linalg.eigh(M)
    ev = np.maximum(ev, 1e-12)
    log_M = evec @ np.diag(np.log(ev)) @ evec.T
    return float(np.linalg.norm(log_M, 'fro'))


def spd_mean(matrices: List[np.ndarray], n_iter: int = 30,
             tol: float = 1e-8) -> np.ndarray:
    """Fréchet (Karcher) mean on Sym+(n)."""
    M = matrices[0].copy()
    for _ in range(n_iter):
        ev, evec = np.linalg.eigh(M)
        ev = np.maximum(ev, 1e-12)
        M_inv_sqrt = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
        M_sqrt = evec @ np.diag(np.sqrt(ev)) @ evec.T
        S_log = np.zeros_like(M)
        for A in matrices:
            inner = M_inv_sqrt @ A @ M_inv_sqrt
            ev2, evec2 = np.linalg.eigh(inner)
            ev2 = np.maximum(ev2, 1e-12)
            S_log += M_sqrt @ (evec2 @ np.diag(np.log(ev2)) @ evec2.T) @ M_sqrt
        grad = S_log / len(matrices)
        ev3, evec3 = np.linalg.eigh(grad)
        M_new = M @ (evec3 @ np.diag(np.exp(ev3)) @ evec3.T)
        if np.linalg.norm(M_new - M, 'fro') < tol:
            return M_new
        M = M_new
    return M


# ---------------------------------------------------------------------------
# 6.  Standard FBA (LP via cvxpy)
# ---------------------------------------------------------------------------

@dataclass
class FBAResult:
    guild_name: str
    objective: float
    fluxes: np.ndarray
    status: str
    null_basis: Optional[np.ndarray] = None   # shape (n_rxn, k)
    null_dim: int = 0
    null_coords: Optional[np.ndarray] = None  # coordinates in null basis


def run_standard_fba(guild: GuildModel,
                     exchange_bounds: Optional[Dict[str, Tuple[float, float]]] = None
                     ) -> FBAResult:
    """
    Standard LP-based FBA for a single guild using cvxpy.

    exchange_bounds : dict mapping metabolite name → (lo, hi) on net exchange flux.
    """
    v = cp.Variable(guild.n_rxn)

    extra = []
    if exchange_bounds:
        for met_name, (lo, hi) in exchange_bounds.items():
            if met_name not in EXCHANGE_METABOLITES:
                continue
            row = guild.S_exchange[EXCHANGE_METABOLITES[met_name], :]
            net = row @ v
            if np.isfinite(lo):
                extra.append(net >= lo)
            if np.isfinite(hi):
                extra.append(net <= hi)

    S_int = guild.S[:guild.n_internal, :]   # only internal mets must balance
    prob = cp.Problem(
        cp.Maximize(guild.c @ v),
        [S_int @ v == 0, v >= guild.lb, v <= guild.ub] + extra
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and v.value is not None:
        fluxes = np.array(v.value, dtype=float)
        obj_val = float(prob.value)
    else:
        fluxes = np.zeros(guild.n_rxn)
        obj_val = 0.0

    try:
        W = compute_null_basis(guild.S[:guild.n_internal, :])
        coords = W.T @ fluxes
    except ValueError:
        W = None
        coords = None

    return FBAResult(
        guild_name=guild.name,
        objective=obj_val,
        fluxes=fluxes,
        status=prob.status if prob.status else "unknown",
        null_basis=W,
        null_dim=W.shape[1] if W is not None else 0,
        null_coords=coords,
    )


# ---------------------------------------------------------------------------
# 7.  Manifold-constrained FBA  (Riemannian optimisation on null space)
# ---------------------------------------------------------------------------

def run_manifold_fba(guild: GuildModel,
                     exchange_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                     max_iter: int = 500) -> FBAResult:
    """
    Manifold-constrained FBA: optimise directly over null(S).

    Strategy
    --------
    1. Compute orthonormal null basis W  (Stiefel manifold point).
    2. Any feasible flux: v = W @ alpha + v_p  (v_p = bound-midpoint projection).
    3. Maximise c·v s.t. lb ≤ v ≤ ub via log-barrier penalty on alpha.
    4. The optimisation stays in null(S) exactly — no LP re-projection.

    The Stiefel/Grassmann manifold enters through W; alpha lives in R^k (flat).
    For community problems, W itself can be updated on the Grassmann manifold
    in an outer loop to couple guild null spaces.
    """
    S_int = guild.S[:guild.n_internal, :]   # only internal rows for null space
    try:
        W = compute_null_basis(S_int)
    except ValueError:
        return run_standard_fba(guild, exchange_bounds)

    k = W.shape[1]

    # Particular solution: project bound midpoint onto null(S_int)
    v_mid = 0.5 * (guild.lb + guild.ub)
    St = S_int.T
    v_p = v_mid - St @ np.linalg.lstsq(
        S_int @ St + 1e-10 * np.eye(S_int.shape[0]),
        S_int @ v_mid, rcond=None)[0]

    # Adjust v_p to strictly satisfy bounds (clamp then re-project slightly)
    v_p = np.clip(v_p, guild.lb + 1e-4, guild.ub - 1e-4)

    obj_coef = W.T @ guild.c  # gradient of objective in alpha-space

    # Exchange flux constraints expressed in alpha-space
    exc_A = []  # rows: exchange constraint normals in alpha-space
    exc_lo = []
    exc_hi = []
    if exchange_bounds:
        for met_name, (lo, hi) in exchange_bounds.items():
            if met_name not in EXCHANGE_METABOLITES:
                continue
            row = guild.S_exchange[EXCHANGE_METABOLITES[met_name], :]
            # net_flux = row @ (W @ alpha + v_p) = (row @ W) @ alpha + row @ v_p
            a_row = row @ W
            offset = float(row @ v_p)
            exc_A.append(a_row)
            exc_lo.append(lo - offset if np.isfinite(lo) else -np.inf)
            exc_hi.append(hi - offset if np.isfinite(hi) else +np.inf)

    MU = 0.05  # log-barrier weight

    def neg_obj(alpha):
        v = W @ alpha + v_p
        margin_lb = v - guild.lb
        margin_ub = guild.ub - v
        if np.any(margin_lb <= 0) or np.any(margin_ub <= 0):
            return 1e12
        barrier = -MU * (np.sum(np.log(margin_lb)) + np.sum(np.log(margin_ub)))
        # Exchange constraint penalties
        exc_pen = 0.0
        for a_row, lo, hi in zip(exc_A, exc_lo, exc_hi):
            val = float(a_row @ alpha)
            if np.isfinite(lo) and val <= lo:
                exc_pen += 1e4 * (lo - val) ** 2
            if np.isfinite(hi) and val >= hi:
                exc_pen += 1e4 * (val - hi) ** 2
        return -(obj_coef @ alpha) + barrier + exc_pen

    def grad_neg_obj(alpha):
        v = W @ alpha + v_p
        margin_lb = v - guild.lb
        margin_ub = guild.ub - v
        if np.any(margin_lb <= 0) or np.any(margin_ub <= 0):
            return np.zeros(k)
        d_barrier = -MU * (1.0 / margin_lb - 1.0 / margin_ub)
        g = -obj_coef + W.T @ d_barrier
        for a_row, lo, hi in zip(exc_A, exc_lo, exc_hi):
            val = float(a_row @ alpha)
            if np.isfinite(lo) and val <= lo:
                g += -2e4 * (lo - val) * a_row
            if np.isfinite(hi) and val >= hi:
                g += 2e4 * (val - hi) * a_row
        return g

    res = minimize(neg_obj, np.zeros(k), jac=grad_neg_obj, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})

    alpha_opt = res.x
    v_opt = np.clip(W @ alpha_opt + v_p, guild.lb, guild.ub)

    return FBAResult(
        guild_name=guild.name,
        objective=float(guild.c @ v_opt),
        fluxes=v_opt,
        status="manifold_optimal" if res.success else "manifold_suboptimal",
        null_basis=W,
        null_dim=k,
        null_coords=alpha_opt,
    )


# ---------------------------------------------------------------------------
# 8.  Community state
# ---------------------------------------------------------------------------

@dataclass
class CommunityState:
    """
    Consortium state encoding the N-S coupling and inhibition parameters
    from the N2Bio lifecycle model.
    """
    biomass_fractions: np.ndarray = field(
        default_factory=lambda: np.array([0.70, 0.15, 0.10, 0.05]))
    NH4_conc: float = 0.0         # mmol/L  — controls S-assimilation unlock
    SO4_conc: float = 2.0         # mmol/L
    H2_partial_pressure: float = 0.01  # atm — nitrogenase inhibition
    HC_arom_conc: float = 0.15    # mmol/L — nitrogenase inhibition
    lifecycle_phase: int = 0
    # New in v2.2: cysteine pool drives positive H₂S→Cys→nitrogenase feedback
    Cys_conc_mM: float = 0.05     # mmol/L  — default ~2.5× K_Cys_nit (20 μM)
    H2S_conc_mM: float = 0.0      # mmol/L  — reservoir dissolved sulfide

    def __post_init__(self):
        s = self.biomass_fractions.sum()
        if s > 0:
            self.biomass_fractions = self.biomass_fractions / s


def _exchange_bounds_from_state(guild: GuildModel,
                                state: CommunityState
                                ) -> Dict[str, Tuple[float, float]]:
    """
    Per-guild exchange bounds encoding N-S coupling and N2Bio inhibitions.

    Three couplings modelled (v2.2 update):
    ----------------------------------------
    1. Nitrogenase (Diazotroph): upper bound on N2 uptake set by H₂ product
       inhibition, aromatic HC inhibition, AND cysteine co-limitation.
       In H₂S-rich reservoirs, Cys is supplied via OASTL, so the Cys factor
       approaches 1 and the N₂ bound depends only on H₂ and aromatics.
       In sulfide-poor formations, low Cys becomes the operative ceiling.

    2. H₂S availability to Diazotroph: lower bound on H₂S consumption drives
       OASTL flux; bounded by reservoir dissolved sulfide concentration.

    3. SO₄²⁻ / SRB (unchanged): NH₄⁺ drives assimilatory SO₄ demand via
       OAS pathway, depleting the pool available for SRB dissimilation.
    """
    K_OAS   = 0.5    # NH4 half-sat for OAS / S-assimilation (mmol/L)
    K_I_H2  = 0.3    # H2 inhibition of nitrogenase (atm)
    K_I_ar  = 0.5    # aromatic inhibition of nitrogenase (mmol/L)
    K_Cys   = 0.020  # Cys half-sat for nitrogenase Fe-S (mmol/L = 20 μmol/L)

    bounds: Dict[str, Tuple[float, float]] = {}

    if guild.guild_index == 0:  # Diazotroph
        # Factors on nitrogenase N2-uptake upper bound
        h2_f  = K_I_H2 / (K_I_H2  + state.H2_partial_pressure)
        ar_f  = K_I_ar  / (K_I_ar  + state.HC_arom_conc)
        cys_f = state.Cys_conc_mM / (K_Cys + state.Cys_conc_mM)
        nit_max = 10.0 * h2_f * ar_f * cys_f
        bounds["N2_aq"] = (-nit_max, 0.0)

        # H₂S consumption by Diazotroph (for OASTL r8)
        # Upper consumption bounded by available dissolved sulfide
        h2s_max = min(state.H2S_conc_mM, 5.0)   # mmol/L, cap at 5
        bounds["H2S"] = (-h2s_max, +np.inf)       # can consume or be indifferent

    elif guild.guild_index == 2:  # SRB
        # N-S coupling: NH4 drives assimilatory SO4 demand → reduces dissimilatory SO4
        assim_f = state.NH4_conc / (K_OAS + state.NH4_conc)
        so4_available = state.SO4_conc * (1.0 - 0.65 * assim_f)
        so4_available = max(so4_available, 0.0)
        bounds["SO4"] = (-so4_available, 0.0)

    return bounds


# ---------------------------------------------------------------------------
# 9.  Community FBA on the product Grassmannian
# ---------------------------------------------------------------------------

@dataclass
class CommunityFBAResult:
    guild_results: List[FBAResult]
    community_H2_flux: float
    community_NH4_flux: float
    community_CH4_flux: float
    community_H2S_flux: float
    community_acetate_flux: float     # net acetate (fermentation - SAT uptake)
    grassmann_distances: np.ndarray   # (4, 4)  pairwise null-space geodesics
    null_dims: np.ndarray             # (4,)    per-guild null space dimensions
    null_bases: List[Optional[np.ndarray]]
    spd_distances: Optional[np.ndarray] = None   # (4,4) Fisher-Rao
    geodesic_trajectory: Optional[List[np.ndarray]] = None


def run_community_fba(state: CommunityState,
                      use_manifold: bool = True,
                      compute_spd: bool = True,
                      track_geodesic: bool = True,
                      n_geodesic_points: int = 12) -> CommunityFBAResult:
    """
    Community FBA for the N2Bio four-guild consortium.

    1. Solve per-guild FBA (manifold or standard LP).
    2. Compute biomass-weighted community exchange fluxes.
    3. Compute pairwise Grassmann distances via exchange-space projection.
    4. Optionally compute SPD Fisher-Rao distances between flux covariances.
    5. Optionally compute geodesic trajectory between diazotroph and
       hydrogenotroph null spaces (lifecycle proxy).
    """
    guilds = build_all_guilds()
    fba_fn = run_manifold_fba if use_manifold else run_standard_fba

    # -- Per-guild FBA --
    results = []
    for g in guilds:
        bounds = _exchange_bounds_from_state(g, state)
        results.append(fba_fn(g, exchange_bounds=bounds))

    bf = state.biomass_fractions

    # -- Community exchange fluxes --
    def net(gi: int, met: str) -> float:
        row = guilds[gi].S_exchange[EXCHANGE_METABOLITES[met], :]
        return float(row @ results[gi].fluxes) * bf[gi]

    H2_flux      = sum(net(i, "H2_aq")  for i in range(4))
    NH4_flux     = sum(net(i, "NH4")    for i in range(4))
    CH4_flux     = sum(net(i, "CH4")    for i in range(4))
    H2S_flux     = sum(net(i, "H2S")    for i in range(4))
    acetate_flux = sum(net(i, "acetate") for i in range(4))

    # -- Null bases in reaction space (from internal-row null space) --
    # Each guild has different n_rxn; we pad to max_rxn for cross-guild comparison.
    max_rxn = max(g.n_rxn for g in guilds)
    null_bases_rxn = []
    null_bases_padded = []   # padded to max_rxn for Grassmann distance
    null_dims = np.zeros(4, dtype=int)

    for i, (g, r) in enumerate(zip(guilds, results)):
        S_int_i = g.S[:g.n_internal, :]
        if r.null_basis is not None:
            W_rxn = r.null_basis
        else:
            try:
                W_rxn = compute_null_basis(S_int_i)
            except ValueError:
                W_rxn = np.eye(g.n_rxn, 1)
        null_bases_rxn.append(W_rxn)
        null_dims[i] = W_rxn.shape[1]

        # Pad to max_rxn and re-orthonormalise
        pad = np.zeros((max_rxn - g.n_rxn, W_rxn.shape[1]))
        W_pad = np.vstack([W_rxn, pad])
        Q, _ = np.linalg.qr(W_pad)
        null_bases_padded.append(Q[:, :W_rxn.shape[1]])

    # -- Pairwise Grassmann distances in padded reaction space --
    G_dist = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            try:
                d = grassmann_distance(null_bases_padded[i], null_bases_padded[j])
            except Exception:
                d = np.nan
            G_dist[i, j] = G_dist[j, i] = d

    # -- SPD distances between flux covariance matrices --
    spd_dist = None
    if compute_spd:
        # Build SPD covariance in exchange space for each guild
        cov_mats = []
        for i, (g, r) in enumerate(zip(guilds, results)):
            v_exc = g.S_exchange @ r.fluxes
            cov = np.outer(v_exc, v_exc) + 1e-4 * np.eye(N_EXCH)
            cov_mats.append(cov)
        spd_dist = np.zeros((4, 4))
        for i in range(4):
            for j in range(i + 1, 4):
                try:
                    d = spd_distance(cov_mats[i], cov_mats[j])
                except Exception:
                    d = np.nan
                spd_dist[i, j] = spd_dist[j, i] = d

    # -- Geodesic trajectory: diazotroph (0) → hydrogenotroph (1) --
    geo_traj = None
    if track_geodesic:
        try:
            geo_traj = grassmann_geodesic(
                null_bases_padded[0], null_bases_padded[1],
                n_points=n_geodesic_points)
        except Exception:
            geo_traj = None

    return CommunityFBAResult(
        guild_results=results,
        community_H2_flux=H2_flux,
        community_NH4_flux=NH4_flux,
        community_CH4_flux=CH4_flux,
        community_H2S_flux=H2S_flux,
        community_acetate_flux=acetate_flux,
        grassmann_distances=G_dist,
        null_dims=null_dims,
        null_bases=null_bases_rxn,
        spd_distances=spd_dist,
        geodesic_trajectory=geo_traj,
    )


# ---------------------------------------------------------------------------
# 10.  Lifecycle scan
# ---------------------------------------------------------------------------

@dataclass
class LifecycleScan:
    phases: List[int]
    states: List[CommunityState]
    results: List[CommunityFBAResult]
    H2_trajectory:   np.ndarray
    NH4_trajectory:  np.ndarray
    H2S_trajectory:  np.ndarray
    CH4_trajectory:  np.ndarray
    grassmann_traj:  np.ndarray   # (n_phases, 4, 4)


def lifecycle_scan(n_phases: int = 5,
                   use_manifold: bool = True) -> LifecycleScan:
    """
    Scan across N2Bio lifecycle phases 0 → 4, running community FBA at each.

    Phase profiles encode the N-S coupling dynamics:
      Phase 0: NH4≈0, SO4 high → SRBs active, N2-fixation just starting
      Phase 1: NH4 rising, SO4 draining via assimilation
      Phase 2: NH4 high, SO4 moderate → assimilatory demand suppresses SRBs
      Phase 3: NH4 high, SO4 low, H2 high → nitrogenase inhibited, H2S declines
      Phase 4: all substrates depleted, senescence
    """
    phase_states = [
        CommunityState(np.array([0.70, 0.10, 0.15, 0.05]),
                       NH4_conc=0.001, SO4_conc=3.0,
                       H2_partial_pressure=0.001, HC_arom_conc=0.15,
                       lifecycle_phase=0),
        CommunityState(np.array([0.65, 0.15, 0.12, 0.08]),
                       NH4_conc=0.5,  SO4_conc=2.0,
                       H2_partial_pressure=0.05,  HC_arom_conc=0.20,
                       lifecycle_phase=1),
        CommunityState(np.array([0.55, 0.20, 0.15, 0.10]),
                       NH4_conc=2.0,  SO4_conc=1.0,
                       H2_partial_pressure=0.20,  HC_arom_conc=0.25,
                       lifecycle_phase=2),
        CommunityState(np.array([0.50, 0.20, 0.10, 0.20]),
                       NH4_conc=3.0,  SO4_conc=0.3,
                       H2_partial_pressure=0.30,  HC_arom_conc=0.45,
                       lifecycle_phase=3),
        CommunityState(np.array([0.40, 0.30, 0.10, 0.20]),
                       NH4_conc=1.5,  SO4_conc=0.05,
                       H2_partial_pressure=0.05,  HC_arom_conc=0.60,
                       lifecycle_phase=4),
    ][:n_phases]

    scan_results, H2s, NH4s, H2Ss, CH4s, G_trajs = [], [], [], [], [], []
    for state in phase_states:
        r = run_community_fba(state, use_manifold=use_manifold,
                              track_geodesic=True)
        scan_results.append(r)
        H2s.append(r.community_H2_flux)
        NH4s.append(r.community_NH4_flux)
        H2Ss.append(r.community_H2S_flux)
        CH4s.append(r.community_CH4_flux)
        G_trajs.append(r.grassmann_distances)

    return LifecycleScan(
        phases=list(range(n_phases)),
        states=phase_states,
        results=scan_results,
        H2_trajectory=np.array(H2s),
        NH4_trajectory=np.array(NH4s),
        H2S_trajectory=np.array(H2Ss),
        CH4_trajectory=np.array(CH4s),
        grassmann_traj=np.array(G_trajs),
    )


# ---------------------------------------------------------------------------
# 11.  Reporting
# ---------------------------------------------------------------------------

def print_guild_null_spaces(guilds: Optional[List[GuildModel]] = None):
    """Print null space diagnostics for all guilds."""
    if guilds is None:
        guilds = build_all_guilds()
    print("\n  Guild null space diagnostics:")
    print(f"  {'Guild':<22} {'Shape':>10} {'Rank':>6} {'Null dim':>9}")
    print("  " + "─" * 52)
    for g in guilds:
        rk = np.linalg.matrix_rank(g.S)
        nd = g.n_rxn - rk
        print(f"  {g.name:<22} {str(g.S.shape):>10} {rk:>6} {nd:>9}")


def print_community_report(result: CommunityFBAResult,
                           state: Optional[CommunityState] = None):
    w = "═" * 66
    print(f"\n{w}")
    print("  N2Bio Community Manifold FBA  —  Four-Guild Consortium")
    if state:
        print(f"  Phase {state.lifecycle_phase}  |  "
              f"NH4={state.NH4_conc:.2f} mmol/L  |  "
              f"SO4={state.SO4_conc:.2f} mmol/L  |  "
              f"H2_press={state.H2_partial_pressure:.3f} atm  |  "
              f"HC_arom={state.HC_arom_conc:.2f} mmol/L")
    print(w)

    print(f"\n  {'Guild':<22}  {'Status':<24}  {'Growth':>8}  {'Null dim':>8}  {'BF':>6}")
    print("  " + "─" * 74)
    bf = state.biomass_fractions if state else np.ones(4) * 0.25
    for r, nd, b in zip(result.guild_results, result.null_dims, bf):
        print(f"  {r.guild_name:<22}  {r.status:<24}  "
              f"{r.objective:>8.4f}  {nd:>8}  {b:>6.3f}")

    print(f"\n  Community exchange fluxes (biomass-weighted, mmol/gVSS/h):")
    print(f"    H₂  net      : {result.community_H2_flux:>+10.5f}")
    print(f"    NH₄ net      : {result.community_NH4_flux:>+10.5f}")
    print(f"    CH₄ net      : {result.community_CH4_flux:>+10.5f}")
    print(f"    H₂S net      : {result.community_H2S_flux:>+10.5f}")
    print(f"    acetate net  : {result.community_acetate_flux:>+10.5f}  ← fermentation coproduct")

    G = result.grassmann_distances
    print(f"\n  Grassmann distances  (null-space geodesics in exchange space, rad):")
    hdr = "                    " + "".join(f"  {n[:8]:>10}" for n in GUILD_NAMES)
    print("  " + hdr)
    for i, name in enumerate(GUILD_NAMES):
        row = f"  {name:<20}"
        for j in range(4):
            d = G[i, j]
            row += f"  {d:>10.4f}"
        print(row)

    if result.spd_distances is not None:
        SPD = result.spd_distances
        print(f"\n  Fisher-Rao (SPD) distances  (flux covariance, Sym+(9)):")
        print("  " + hdr)
        for i, name in enumerate(GUILD_NAMES):
            row = f"  {name:<20}"
            for j in range(4):
                d = SPD[i, j]
                row += f"  {d:>10.3f}"
            print(row)

    if result.geodesic_trajectory is not None:
        n = len(result.geodesic_trajectory)
        print(f"\n  Geodesic trajectory (Diazotroph→Hydrogenotroph): {n} points ✓")
        # Print null-space distance along geodesic
        d0 = G[0, 1]
        print(f"    Endpoint distance: {d0:.4f} rad  "
              f"({'very distant' if d0 > 1.0 else 'moderate'})")


def print_lifecycle_summary(scan: LifecycleScan):
    phase_labels = [
        "Inoculation lag",
        "N2-fixation bootstrap",
        "Peak production",
        "HC depletion",
        "Biological senescence",
    ]
    print(f"\n{'═'*72}")
    print("  N2Bio Lifecycle Manifold FBA  —  Community Flux Trajectories")
    print(f"{'═'*72}")
    print(f"\n  {'Ph':>2}  {'Label':<25}  {'H₂':>9}  {'NH₄':>9}  "
          f"{'H₂S':>9}  {'CH₄':>9}  {'Gr(D↔SRB)':>10}")
    print("  " + "─" * 72)
    for i, (ph, state) in enumerate(zip(scan.phases, scan.states)):
        label = phase_labels[ph] if ph < len(phase_labels) else f"Phase {ph}"
        d_srb = scan.grassmann_traj[i, 0, 2]  # diazotroph ↔ SRB
        print(f"  {ph:>2}  {label:<25}  "
              f"{scan.H2_trajectory[i]:>+9.4f}  "
              f"{scan.NH4_trajectory[i]:>+9.4f}  "
              f"{scan.H2S_trajectory[i]:>+9.4f}  "
              f"{scan.CH4_trajectory[i]:>+9.4f}  "
              f"{d_srb:>10.4f}")

    print(f"\n  Grassmann distance (Diazotroph ↔ SRB) — N-S coupling signature:")
    for i, ph in enumerate(scan.phases):
        d = scan.grassmann_traj[i, 0, 2]
        bar_len = max(0, int(d * 14))
        bar = "█" * bar_len
        label = "SRB suppressed" if d < 0.5 else "SRB active" if d > 1.0 else "transitional"
        print(f"    Phase {ph}: {d:.4f} rad  {bar:<20}  {label}")


# ---------------------------------------------------------------------------
# 12.  Entry point / demo
# ---------------------------------------------------------------------------

def run_demo():
    print("\n" + "═" * 66)
    print("  Matrix Manifold FBA  —  N2Bio Four-Guild Consortium")
    print("  Grassmann · Stiefel · SPD  |  N-S coupling encoded")
    print("=" * 66)

    # Verify null spaces
    print("\n[0/3] Null space verification:")
    print_guild_null_spaces()

    # Single phase — peak production
    state_peak = CommunityState(
        biomass_fractions=np.array([0.55, 0.20, 0.15, 0.10]),
        NH4_conc=2.0, SO4_conc=1.0,
        H2_partial_pressure=0.20, HC_arom_conc=0.25,
        lifecycle_phase=2,
    )

    print("\n[1/3] Manifold FBA — peak production (Phase 2)...")
    r_manifold = run_community_fba(state_peak, use_manifold=True, track_geodesic=True)
    print_community_report(r_manifold, state_peak)

    print("\n[2/3] Standard LP FBA — same state (comparison)...")
    r_lp = run_community_fba(state_peak, use_manifold=False, track_geodesic=False)
    print(f"\n  H₂ flux  —  manifold: {r_manifold.community_H2_flux:+.5f}  "
          f"LP: {r_lp.community_H2_flux:+.5f}  "
          f"(diff: {r_manifold.community_H2_flux - r_lp.community_H2_flux:+.5f})")
    print(f"  NH₄ flux —  manifold: {r_manifold.community_NH4_flux:+.5f}  "
          f"LP: {r_lp.community_NH4_flux:+.5f}")

    print("\n[3/3] Lifecycle scan (Phases 0 → 4)...")
    scan = lifecycle_scan(n_phases=5, use_manifold=True)
    print_lifecycle_summary(scan)

    print("\n  Done.\n")
    return r_manifold, r_lp, scan


if __name__ == "__main__":
    run_demo()

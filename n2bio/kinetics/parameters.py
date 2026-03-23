"""
kinetics/parameters.py
======================
Kinetic and reservoir parameter dataclasses for the N2Bio framework.

All kinetic parameters are calibrated to Thermotoga-like diazotrophic
populations under deep subsurface conditions (Table 1 of the N2Bio framework).
"""

from dataclasses import dataclass, field


@dataclass
class KineticParameters:
    """
    Biological kinetic parameters for the N2Bio model.

    All rate constants are on an hourly basis to be consistent with the ODE
    solver (scipy.integrate.solve_ivp called with time in hours).

    Attributes
    ----------
    mu_max : float
        Maximum diazotrophic growth rate [h⁻¹].
    K_N2 : float
        Half-saturation constant for dissolved N₂ [mol/L].
    K_S : float
        Half-saturation constant for HC_ali (aliphatic HC pool) [mol/L].
    Y : float
        Biomass yield coefficient [g VSS / g COD].
    b_decay : float
        Endogenous decay rate [h⁻¹].
    K_I_H2 : float
        H₂ inhibition constant on nitrogenase [atm].
        Nitrogenase activity falls to 50% at P_H₂ = K_I_H₂.
    K_I_NH4 : float
        NH₄⁺ catabolite repression constant [mol/L].
        High [NH₄⁺] suppresses nitrogenase expression.
    r_nit_max : float
        Maximum specific nitrogenase rate [mol N₂ / g VSS / h].
    k_ali : float
        First-order aliphatic HC degradation rate constant [h⁻¹].
        Equivalent to 0.01 d⁻¹.
    k_arom : float
        First-order aromatic HC transformation rate constant [h⁻¹].
        Equivalent to 0.0005 d⁻¹.
    K_I_arom : float
        Aromatic HC inhibition constant for nitrogenase [mol/L].
    f_arom0 : float
        Initial aromatic fraction of the total HC pool [-].
    Y_HC_H2 : float
        HC-to-H₂ fermentation yield [mol H₂ / g COD].
    K_N_fixed : float
        NH₄⁺ half-saturation for biomass synthesis [mol/L].
        Controls bootstrap from N-fixation product.
    Y_N : float
        Nitrogen content of biomass [g N / g VSS].
    """

    # ---- Growth kinetics ----
    mu_max: float = 0.08          # h⁻¹
    K_N2: float = 5.0e-5          # mol/L  (= 0.05 mmol/L)
    K_S: float = 5.0e-4           # mol/L  (= 0.5 mmol/L)
    Y: float = 0.07               # g VSS / g COD
    b_decay: float = 0.005        # h⁻¹

    # ---- Nitrogenase inhibition ----
    K_I_H2: float = 0.3           # atm
    K_I_NH4: float = 0.1          # mol/L
    r_nit_max: float = 1.0        # mol N₂ / g VSS / h

    # ---- HC degradation ----
    k_ali: float = 4.167e-4       # h⁻¹  (= 0.01 d⁻¹)
    k_arom: float = 2.083e-5      # h⁻¹  (= 0.0005 d⁻¹)
    K_I_arom: float = 5.0e-4      # mol/L
    f_arom0: float = 0.15         # initial aromatic fraction [-]
    Y_HC_H2: float = 0.35         # mol H₂ / g COD

    # ---- NH₄⁺ biosynthesis ----
    K_N_fixed: float = 5.0e-6     # mol/L
    Y_N: float = 0.14             # g N / g VSS

    # ---- Sulfur assimilation: SAT / OASTL pathway ----
    # SAT (serine acetyltransferase): serine + acetyl-CoA → OAS + CoA
    r_SAT_max: float = 2.0e-3     # mol OAS / g VSS / h
    K_Ac_SAT: float = 5.0e-4      # mol/L  acetate (acetyl-CoA proxy) half-sat for SAT
    K_I_Cys_SAT: float = 1.0e-3   # mol/L  cysteine allosteric feedback on SAT
    K_OAS_SAT: float = 1.0e-4     # mol/L  OAS half-sat (minor back-constraint)

    # OASTL (O-acetylserine thiol-lyase): OAS + H₂S → Cys + acetate
    r_OASTL_max: float = 5.0e-3   # mol Cys / g VSS / h
    K_OAS_OASTL: float = 5.0e-5   # mol/L  OAS half-sat for OASTL
    K_H2S_OASTL: float = 3.0e-5   # mol/L  dissolved sulfide half-sat for OASTL
    pKa_H2S: float = 7.0          # pKa of H₂S/HS⁻ at 80 °C (slightly < 25 °C value of 7.05)

    # Fe-S cluster maintenance drain on Cys (continuous turnover in nitrogenase)
    r_FeS_maintenance: float = 5.0e-5  # mol Cys / g VSS / h

    # Cys co-limitation of nitrogenase (Fe-S cluster assembly requirement)
    K_Cys_nit: float = 2.0e-5    # mol/L  (20 μM) — half-sat for Cys on nitrogenase

    # ---- Acetate mass balance (fermentation coproduct) ----
    Y_acetate: float = 0.01       # mol acetate / g COD  (≈ 2 mol/mol hexose ÷ 192 g COD/mol)
    Y_S: float = 0.010            # g S / g VSS  — sulfur content of biomass

    # ---- H₂S transport ----
    k_strip_H2S: float = 0.05     # h⁻¹  — dissolved H₂S gas-phase stripping
    H2S_source: float = 0.0       # mol/L/h — geological/SRB H₂S source (boundary)

    # ---- Buoyancy / gravitational segregation ----
    # f_buoyancy_seg: fraction of bulk gas-cap P(H₂) that the biofilm "sees" at
    # the injection front.  Arises because biogenic H₂ (M = 2 g/mol) is ~14×
    # less dense than N₂ (M = 28 g/mol) and migrates buoyantly to the structural
    # high of the anticline, while the microbial biofilm colonises the deeper
    # pore space near the injector.  The local gas phase near the biofilm is
    # therefore dominated by the injected N₂ cushion, with a P(H₂) substantially
    # below the reservoir-average gas-cap value.
    #
    # f_buoyancy_seg = 1.0  →  well-mixed assumption (legacy, conservative)
    # f_buoyancy_seg = 0.1  →  strong segregation; biofilm sees ~10% of bulk P(H₂)
    #
    # Calibration guidance:
    #   Gravity number Γ for H₂/N₂ systems is ~7% higher than CH₄/N₂ UGS,
    #   and ~3× higher than CO₂ storage (Luboń & Tarkowski 2021).  Field-scale
    #   reservoir simulations of UHS with N₂ cushion gas show H₂ saturation
    #   near-zero at the injector and near-maximum at the structural crest
    #   (Hagemann et al. 2015, Muhammed et al. 2022).  A value of 0.05–0.20
    #   is physically reasonable for a well-designed anticline with dedicated
    #   injector/producer wells.
    f_buoyancy_seg: float = 1.0   # [-]  default: well-mixed (no correction)

    def __post_init__(self):
        """Validate parameter ranges."""
        if self.mu_max <= 0:
            raise ValueError(f"mu_max must be positive, got {self.mu_max}")
        if not 0.0 < self.Y <= 1.0:
            raise ValueError(f"Biomass yield Y must be in (0, 1], got {self.Y}")
        if self.b_decay < 0:
            raise ValueError(f"b_decay must be non-negative, got {self.b_decay}")
        if not 0.0 <= self.f_arom0 <= 1.0:
            raise ValueError(f"f_arom0 must be in [0,1], got {self.f_arom0}")
        if self.K_I_H2 <= 0:
            raise ValueError(f"K_I_H2 must be positive, got {self.K_I_H2}")
        if not 0.0 <= self.f_buoyancy_seg <= 1.0:
            raise ValueError(f"f_buoyancy_seg must be in [0,1], got {self.f_buoyancy_seg}")

    def summary(self) -> str:
        """Return a human-readable summary of kinetic parameters."""
        lines = [
            "KineticParameters Summary",
            "=" * 40,
            f"  μ_max            = {self.mu_max:.4f} h⁻¹",
            f"  K_N2             = {self.K_N2*1e6:.1f} μmol/L",
            f"  K_S              = {self.K_S*1e3:.3f} mmol/L",
            f"  Y                = {self.Y:.3f} g VSS/g COD",
            f"  b_decay          = {self.b_decay:.4f} h⁻¹",
            f"  K_I_H2           = {self.K_I_H2:.3f} atm",
            f"  K_I_NH4          = {self.K_I_NH4*1e3:.1f} mmol/L",
            f"  r_nit_max        = {self.r_nit_max:.2f} mol N₂/g VSS/h",
            f"  K_Cys_nit        = {self.K_Cys_nit*1e6:.1f} μmol/L",
            f"  k_ali            = {self.k_ali:.3e} h⁻¹",
            f"  k_arom           = {self.k_arom:.3e} h⁻¹",
            f"  K_I_arom         = {self.K_I_arom*1e3:.3f} mmol/L",
            f"  f_arom0          = {self.f_arom0:.2f}",
            f"  Y_HC_H2          = {self.Y_HC_H2:.3f} mol H₂/g COD",
            f"  Y_acetate        = {self.Y_acetate:.4f} mol acetate/g COD",
            f"  K_N_fixed        = {self.K_N_fixed*1e6:.2f} μmol/L",
            f"  Y_N              = {self.Y_N:.3f} g N/g VSS",
            f"  Y_S              = {self.Y_S:.3f} g S/g VSS",
            f"  r_SAT_max        = {self.r_SAT_max:.2e} mol OAS/g VSS/h",
            f"  r_OASTL_max      = {self.r_OASTL_max:.2e} mol Cys/g VSS/h",
            f"  r_FeS_maint      = {self.r_FeS_maintenance:.2e} mol Cys/g VSS/h",
            f"  k_strip_H2S      = {self.k_strip_H2S:.3f} h⁻¹",
            f"  H2S_source       = {self.H2S_source:.2e} mol/L/h",
            f"  f_buoyancy_seg   = {self.f_buoyancy_seg:.3f}  (buoyancy/gravitational segregation factor)",
        ]
        return "\n".join(lines)


@dataclass
class ReservoirConditions:
    """
    Physical and chemical conditions of the subsurface reservoir.

    Attributes
    ----------
    temperature : float
        Reservoir temperature [K].  Default 358.15 K (85°C).
    pressure : float
        Total reservoir pressure [bar].  Default 150 bar.
    pH : float
        Aqueous phase pH [-].
    salinity_NaCl : float
        NaCl salinity [mol/kg water] (molality).
    porosity : float
        Formation porosity φ [-].
    permeability : float
        Absolute permeability [m²].  Default 50 mD = 50e-15 m².
    """

    temperature: float = 358.15      # K  (85°C)
    pressure: float = 150.0          # bar
    pH: float = 7.0
    salinity_NaCl: float = 0.5       # mol/kg
    porosity: float = 0.15
    permeability: float = 50.0e-15   # m²  (50 mD)

    @property
    def temperature_C(self) -> float:
        """Temperature in Celsius."""
        return self.temperature - 273.15

    @property
    def pressure_Pa(self) -> float:
        """Pressure in Pa."""
        return self.pressure * 1.0e5

    def __post_init__(self):
        """Validate reservoir conditions."""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive [K], got {self.temperature}")
        if self.pressure <= 0:
            raise ValueError(f"Pressure must be positive [bar], got {self.pressure}")
        if not 0.0 < self.porosity < 1.0:
            raise ValueError(f"Porosity must be in (0,1), got {self.porosity}")
        if self.permeability <= 0:
            raise ValueError(f"Permeability must be positive [m²], got {self.permeability}")
        if not 0.0 <= self.pH <= 14.0:
            raise ValueError(f"pH must be in [0,14], got {self.pH}")

    def summary(self) -> str:
        """Human-readable summary of reservoir conditions."""
        lines = [
            "ReservoirConditions Summary",
            "=" * 40,
            f"  Temperature : {self.temperature_C:.1f} °C  ({self.temperature:.2f} K)",
            f"  Pressure    : {self.pressure:.1f} bar",
            f"  pH          : {self.pH:.2f}",
            f"  Salinity    : {self.salinity_NaCl:.3f} mol/kg NaCl",
            f"  Porosity    : {self.porosity:.3f}",
            f"  Permeability: {self.permeability*1e15:.1f} mD",
        ]
        return "\n".join(lines)

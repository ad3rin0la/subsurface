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

    def summary(self) -> str:
        """Return a human-readable summary of kinetic parameters."""
        lines = [
            "KineticParameters Summary",
            "=" * 40,
            f"  μ_max       = {self.mu_max:.4f} h⁻¹",
            f"  K_N2        = {self.K_N2*1e6:.1f} μmol/L",
            f"  K_S         = {self.K_S*1e3:.3f} mmol/L",
            f"  Y           = {self.Y:.3f} g VSS/g COD",
            f"  b_decay     = {self.b_decay:.4f} h⁻¹",
            f"  K_I_H2      = {self.K_I_H2:.3f} atm",
            f"  K_I_NH4     = {self.K_I_NH4*1e3:.1f} mmol/L",
            f"  r_nit_max   = {self.r_nit_max:.2f} mol N₂/g VSS/h",
            f"  k_ali       = {self.k_ali:.3e} h⁻¹",
            f"  k_arom      = {self.k_arom:.3e} h⁻¹",
            f"  K_I_arom    = {self.K_I_arom*1e3:.3f} mmol/L",
            f"  f_arom0     = {self.f_arom0:.2f}",
            f"  Y_HC_H2     = {self.Y_HC_H2:.3f} mol H₂/g COD",
            f"  K_N_fixed   = {self.K_N_fixed*1e6:.2f} μmol/L",
            f"  Y_N         = {self.Y_N:.3f} g N/g VSS",
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

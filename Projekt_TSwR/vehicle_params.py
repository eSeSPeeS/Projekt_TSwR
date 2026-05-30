"""
vehicle_params.py
=================
Parametry fizyczne pojazdu F1/10 (skala 1:10).
Wzorowane na arXiv:2003.04882 (Kabzan et al.)
"""

from dataclasses import dataclass


@dataclass
class VehicleParams:
    """Parametry fizyczne pojazdu F1/10."""

    # Geometria
    lf: float = 0.15875   # odległość CoG → oś przednia [m]
    lr: float = 0.17145   # odległość CoG → oś tylna [m]

    # Masa i bezwładność
    m:  float = 3.47      # masa [kg]
    Iz: float = 0.04712   # moment bezwładności wokół osi Z [kg·m²]

    # Opony – model Pacejki: Fy = D·sin(C·arctan(B·alpha))
    Bf: float = 9.242     # sztywność – przód
    Cf: float = 1.2       # kształt – przód
    Df: float = 134.585   # siła szczytowa – przód [N]

    Br: float = 17.716    # sztywność – tył
    Cr: float = 1.2       # kształt – tył
    Dr: float = 159.919   # siła szczytowa – tył [N]

    # Napęd: Fx = Cm1·d - Cr0 - Cr2·vx²
    Cm1: float = 20.0     # współczynnik siły napędowej [N]
    Cm2: float = 0.0      # (nieużywany, zachowany dla zgodności)
    Cr0: float = 0.05     # opór toczenia [N]
    Cr2: float = 0.0      # opór aerodynamiczny [N·s²/m²]

    # Ograniczenia sterowania
    delta_max: float = 0.35   # max kąt skrętu [rad] (~20°)
    v_min:     float = 0.0    # min prędkość [m/s]
    v_max:     float = 16.0    # max prędkość [m/s]

    # Grawitacja
    g: float = 9.81

    @property
    def L(self) -> float:
        """Rozstaw osi [m]."""
        return self.lf + self.lr
from __future__ import annotations
import numpy as np
from physics.atomic_data import ATOMIC_DATA
from physics.attenuation import MEC2_KEV, weight_fractions

B = {1:1.4914e-1,2:9.05e-1,3:1.42e-3,4:9.27e-1,5:9.60264e-1,6:3.428e-4,7:9.5696e-1,8:4.58e-4,9:9.52525e-1,10:2.993e-4,11:1.308e0,12:5.48e-3,13:2.1e-5,14:1.0746e0,15:1.072e-2,16:1.406e4,17:4.259e2,18:4.824e0,19:1.003e8,20:5.94e0,21:4.2e-2,22:4.0e-5,23:4.6e1,24:1.324e1}


def compound_mean_Z_A_tabata(formula):
    wf = weight_fractions(formula)
    Zm = sum(w*(ATOMIC_DATA[el]['Z']**2)/ATOMIC_DATA[el]['A'] for el,w in wf.items()) / sum(w*ATOMIC_DATA[el]['Z']/ATOMIC_DATA[el]['A'] for el,w in wf.items())
    ZA_m = sum(w*(ATOMIC_DATA[el]['Z']**B[2])/ATOMIC_DATA[el]['A'] for el,w in wf.items())
    Am = (Zm**B[2]) / ZA_m
    return Zm, Am

class TabataRangeModel:
    def __init__(self, formula, density_g_cm3: float):
        self.formula = formula
        self.density_g_cm3 = density_g_cm3
        self.Zm, self.Am = compound_mean_Z_A_tabata(formula)
        self.energy_grid_keV = np.geomspace(1.0, 511.0, 1200)
        self.range_grid_g_cm2 = np.array([self._range_scalar(E) for E in self.energy_grid_keV])
        self.range_grid_mm = 10.0 * self.range_grid_g_cm2 / density_g_cm3

    def _range_scalar(self, E_keV):
        tau = max(E_keV/MEC2_KEV, 1e-12)
        Zm, Am = self.Zm, self.Am
        a1 = B[1] * Am / (Zm**B[2])
        a2 = B[3] * (Zm**B[4])
        a3 = B[5] + B[6]*Zm
        a4 = B[7] - B[8]*Zm
        a5 = B[9] + B[10]*Zm
        a6 = B[11] + B[12]*Zm + B[13]*(Zm**2)
        a7 = B[14] / (Zm**B[15])
        a8 = B[16] + B[17]*Zm + B[18]*(Zm**2)
        a9 = B[19] * (1 + B[20]*np.exp(-B[21]*Zm))
        a10 = B[22] * (Zm**(B[23] - B[24]*np.log(Zm)))
        r = a1 * (np.log(1 + a2*(tau**a3))/a2 - a4*(tau**a5)/(1 + a6*(tau**a7)) + a8*(tau**3)/(1 + a9*(tau**3)) + a10*(tau**5))
        return max(float(r), 0.0)

    def range_mm(self, E_keV):
        return float(np.interp(E_keV, self.energy_grid_keV, self.range_grid_mm))

    def residual_energy_from_path(self, E0_keV, path_mm):
        path_g_cm2 = self.density_g_cm3 * path_mm / 10.0
        R0 = float(np.interp(E0_keV, self.energy_grid_keV, self.range_grid_g_cm2))
        target = R0 - path_g_cm2
        if target <= 0.0:
            return 0.0
        return float(np.interp(target, self.range_grid_g_cm2, self.energy_grid_keV))

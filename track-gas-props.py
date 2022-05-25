import yt
import numpy as np

ds = yt.load("/data/draco/flashgrp/slewis/Paper1-runs-snapshots/L3-v/2tff_hdf5_plt_2070")
ad = ds.all_data()

cell_mass = ad['cell_mass'].v
cell_vx = ad['velx'].v
cell_vy = ad['vely'].v
cell_vz = ad['velz'].v

gas_KE = 0.5*cell_mass*(cell_vx**2 + cell_vy**2 + cell_vz**2)
gas_EI = cell_mass*ad['eint'].v
gas_ME = ad['magp'].v*ad['cell_volume'].v / 4.0 / np.pi

g_on_g_PE = 0.5*ad['gpot'].v*cell_mass
s_on_g_PE = ad['bgpt'].v*cell_mass

gas_TE = gas_KE+gas_EI+gas_ME+g_on_g_PE+s_on_g_PE

unbound_ind = np.where(gas_TE > 0.)
bound_ind = np.where(gas_TE < 0.)
bound_dens_ind = np.where((gas_TE < 0.) & (ad['dens'] > 1e2))

gas_TM = np.sum(cell_mass)
frac_unbound = np.sum(cell_mass[unbound_ind])/gas_TM
frac_bound = np.sum(cell_mass[bound_ind])/gas_TM
frac_bound_dens = np.sum(cell_mass[bound_dens_ind])/gas_TM

print("Frac unbound {}, mass unbound {}".format(frac_unbound, np.sum(cell_mass[unbound_ind])))
print("Frac bound {}, mass bound {}".format(frac_bound, np.sum(cell_mass[bound_ind])))
print("Frac bound and dense {}, mass bound and dense {}".format(frac_bound_dens, np.sum(cell_mass[bound_dens_ind])))

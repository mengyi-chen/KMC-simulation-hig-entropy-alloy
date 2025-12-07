from ElectrodeSimulator.utils import mkdir
from pymatgen.core import Structure, Species, PeriodicSite
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
import random
import os
import json
import time
from monty.json import jsanitize

from chgnet.model.model import CHGNet
from chgnet.model import StructOptimizer

# TODO: change to MACE (torch-sim) 
# mace_omat 
# no need to do relaxation? 

model = CHGNet()
optimizer = StructOptimizer(model = model)

##### start process ######
file_list = os.listdir('./MC_sampling_anneal/')

save_list = []

for file in file_list:

    if not ('cif' in file):
        continue

    s0 = Structure.from_file('./MC_sampling_anneal/' + file)
    # s0.add_oxidation_state_by_element({'Li': 1, 'Zr':4, 'Cl': -1})

    result = optimizer.relax(s0, fmax = 0.075, steps= 50000)
    s_relax = result["final_structure"]
    toten = result['trajectory'].energies[-1]

    s_scale = s_relax.copy()

    print("total charge: ", s_scale.charge)

    if (s_scale.charge != 0):
        continue

    ## process the volume
    vol_ratio = s_relax.volume / s0.volume
    alpha = vol_ratio ** (1/3)
    s_scale.apply_strain(1 - alpha)

    save_dict = {'s0': s0.as_dict(),
                 's_scale': s_scale.as_dict(),
                 's_relax': s_relax.as_dict(),
                 'toten': float(toten)}

    save_list.append(save_dict)


### save the relaxation ###
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)

with open('./dataset_CE/structure_energy/relax_'+ str(current_time) +'.json', 'w') as fp:
    json.dump(jsanitize(save_list), fp)


#### in kMC: how to read the CE energy

# wrangler = StructureWrangler.from_dict(wrangler_json)
# subspace = wrangler.cluster_subspace

# corr_vector = subspace.corr_from_structure(s_scale) #. get_correlation_vector(s_scale)
# energy_CE = np.dot(corr_vector, ecis_l1)


# spinel_super = Structure.from_file('../orderings/LiCoO2_super_refine.cif')
# supercell_matrix = subspace.scmatrix_from_structure(structure= spinel_super)

# corr = subspace.corr_from_structure(self, structure, normalized=True, scmatrix=supercell_matrix)



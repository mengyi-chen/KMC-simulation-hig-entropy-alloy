import numpy as np
import json
from monty.serialization import loadfn, dumpfn
from pymatgen.core import Structure, Site, PeriodicSite
from pymatgen.entries.computed_entries import ComputedStructureEntry
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.cofe.space import Cluster
from smol.cofe.extern import EwaldTerm
from smol.cofe.wrangling.tools import unique_corr_vector_indices

from itertools import combinations
from copy import deepcopy
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")


class processDFTtoCE:

    def __init__(self, DFT_path):
        """
        DFT_path is a path file points where you store the json of DFT and energies
        """

        self.DFT_path = DFT_path


    def read_file(self, if_boundary = False):

        self.calc_data = []
        file_list = os.listdir(self.DFT_path)
        file_list.sort()
        for file in file_list:
            print(file)
            if not ('.json' in file):
                continue
            with open(self.DFT_path + file, 'r') as fin: read_data = json.loads(fin.read())

            self.calc_data.extend(read_data)

        # self.toCE_structs = []
        #
        # for calc_i, calc in enumerate(self.calc_data):
        #     struct = Structure.from_dict(calc['s_scale'])
        #     self.toCE_structs.append((struct, calc['toten']))
        print("Read {} structures from DFT/MLP".format(len(self.calc_data)))


    def calculate_corr_by_wrangler(self, subspace, remove_charge = False, add_charge_dict = None):
        self.wrangler = StructureWrangler(subspace)

        # you can add any number of properties and name them
        # whatever you want. You should use something descriptive.
        # In this case we'll call it 'total_energy'.
        for ii, calc in enumerate(self.calc_data):
            print("processed: {}/{}".format(ii, len(self.calc_data)))
            structure =  Structure.from_dict(calc['s_scale'])
            toten = calc['toten']
            if add_charge_dict != None:
                structure.add_oxidation_state_by_element(add_charge_dict)
            if remove_charge:
                structure.remove_oxidation_states()
            entry = ComputedStructureEntry(structure= structure, energy= toten )
            self.wrangler.add_entry(entry = entry,
                                    verbose=True)
        # The verbose flag will print structures that fail to match.

        print(f'\nTotal structures that match {self.wrangler.num_structures}/{len(self.calc_data)}')

    def calculate_corr_by_refinement(self, subspace, remove_charge = False):
        self.wrangler = StructureWrangler(subspace)

        # you can add any number of properties and name them
        # whatever you want. You should use something descriptive.
        # In this case we'll call it 'total_energy'.
        for ii, calc in enumerate(self.calc_data):
            print("processed: {}/{}".format(ii, len(self.calc_data)))
            structure =  Structure.from_dict(calc['s_scale'])
            s0 = Structure.from_dict(calc['s0'])
            toten = calc['toten']

            try:
                sc_matrix = subspace.scmatrix_from_structure(s0)
                s_refine = subspace.refine_structure(structure, scmatrix = sc_matrix)
                print("Found supercell")
            except:
                s_refine = structure


            if remove_charge:
                structure.remove_oxidation_states()
            entry = ComputedStructureEntry(structure= s_refine, energy= toten )
            self.wrangler.add_entry(entry = entry,
                                    verbose=True)
        # The verbose flag will print structures that fail to match.

        print(f'\nTotal structures that match {self.wrangler.num_structures}/{len(self.calc_data)}')

def main():

    unit = Structure.from_file('./POSCAR.POSCAR.vasp')
    prim = unit.get_primitive_structure()
    prim.replace_species({'Fe': {'Cu': 1/7, 'Ni': 1/7, 'Fe': 1/7, 'Al': 1/7, 'Cr': 1/7, 'Co': 1/7 , 'Li': 1/7}, 
                          'O': {'O': 1/2, 'F': 1/2} }
                          )
# TODO: # < 10 mev  / atom 

    print(prim)

    subspace = ClusterSubspace.from_cutoffs(prim, 
                                            ltol = 0.2, stol = 0.3, angle_tol = 5,
                                            cutoffs={2: 6, 3: 4}, # will include orbits of 2 and 3 sites. 5 and 5.3 makes no difference
                                            basis='indicator', # can be 'indicator' or 'orthonormal'
                                            supercell_size='volume', 
                                            )

    # subspace.add_external_term(EwaldTerm(eta=None))

    print("Number of corr function: ", subspace.num_corr_functions)

    mkdir('./dataset_CE/structure_energy/')
    processor = processDFTtoCE('./dataset_CE/structure_energy/')
    processor.read_file()
    processor.calculate_corr_by_wrangler(subspace=subspace)

    wrangler_toSave = processor.wrangler
    wrangler_dict = wrangler_toSave.as_dict()
    with open('./dataset_CE/wrangler.json', 'w') as fp:
        json.dump(wrangler_dict, fp)


if __name__ == '__main__':
    main()

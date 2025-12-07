"""
Convert NPZ structure file to VASP format.

Usage:
    python npz_to_vasp.py
"""

import numpy as np
from pymatgen.core import Structure, Lattice

# Input and output paths
npz_path = './local_structures/template_1_step0_vac13457_X.npz'
vasp_path = './local_structures/template_1_step0_vac13457_X.vasp'

# Load NPZ file
data = np.load(npz_path, allow_pickle=True)

positions = data['positions']  # Cartesian coordinates
cell = data['cell']            # 3x3 lattice matrix
symbols = data['symbols']      # Element symbols

print(f"Loaded: {npz_path}")
print(f"  Atoms: {len(positions)}")
print(f"  Cell: {cell[0,0]:.4f} x {cell[1,1]:.4f} x {cell[2,2]:.4f} Å")
print(f"  Species: {set(symbols)}")

# Create pymatgen Structure
structure = Structure(
    lattice=Lattice(cell),
    species=symbols.tolist(),
    coords=positions,
    coords_are_cartesian=True
)

# Save to VASP format
structure.to(filename=vasp_path, fmt='poscar')
print(f"\nSaved: {vasp_path}")

"""
Energy model abstraction for flexible ML potential integration

Supports multiple ML models:
- CHGNet
- MACE foundation models (via torch-sim for GPU batched computation)
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from ase import Atoms
import torch

# Available MACE foundation models
MACE_FOUNDATION_MODELS = [
    'small', 'medium', 'large',  # MP models
    'small-0b', 'medium-0b', 'small-0b2', 'medium-0b2', 'medium-0b3', 'large-0b2',  # MP versions
    'medium-omat-0', 'small-omat-0',  # OMAT models
    'medium-mpa-0',  # MPA model
]

class EnergyModel(ABC):
    """Abstract base class for energy models

    This allows flexible switching between different ML potentials
    while maintaining a consistent interface.
    """

    @abstractmethod
    def predict_energy(self,
                       structures: Union[Atoms, List[Atoms]],
                       batch_size: int = 64) -> np.ndarray:
        """Predict energies for one or more structures

        Args:
            structures: Single structure or list of structures (ASE Atoms format)
            batch_size: Batch size for prediction

        Returns:
            energies: Array of energies in eV
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model"""
        pass


class CHGNetModel(EnergyModel):
    """CHGNet energy model wrapper"""

    def __init__(self, device='cpu'):
        """Initialize CHGNet model

        Args:
            device: Device for computation ('cpu' or 'cuda:0', etc.)
        """
        from chgnet.model import CHGNet
        from pymatgen.io.ase import AseAtomsAdaptor

        self.model = CHGNet.load(use_device=device)
        
        # self.model.graph_converter.atom_graph_cutoff = 10.0

        # NOTE: don't use self.model.is_intensive = False, this will give positive energy, which is weird
        # NOTE: CHGNet will give energy per atom, we multiple it by the number of atoms 
        # self.model.is_intensive = False 
        self.device = device
        self.adaptor = AseAtomsAdaptor()

    def predict_energy(self,
                       structures: Union[Atoms, List[Atoms]],
                       batch_size: int = 64) -> np.ndarray:
        """Predict energies using CHGNet

        Args:
            structures: Single structure or list of structures (ASE Atoms)
            batch_size: Batch size for prediction

        Returns:
            energies: Array of energies in eV
        """

        # Convert ASE Atoms to pymatgen Structure for CHGNet
        if not isinstance(structures, list):
            structures = [structures]

        pmg_structures = [self.adaptor.get_structure(s) for s in structures]

        # Use no_grad since we only need energy, not forces/gradients
        with torch.no_grad():
            results = self.model.predict_structure(
                pmg_structures,
                task='e',
                batch_size=batch_size
            )

        # Extract energies (CHGNet returns energy per atom, multiply by num_atoms for total energy)
        if isinstance(results, list):
            return np.array([r['e'] * len(pmg_structures[i]) for i, r in enumerate(results)])
        else:
            return np.array([results['e'] * len(pmg_structures[0])])

    def get_model_name(self) -> str:
        return "CHGNet"

class MACEModel(EnergyModel):
    """MACE foundation model wrapper using mace_mp ASE calculator

    This class uses the MACE foundation models directly via the ASE calculator interface.
    """
    def __init__(self, device='cuda:0', model_name: str = 'medium-omat-0'):
        """Initialize MACE foundation model with torch-sim

        Args:
            device: Device for computation ('cpu', 'cuda:0', etc.)
            model_name: Name of MACE foundation model. Options:
                - MP models: 'small', 'medium', 'large'
                - MP versions: 'small-0b', 'medium-0b', 'small-0b2', 'medium-0b2',
                              'medium-0b3', 'large-0b2'
                - OMAT models: 'medium-omat-0', 'small-omat-0'
                - MPA model: 'medium-mpa-0'
        """
        import torch_sim as ts
        from torch_sim.models.mace import MaceModel
        from mace.calculators import mace_mp
        from pymatgen.io.ase import AseAtomsAdaptor

        self.model_name = model_name
        self.adaptor = AseAtomsAdaptor()

        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = device
        self.device = device_str

        # Load MACE foundation model
        print(f"Loading MACE foundation model '{model_name}' with torch-sim on {device_str}...")
        mace_raw = mace_mp(model=model_name, return_raw_model=True)

        self.model = MaceModel(
            model=mace_raw,
            device=torch.device(device_str),
            dtype=torch.float32,
            compute_forces=False,
            compute_stress=False
        )

        # Store torch_sim module reference for batched computation
        self._ts = ts

    def predict_energy(self,
                       structures: Union[Atoms, List[Atoms]],
                       batch_size: int = 64) -> np.ndarray:
        """Predict energies using MACE foundation model with torch-sim batched computation

        Args:
            structures: Single structure or list of structures (ASE Atoms)
            batch_size: Batch size for GPU computation (used by autobatcher)

        Returns:
            energies: Array of energies in eV
        """
        if not isinstance(structures, list):
            structures = [structures]

        # Define prop_calculators to extract potential_energy from state
        prop_calculators = {
            1: {"potential_energy": lambda state: state.energy},
        }

        with torch.no_grad():
            # Use ts.static with autobatching for efficient GPU memory management
            results = self._ts.static(
                system=structures,
                model=self.model,
                trajectory_reporter={
                    "filenames": None,  # Don't save trajectories, just get properties
                    "prop_calculators": prop_calculators,
                },
                autobatcher=True,  # Let torch-sim handle batching automatically
            )

        # Extract energies from results
        energies = []
        for r in results:
            e = r["potential_energy"][-1]  # Get the energy value
            if isinstance(e, torch.Tensor):
                energies.append(float(e.cpu().item()))
            else:
                energies.append(float(e))

        return np.array(energies)

    def get_model_name(self) -> str:
        return f"MACE-{self.model_name}"


# Factory function for easy model creation
def create_energy_model(model_type: str, **kwargs) -> EnergyModel:
    """Factory function to create energy models

    Args:
        model_type: Type of model ('chgnet', 'mace')
        **kwargs: Model-specific arguments
            - For 'chgnet': device
            - For 'mace': device, model_name (foundation model name)

    Returns:
        EnergyModel instance

    Examples:
        >>> model = create_energy_model('chgnet', device='cuda:0')
        >>> model = create_energy_model('mace', model_name='medium-omat-0')
    """
    model_type = model_type.lower()

    if model_type == 'chgnet':
        return CHGNetModel(**kwargs)
    elif model_type == 'mace':
        return MACEModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'chgnet', 'mace'"
        )

"""
Step 1: Compute energies for local environments using MACE with torch-sim

This script uses torch-sim for GPU-batched energy computation:
1. Loads train/test NPZ files created in Step 0
2. Uses torch-sim's batched computation with MACE model
3. Computes energies efficiently on GPU in batches
4. Outputs separate JSON files for train and test energies

Usage:
    python 1_compute_mace_energies.py --batch_size 32 --model medium-omat-0 --gpu_idx 4
"""

import os
import sys
import argparse

# Parse arguments FIRST, before any CUDA/torch imports
parser = argparse.ArgumentParser(description='Compute MACE energies with torch-sim')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for GPU computation (default: 16)')
parser.add_argument('--gpu_idx', type=int, default=5,
                    help='GPU index to use (default: 5)')
parser.add_argument('--model', type=str, default='medium-omat-0',
                    help='MACE model name')
args = parser.parse_args()

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)

# Now import torch and other dependencies
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch
import gc
from ase import Atoms
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


# Available MACE foundation models
AVAILABLE_MODELS = [
    'small', 'medium', 'large',  # MP models
    'small-0b', 'medium-0b', 'small-0b2', 'medium-0b2', 'medium-0b3', 'large-0b2',  # MP versions
    'medium-omat-0', 'small-omat-0',  # OMAT models
    'medium-mpa-0',  # MPA model
]


def load_structures_from_npz(npz_path, dataset_name="dataset"):
    """Load structures from NPZ file created in Step 0.

    This file contains:
    - atomic_numbers: array of shape (n_structures, n_atoms) with atomic numbers
    - positions: template positions array (shared across all structures)
    - cell: template cell array (shared across all structures)

    Args:
        npz_path: Path to NPZ file
        dataset_name: Name for progress bar (e.g., "train", "test")

    Returns:
        atoms_list: List of ASE Atoms objects
        metadata_list: List of metadata dicts with structure_idx and n_atoms
    """
    npz_path = Path(npz_path)

    print(f"Loading {dataset_name} structures from: {npz_path}")
    data = np.load(npz_path)

    atomic_numbers_bulk = data['atomic_numbers']  # Shape: (n_structures, n_atoms)
    template_positions = data['positions']         # Shape: (n_atoms, 3)
    template_cell = data['cell']                   # Shape: (3, 3)

    print(f"  Atomic numbers shape: {atomic_numbers_bulk.shape}")
    print(f"  Template positions shape: {template_positions.shape}")
    print(f"  Template cell shape: {template_cell.shape}")

    atoms_list = []
    metadata_list = []

    for idx in tqdm(range(len(atomic_numbers_bulk)), desc=f"Creating {dataset_name} ASE Atoms"):
        atomic_numbers = atomic_numbers_bulk[idx]

        # Skip empty structures (all zeros)
        if np.all(atomic_numbers == 0):
            continue

        # Filter out Li (3) and F (9) - ghost atoms used for vacancy markers
        mask = (atomic_numbers != 3) & (atomic_numbers != 9)
        filtered_atomic_numbers = atomic_numbers[mask]
        filtered_positions = template_positions[mask]

        # Skip if no atoms remain after filtering
        if len(filtered_atomic_numbers) == 0:
            continue

        # Convert atomic numbers to Python ints to avoid numpy int32 issues
        atoms = Atoms(
            numbers=[int(z) for z in filtered_atomic_numbers],
            positions=filtered_positions.copy(),
            cell=template_cell.copy(),
            pbc=[True, True, False]
        )
        atoms_list.append(atoms)

        metadata_list.append({
            'structure_idx': idx,  # Index within this dataset (train or test)
            'n_atoms': int(len(filtered_atomic_numbers)),
        })

    return atoms_list, metadata_list


def compute_energies_batched(atoms_list, batch_size=32, device='cuda', model_name='medium-omat-0'):
    """Compute energies using torch-sim batched computation."""
    import torch_sim as ts
    from torch_sim.models.mace import MaceModel
    from mace.calculators import mace_mp
    
    print(f"Loading MACE model '{model_name}' on {device}...")
    
    # Load MACE model (works for all foundation models including OMAT)
    mace_raw = mace_mp(model=model_name, return_raw_model=True)
    model = MaceModel(
        model=mace_raw,
        device=torch.device(device),
        dtype=torch.float32,
        compute_forces=False,
        compute_stress=False
    )
    
    print(f"Model r_max: {mace_raw.r_max}")
    print(f"Computing energies for {len(atoms_list)} structures in batches of {batch_size}...")
    
    energies = []
    n_batches = (len(atoms_list) + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Computing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(atoms_list))
        batch = atoms_list[start_idx:end_idx]
        
        with torch.no_grad():
            
            results = ts.static(system=batch, model=model)
            
            for r in results:
                # torch-sim returns 'potential_energy' not 'energy'
                e = r.get('potential_energy', r.get('energy', None))
                if e is not None:
                    if isinstance(e, torch.Tensor):
                        energies.append(float(e.cpu().numpy().item()))
                    else:
                        energies.append(float(e))
                else:
                    energies.append(None)
                                
    return energies


def process_dataset(npz_file, output_file, dataset_name, batch_size, device, model_name):
    """Process a single dataset (train or test) and save energies.

    Args:
        npz_file: Path to input NPZ file
        output_file: Path to output JSON file
        dataset_name: Name of dataset ("train" or "test")
        batch_size: Batch size for GPU computation
        device: Device to use
        model_name: MACE model name

    Returns:
        dict with statistics
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")

    atoms_list, metadata_list = load_structures_from_npz(npz_file, dataset_name)
    print(f"Loaded {len(atoms_list)} {dataset_name} structures")

    if len(atoms_list) == 0:
        print(f"No structures to process for {dataset_name}")
        return None

    energies = compute_energies_batched(
        atoms_list,
        batch_size=batch_size,
        device=device,
        model_name=model_name
    )

    results = []
    valid_count = 0
    for meta, energy in zip(metadata_list, energies):
        meta['energy'] = energy
        meta['model'] = model_name
        results.append(meta)
        if energy is not None:
            valid_count += 1

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    valid_energies = [r['energy'] for r in results if r['energy'] is not None]

    stats = {
        'dataset': dataset_name,
        'total': len(results),
        'valid': valid_count,
        'output_file': str(output_file)
    }

    if valid_energies:
        stats['energy_min'] = float(np.min(valid_energies))
        stats['energy_max'] = float(np.max(valid_energies))
        stats['energy_mean'] = float(np.mean(valid_energies))
        stats['energy_std'] = float(np.std(valid_energies))

    print(f"\n{dataset_name.upper()} results:")
    print(f"  Total structures: {len(results)}")
    print(f"  Successfully computed: {valid_count}")
    if valid_energies:
        print(f"  Energy range: [{np.min(valid_energies):.4f}, {np.max(valid_energies):.4f}] eV")
        print(f"  Energy mean: {np.mean(valid_energies):.4f} ± {np.std(valid_energies):.4f} eV")
    print(f"  Saved to: {output_file}")

    return stats


def main():
    local_struct_dir = Path('./local_structures')
    output_dir = Path('./mace_energies')
    output_dir.mkdir(parents=True, exist_ok=True)

    # After setting CUDA_VISIBLE_DEVICES, use cuda:0 since only one GPU is visible
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Computing MACE Energies with torch-sim")
    print("="*60)
    print(f"Using GPU: {args.gpu_idx} (via CUDA_VISIBLE_DEVICES)")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")

    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

    model_suffix = args.model.replace('-', '_').replace('.', '_')
    all_stats = []

    # Process train dataset
    train_file = local_struct_dir / 'train_atomic_numbers.npz'
    if train_file.exists():
        train_output = output_dir / f'train_energies_{model_suffix}.json'
        train_stats = process_dataset(
            train_file, train_output, "train",
            args.batch_size, device, args.model
        )
        if train_stats:
            all_stats.append(train_stats)
    else:
        print(f"\nWarning: Train file not found: {train_file}")

    # Process test dataset
    test_file = local_struct_dir / 'test_atomic_numbers.npz'
    if test_file.exists():
        test_output = output_dir / f'test_energies_{model_suffix}.json'
        test_stats = process_dataset(
            test_file, test_output, "test",
            args.batch_size, device, args.model
        )
        if test_stats:
            all_stats.append(test_stats)
    else:
        print(f"\nWarning: Test file not found: {test_file}")

    # Save summary statistics
    summary_file = output_dir / f'energy_stats_{model_suffix}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'model': args.model,
            'datasets': all_stats
        }, f, indent=2)

    print()
    print("="*60)
    print("Energy Computation Complete!")
    print("="*60)
    print(f"Model: {args.model}")
    for stats in all_stats:
        print(f"  {stats['dataset']}: {stats['valid']}/{stats['total']} structures")
    print(f"\nSummary saved to: {summary_file}")
    print("="*60)


if __name__ == '__main__':
    main()

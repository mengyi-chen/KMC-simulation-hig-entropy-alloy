# Cluster Expansion for High-Entropy Alloy Spinel Oxide

This directory contains code for fitting a cluster expansion (CE) model to the high-entropy alloy (HEA) spinel oxide system with vacancies.

## System Description

**Material**: (Cu, Ni, Fe, Al, Cr, Co)₃O₄ spinel oxide

**Species**:
- **Cation sites**: Cu, Ni, Fe, Al, Cr, Co (6 transition metals)
- **Anion sites**: O (oxygen)
- **Vacancies**:
  - Cation vacancy: `X` (represented as `Li` in cluster expansion)
  - Oxygen vacancy: `XO` (represented as `F` in cluster expansion)

The vacancy placeholders (Li, F) are used as a trick for the cluster expansion fitting since `smol` requires real elements.

## Workflow

### Step 0: Extract Local Environments (`0_extract_local_envs_parallel.py`)

Extracts local 3×3×3 unit cell cubic environments centered at vacancies from large supercell configurations.

**Features**:
- Reads VASP POSCAR files from KMC simulation snapshots (`configs_2025_11_26_14_20_16/`)
- Identifies cation vacancies (X) and oxygen vacancies (XO)
- Extracts 3×3×3 unit cell cubic environments (~25.6 Å) centered at each vacancy
- Sorts atoms by position (XYZ order) to match ClusterExpansionProcessor ordering
- Replaces X→Li and XO→F for cluster expansion compatibility
- Generates random structures for data augmentation (configurable vacancy percentage)
- Performs train/test split with configurable ratio
- Saves structures in NPZ format for fast loading

**Usage**:
```bash
python 0_extract_local_envs_parallel.py --n_workers 32 --max_vac_per_file 50 --test_ratio 0.2
```

**Key Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--n_workers` | 64 | Number of parallel workers |
| `--max_vac_per_file` | 50 | Maximum vacancies to sample per file |
| `--min_atoms` | 100 | Skip structures with ≤ this many atoms |
| `--test_ratio` | 0.2 | Fraction of data for testing |
| `--seed` | 42 | Random seed for train/test split |
| `--n_random` | 5000 | Number of random structures to generate |
| `--vac_pct_min` | 0.1 | Minimum vacancy percentage for random structures |
| `--vac_pct_max` | 0.5 | Maximum vacancy percentage for random structures |

**Output**: `local_structures/` directory with:
- `train_atomic_numbers.npz` - Training set atomic numbers, positions, cell
- `test_atomic_numbers.npz` - Test set atomic numbers, positions, cell
- `all_structures_atomic_numbers.npz` - Combined dataset (backward compatibility)
- `train_metadata.json`, `test_metadata.json` - Metadata for each structure
- `template_*.npz` - First 10 structures saved individually for testing
- `extraction_params.json` - Extraction parameters

---

### Step 1: Compute MACE Energies (`1_compute_mace_energies.py`)

Computes energies using the MACE universal potential with torch-sim for GPU batching.

**Features**:
- Uses torch-sim for efficient batched GPU computation
- Supports various MACE foundation models (medium-omat-0 recommended)
- Removes Li/F (vacancy placeholders) before energy calculation
- Processes train and test sets separately
- Sets `pbc=[True, True, False]` (periodic in XY, non-periodic in Z)
- Outputs energies in JSON format

**Usage**:
```bash
python 1_compute_mace_energies.py --batch_size 32 --gpu_idx 4 --model medium-omat-0
```

**Key Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 16 | Batch size for GPU computation |
| `--gpu_idx` | 5 | GPU index to use |
| `--model` | medium-omat-0 | MACE model name |

**Available MACE Models**:
- MP models: `small`, `medium`, `large`
- MP versions: `small-0b`, `medium-0b`, `small-0b2`, `medium-0b2`, `medium-0b3`, `large-0b2`
- OMAT models: `medium-omat-0`, `small-omat-0`
- MPA model: `medium-mpa-0`

**Output**: `mace_energies/` directory with:
- `train_energies_medium_omat_0.json` - Train set energies
- `test_energies_medium_omat_0.json` - Test set energies
- `energy_stats_medium_omat_0.json` - Summary statistics

---

### Step 2: Process and Save Wrangler

Two versions available:
- `2_process_save_wrangler_parallel.py` - Multi-CPU parallel version (recommended)
- `2_process_save_wrangler_sequential.py` - Simple sequential version

Builds the cluster expansion framework using `smol`.

**Features**:
- Creates primitive structure with mixed occupancies (7 cation species, 2 anion species)
- Defines ClusterSubspace with:
  - Pair cutoff: 6.0 Å
  - Triplet cutoff: 4.0 Å
  - Indicator basis for multi-component alloys
- Uses `subspace.structure_site_mapping()` for correct site mapping
- Uses `processor.allowed_species[i].index(spec)` for correct occupancy encoding
- Computes correlation vectors (feature vectors) for all structures
- Verifies occupancy mapping against template structures
- Parallelizes correlation computation across multiple CPUs

**Usage**:
```bash
# Parallel version (recommended)
python 2_process_save_wrangler_parallel.py --n_workers 32

# Sequential version
python 2_process_save_wrangler_sequential.py
```

**Output**: `ce_data/` directory with:
- `ce_data.npz` - Train feature matrix, energies, structure indices
- `test_ce_data.npz` - Test feature matrix, energies, structure indices
- `subspace.pkl` - ClusterSubspace object
- `processor.pkl` - ClusterExpansionProcessor object
- `site_mapping.json` - Structure atom index → processor site index mapping
- `struct_idx_for_proc_site.json` - Reverse mapping
- `species_to_code.json` - Species → occupancy code mapping
- `stats.json` - Processing statistics

---

### Step 3: Fit ECIs (`3_fit_ECIs.py`)

Fits Effective Cluster Interactions using L1 regularization (LASSO).

**Features**:
- Filters to unique structures by correlation vector (keeps lowest energy)
- Supports two fitting modes:
  - **Single-step**: Standard LASSO with regularization parameter `mu`
  - **Two-step**: Fit point terms first, then cluster terms on residuals (recommended for multi-component)
- Supports cross-validation for hyperparameter selection
- Uses cvxopt (preferred) or sklearn for L1 optimization
- Evaluates on both train and test sets
- Analyzes ECIs by cluster size
- Reports errors per structure and per atom

**Usage**:
```bash
# Single-step fitting
python 3_fit_ECIs.py --mu 1e-4

# Two-step fitting (recommended for multi-component)
python 3_fit_ECIs.py --two_step --mu_point 1e-3 --mu 1e-4

# With cross-validation
python 3_fit_ECIs.py --cv
```

**Key Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--mu` | 1e-4 | Regularization parameter for cluster terms |
| `--mu_point` | 1e-3 | Regularization for point terms (two-step only) |
| `--two_step` | False | Use two-step fitting |
| `--cv` | False | Use cross-validation for regularization selection |

**Output**:
- `ce_data/ecis_L1.pkl` - ECIs in pickle format
- `ce_data/ecis_L1.npy` - ECIs in numpy format
- `ce_data/fitting_stats.json` - Fitting statistics (train/test RMSE, MAE, etc.)

---

### Step 4: Visualization (`4_compare_results.ipynb`)

Jupyter notebook for visualizing results.

**Contents**:
1. Parity plot (predicted vs true energies)
2. Residuals distribution
3. ECI analysis by cluster size
4. Energy distribution by vacancy type
5. Correlation matrix analysis
6. Learning curve (RMSE vs regularization)

---

## Directory Structure

```
cluster_expansion/
├── 0_extract_local_envs_parallel.py      # Extract local environments (parallel)
├── 1_compute_mace_energies.py            # Compute MACE energies
├── 2_process_save_wrangler_parallel.py   # Build CE framework (parallel)
├── 2_process_save_wrangler_sequential.py # Build CE framework (sequential)
├── 3_fit_ECIs.py                         # Fit ECIs
├── 4_compare_results.ipynb               # Visualization notebook
├── test_occupancy.ipynb                  # Occupancy testing notebook
├── README.md                             # This file
├── local_structures/                     # Extracted local environments
│   ├── train_atomic_numbers.npz          # Training structures
│   ├── test_atomic_numbers.npz           # Test structures
│   ├── all_structures_atomic_numbers.npz # Combined structures
│   ├── train_metadata.json               # Training metadata
│   ├── test_metadata.json                # Test metadata
│   ├── metadata.json                     # Combined metadata
│   ├── train_test_split.json             # Split indices
│   ├── template_*.npz                    # Template structures
│   └── extraction_params.json            # Extraction parameters
├── mace_energies/                        # MACE energy predictions
│   ├── train_energies_*.json             # Train energies
│   ├── test_energies_*.json              # Test energies
│   └── energy_stats_*.json               # Energy statistics
├── ce_data/                              # Cluster expansion data
│   ├── ce_data.npz                       # Train feature matrix & energies
│   ├── test_ce_data.npz                  # Test feature matrix & energies
│   ├── subspace.pkl                      # ClusterSubspace
│   ├── processor.pkl                     # ClusterExpansionProcessor
│   ├── site_mapping.json                 # Site mapping
│   ├── struct_idx_for_proc_site.json     # Reverse site mapping
│   ├── species_to_code.json              # Species encoding
│   ├── stats.json                        # Processing statistics
│   ├── ecis_L1.pkl                       # Fitted ECIs (pickle)
│   ├── ecis_L1.npy                       # Fitted ECIs (numpy)
│   └── fitting_stats.json                # Fitting statistics
└── comparison_results/                   # Comparison analysis results
```

## Dependencies

- Python 3.8+
- numpy
- pymatgen
- smol (for cluster expansion)
- torch, torch-sim (for MACE)
- mace-torch (MACE models)
- ase (Atomic Simulation Environment)
- cvxopt or sklearn (for L1 optimization)
- matplotlib, seaborn (for visualization)
- tqdm (progress bars)

## Running the Full Pipeline

```bash
# Activate environment with dependencies
conda activate torchsim

# Step 0: Extract structures
python 0_extract_local_envs_parallel.py --n_workers 32 --n_random 5000

# Step 1: Compute MACE energies
python 1_compute_mace_energies.py --batch_size 32 --gpu_idx 4

# Step 2: Build cluster expansion (parallel version)
python 2_process_save_wrangler_parallel.py --n_workers 32

# Step 3: Fit ECIs
python 3_fit_ECIs.py --two_step --mu_point 1e-3 --mu 1e-4

# Step 4: Visualize (in Jupyter)
jupyter notebook 4_compare_results.ipynb
```

## Technical Notes

1. **Vacancy Placeholders**: The cluster expansion uses Li and F as placeholders for cation and oxygen vacancies respectively. This is because the `smol` package requires real elements for the cluster expansion basis.

2. **Energy Calculation**: When computing MACE energies, the vacancy placeholders (Li, F) are removed since they are not real atoms. The energy represents only the real atoms in the local environment.

3. **Structure Ordering**: Atoms are sorted by position (X → Y → Z, XYZ order) to match the `ClusterExpansionProcessor` ordering. This is critical for correct occupancy mapping.

4. **Site Mapping**: The code uses `subspace.structure_site_mapping()` to establish the authoritative mapping between structure atom indices and processor site indices.

5. **Periodic Boundary Conditions**: 
   - Local structures have PBC in X and Y directions, but not Z
   - MACE computation uses `pbc=[True, True, False]`

6. **Two-Step Fitting**: Recommended for multi-component alloys. First fits point terms (single-site clusters) with stronger regularization, then fits cluster terms on residuals with weaker regularization.

7. **Regularization**: The L1 regularization parameter (`mu`) controls sparsity. Smaller values give more non-zero ECIs but may overfit. Use cross-validation (`--cv`) to select optimal regularization.

## References

- SMOL: https://cedergrouphub.github.io/smol/
- MACE: https://github.com/ACEsuit/mace
- torch-sim: https://github.com/LLNL/torch-sim

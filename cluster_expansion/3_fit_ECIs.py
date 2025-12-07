"""
Step 3: Fit Effective Cluster Interactions (ECIs) using L1 regularization

This script:
1. Loads the TRAIN feature matrix and energies from ce_data.npz
2. Loads the ClusterSubspace from subspace.pkl
3. Performs L1 regularized fitting (LASSO) to get sparse ECIs
4. Optionally separates point terms from cluster terms (two-step fitting)
5. Evaluates on TEST data if available
6. Saves ECIs and fitting statistics

Usage:
    python 3_fit_ECIs.py [--mu 1e-4] [--two_step] [--cv]
"""

import numpy as np
import json
import pickle
import argparse
from pathlib import Path

from smol.cofe import ClusterSubspace

from cvxopt import matrix, spdiag, mul, div, sqrt
from cvxopt import blas, lapack, solvers
import math
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV

def cluster_by_id(subspace):
    """
    Return the cluster size for each correlation function.

    Returns:
        clusterSize: Array of cluster sizes (number of sites in each cluster)
    """
    clusterSize = np.zeros(subspace.num_corr_functions)

    for ii, orbit in enumerate(subspace.orbits):
        first_id = orbit.bit_id
        if ii < len(subspace.orbits) - 1:
            last_id = subspace.orbits[ii + 1].bit_id - 1
        else:
            last_id = subspace.num_corr_functions - 1

        for idx in range(first_id, last_id + 1):
            clusterSize[idx] = len(orbit.base_cluster.sites)

    return clusterSize


def l1regls(A, b):
    """
    L1-regularized least squares using cvxopt.
    Solves: minimize || A*x - b ||_2^2 + || x ||_1
    """
    m, n = A.size
    q = matrix(1.0, (2*n, 1))
    q[:n] = -2.0 * A.T * b

    def P(u, v, alpha=1.0, beta=0.0):
        v *= beta
        v[:n] += alpha * 2.0 * A.T * (A * u[:n])

    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        v *= beta
        v[:n] += alpha * (u[:n] - u[n:])
        v[n:] += alpha * (-u[:n] - u[n:])

    h = matrix(0.0, (2*n, 1))
    S = matrix(0.0, (m, m))
    v = matrix(0.0, (m, 1))

    def Fkkt(W):
        d1, d2 = W['di'][:n]**2, W['di'][n:]**2
        ds = math.sqrt(2.0) * div(mul(W['di'][:n], W['di'][n:]), sqrt(d1 + d2))
        d3 = div(d2 - d1, d1 + d2)
        Asc = A * spdiag(ds**-1)
        blas.syrk(Asc, S)
        S[::m+1] += 1.0
        lapack.potrf(S)

        def g(x, y, z):
            x[:n] = 0.5 * (x[:n] - mul(d3, x[n:]) +
                           mul(d1, z[:n] + mul(d3, z[:n])) -
                           mul(d2, z[n:] - mul(d3, z[n:])))
            x[:n] = div(x[:n], ds)
            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)
            x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1 + d2) - mul(d3, x[:n])
            z[:n] = mul(W['di'][:n], x[:n] - x[n:] - z[:n])
            z[n:] = mul(W['di'][n:], -x[:n] - x[n:] - z[n:])
        return g

    solvers.options['show_progress'] = False
    return solvers.coneqp(P, q, G, h, kktsolver=Fkkt)['x'][:n]


def l1_optimize(A, f, mu):
    """
    L1 optimization using cvxopt.
    
    Original problem: min ||f - Ax||^2 + mu * ||x||_1
    
    Multiply by 1/mu: min (1/mu)||f - Ax||^2 + ||x||_1
    
    This equals: min ||f/sqrt(mu) - A/sqrt(mu) * x||^2 + ||x||_1
    
    cvxopt solves: min ||Ax - b||^2 + ||x||_1
    So set A' = A/sqrt(mu), b' = f/sqrt(mu)
    """
    scale = 1.0 / np.sqrt(mu)
    A1 = matrix(A * scale)
    b = matrix(f * scale)
    ecis = np.array(l1regls(A1, b)).flatten()
    return ecis

def fit_ecis_two_step(A, f, subspace, mu_point=1e-3, mu_cluster=1e-4):
    """
    Two-step ECI fitting:
    1. Fit point terms (single-site clusters) first
    2. Fit remaining cluster terms on residuals
    """
    clusterSize = cluster_by_id(subspace)

    # Point terms: cluster size <= 1
    pointID = np.where(clusterSize <= 1)[0]

    # Step 1: Fit point terms
    A_point = A[:, pointID].copy()
    w_point = l1_optimize(A=A_point, f=f, mu=mu_point)

    # Compute residuals
    f_point = A_point @ w_point
    f_res = f - f_point

    rmse_point = np.sqrt(np.mean(f_res**2)) * 1000
    print(f"  Point term RMSE: {rmse_point:.2f} meV")

    # Step 2: Fit cluster terms on residuals
    A_res = A.copy()
    A_res[:, pointID] = 0

    ecis_cluster = l1_optimize(A=A_res, f=f_res, mu=mu_cluster)

    # Combine
    ecis = ecis_cluster.copy()
    ecis[pointID] = w_point

    return ecis


def fit_ecis_single_step(A, f, mu=1e-4):
    """Single-step L1 regularized fitting."""
    return l1_optimize(A=A, f=f, mu=mu)


def get_unique_indices(feature_matrix, energies, tol=1e-6):
    """Get indices of unique correlation vectors, keeping lowest energy."""
    rounded = np.round(feature_matrix, decimals=int(-np.log10(tol)))
    unique_dict = {}
    for i in range(len(feature_matrix)):
        key = tuple(rounded[i])
        if key not in unique_dict or energies[i] < energies[unique_dict[key]]:
            unique_dict[key] = i
    return np.array(sorted(unique_dict.values()))


def main():
    parser = argparse.ArgumentParser(description='Fit ECIs with different regularization methods')
    parser.add_argument('--method', type=str, default='lasso', 
                        choices=['lasso', 'ridge'],
                        help='Regularization method (default: lasso)')
    parser.add_argument('--mu', type=float, default=1,
                        help='Regularization parameter (default: 1e)')
    parser.add_argument('--mu_point', type=float, default=1,
                        help='Regularization for point terms in two-step (default: 1e)')
    parser.add_argument('--two_step', action='store_true',
                        help='Use two-step fitting (point terms first)')
    parser.add_argument('--cv', action='store_true',
                        help='Use cross-validation to select regularization')
    args = parser.parse_args()

    # Paths
    ce_data_dir = Path('./ce_data')
    ce_data_file = ce_data_dir / 'ce_data.npz'
    subspace_file = ce_data_dir / 'subspace.pkl'

    print("="*60)
    print("Fitting Effective Cluster Interactions (ECIs)")
    print("="*60)

    # 1. Load TRAIN data
    print("\n[1] Loading TRAIN data...")

    if not ce_data_file.exists():
        print(f"Error: {ce_data_file} not found!")
        print("Please run 2_process_save_wrangler_parallel.py first.")
        return

    data = np.load(ce_data_file)
    A_full = data['feature_matrix']
    f_full = data['energies']
    n_atoms = data['n_atoms']
    n_real_atoms = data['n_real_atoms']

    print(f"  Loaded {len(f_full)} TRAIN structures")
    print(f"  Feature matrix shape: {A_full.shape}")

    with open(subspace_file, 'rb') as fp:
        subspace = pickle.load(fp)
    print(f"  Correlation functions: {subspace.num_corr_functions}")

    # 2. Filter unique TRAIN structures
    print("\n[2] Filtering unique TRAIN structures...")
    indices = get_unique_indices(A_full, f_full)
    A = A_full[indices]
    f = f_full[indices]

    print(f"  Unique TRAIN: {len(indices)} / {len(f_full)}")
    print(f"  Matrix rank: {np.linalg.matrix_rank(A)}")
    print(f"  Energy range: [{f.min():.2f}, {f.max():.2f}] eV")

    # 3. Fit ECIs
    print("\n[3] Fitting ECIs...")
    print(f"  Method: {args.method.upper()}")

    if args.cv:
        print("  Using cross-validation...")
        n_samples = len(f)
        
        if args.method == 'lasso':
            # LASSO with CV
            # sklearn LASSO: min (1/2n)||y-Xw||² + alpha||w||₁
            # Standard form: min ||y-Xw||² + mu||w||₁
            # => alpha = mu / (2n)
            alphas = np.logspace(-6, 6, 50) / (2.0 * n_samples)
            model = LassoCV(alphas=alphas, cv=5, fit_intercept=False, max_iter=10000)
            model.fit(A, f)
            ecis = model.coef_
            mu_selected = model.alpha_ * 2.0 * n_samples
            print(f"  Selected alpha: {model.alpha_:.6f} (mu: {mu_selected:.6e})")
            
        elif args.method == 'ridge':
            # Ridge with CV
            # sklearn Ridge: min ||y-Xw||² + alpha||w||²
            # Standard form:  min ||y-Xw||² + mu||w||²
            # => alpha = mu (direct equality, no normalization)
            alphas = np.logspace(-6, 2, 50)
            model = RidgeCV(alphas=alphas, cv=5, fit_intercept=False)
            model.fit(A, f)
            ecis = model.coef_
            mu_selected = model.alpha_
            print(f"  Selected alpha: {model.alpha_:.6f} (mu: {mu_selected:.6e})")
            
        print(f"  Non-zero ECIs: {np.sum(np.abs(ecis) > 1e-8)}")
        
    elif args.two_step:
        print(f"  Two-step fitting (mu_point={args.mu_point}, mu_cluster={args.mu})")
        if args.method != 'lasso':
            print(f"  Warning: Two-step only implemented for LASSO, using LASSO")
        ecis = fit_ecis_two_step(A, f, subspace,
                                  mu_point=args.mu_point,
                                  mu_cluster=args.mu)
    else:
        # Single-step fitting
        print(f"  Single-step fitting (mu={args.mu})")
        
        if args.method == 'lasso':
            ecis = fit_ecis_single_step(A, f, mu=args.mu)
            
        elif args.method == 'ridge':
            # Ridge: min ||f - Ax||^2 + mu * ||x||^2
            # sklearn Ridge: min ||f - Ax||^2 + alpha * ||x||^2
            # => alpha = mu (direct equality, Ridge has NO normalization)
            alpha = args.mu
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(A, f)
            ecis = model.coef_
            print(f"  (Ridge uses L2 regularization - all ECIs non-zero)")

    # 4. Evaluate on TRAIN data
    print("\n[4] Evaluating on TRAIN data...")
    f_pred = A @ ecis
    residuals = f_pred - f

    # Per-structure errors
    rmse_train = np.sqrt(np.mean(residuals**2))
    mae_train = np.mean(np.abs(residuals))
    max_err_train = np.max(np.abs(residuals))

    # Per-atom errors (using real atoms, excluding vacancies)
    n_real_atoms_used = n_real_atoms[indices]
    residuals_per_atom = residuals / n_real_atoms_used
    rmse_per_atom_train = np.sqrt(np.mean(residuals_per_atom**2))
    mae_per_atom_train = np.mean(np.abs(residuals_per_atom))

    n_nonzero = np.sum(np.abs(ecis) > 1e-8)

    print(f"\n  TRAIN Results (per structure):")
    print(f"    RMSE:       {rmse_train:.4f} eV  ({rmse_train*1000:.2f} meV)")
    print(f"    MAE:        {mae_train:.4f} eV  ({mae_train*1000:.2f} meV)")
    print(f"    Max error:  {max_err_train:.4f} eV  ({max_err_train*1000:.2f} meV)")
    print(f"\n  TRAIN Results (per atom):")
    print(f"    RMSE:       {rmse_per_atom_train*1000:.4f} meV/atom")
    print(f"    MAE:        {mae_per_atom_train*1000:.4f} meV/atom")
    print(f"\n  Non-zero ECIs: {n_nonzero} / {len(ecis)}")

    # 4b. Evaluate on TEST data (if available)
    test_stats = None
    test_data_file = ce_data_dir / 'test_ce_data.npz'
    if test_data_file.exists():
        print("\n[4b] Evaluating on TEST data...")
        test_data = np.load(test_data_file)
        A_test = test_data['feature_matrix']
        f_test = test_data['energies']
        n_atoms_test = test_data['n_atoms']
        n_real_atoms_test = test_data['n_real_atoms']

        f_pred_test = A_test @ ecis
        residuals_test = f_pred_test - f_test

        rmse_test = np.sqrt(np.mean(residuals_test**2))
        mae_test = np.mean(np.abs(residuals_test))
        max_err_test = np.max(np.abs(residuals_test))

        residuals_per_atom_test = residuals_test / n_real_atoms_test
        rmse_per_atom_test = np.sqrt(np.mean(residuals_per_atom_test**2))
        mae_per_atom_test = np.mean(np.abs(residuals_per_atom_test))

        print(f"\n  TEST Results (per structure):")
        print(f"    RMSE:       {rmse_test:.4f} eV  ({rmse_test*1000:.2f} meV)")
        print(f"    MAE:        {mae_test:.4f} eV  ({mae_test*1000:.2f} meV)")
        print(f"    Max error:  {max_err_test:.4f} eV  ({max_err_test*1000:.2f} meV)")
        print(f"\n  TEST Results (per atom):")
        print(f"    RMSE:       {rmse_per_atom_test*1000:.4f} meV/atom")
        print(f"    MAE:        {mae_per_atom_test*1000:.4f} meV/atom")

        test_stats = {
            'n_test_structures': int(len(f_test)),
            'rmse_eV_test': float(rmse_test),
            'mae_eV_test': float(mae_test),
            'max_error_eV_test': float(max_err_test),
            'rmse_meV_per_atom_test': float(rmse_per_atom_test * 1000),
            'mae_meV_per_atom_test': float(mae_per_atom_test * 1000),
        }
    else:
        print("\n  (No TEST data found - skipping test evaluation)")

    # ECIs by cluster size
    clusterSize = cluster_by_id(subspace)
    print("\n  ECIs by cluster size:")
    for size in range(int(max(clusterSize)) + 1):
        mask = clusterSize == size
        n_total = np.sum(mask)
        n_nz = np.sum(np.abs(ecis[mask]) > 1e-8)
        if n_total > 0:
            print(f"    Size {size}: {n_nz}/{n_total} non-zero")

    # 5. Save
    print("\n[5] Saving results...")

    with open(ce_data_dir / 'ecis_L1.pkl', 'wb') as fp:
        pickle.dump(ecis, fp)

    np.save(ce_data_dir / 'ecis_L1.npy', ecis)

    stats = {
        'n_train_structures': int(len(f_full)),
        'n_train_unique': int(len(indices)),
        'n_corr_functions': int(len(ecis)),
        'n_nonzero_ecis': int(n_nonzero),
        'rmse_eV_train': float(rmse_train),
        'mae_eV_train': float(mae_train),
        'max_error_eV_train': float(max_err_train),
        'rmse_meV_per_atom_train': float(rmse_per_atom_train * 1000),
        'mae_meV_per_atom_train': float(mae_per_atom_train * 1000),
        'method': args.method,
        'mu': float(args.mu),
        'mu_point': float(args.mu_point),
        'two_step': args.two_step,
        'cv': args.cv
    }

    # Add test stats if available
    if test_stats is not None:
        stats.update(test_stats)

    with open(ce_data_dir / 'fitting_stats.json', 'w') as fp:
        json.dump(stats, fp, indent=2)

    print(f"\n  Saved:")
    print(f"    - {ce_data_dir / 'ecis_L1.pkl'}")
    print(f"    - {ce_data_dir / 'ecis_L1.npy'}")
    print(f"    - {ce_data_dir / 'fitting_stats.json'}")

    print("\n" + "="*60)
    print("ECI Fitting Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

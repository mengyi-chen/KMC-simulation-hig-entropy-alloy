import numpy as np
import json
from monty.serialization import loadfn, dumpfn
from pymatgen.core import Structure
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.cofe.space import Cluster
from smol.cofe.extern import EwaldTerm
from smol.cofe.wrangling.tools import unique_corr_vector_indices
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, CompoundPhaseDiagram
from pymatgen.core import Composition
from itertools import combinations
from copy import deepcopy
import time
import pickle

from cvxopt import matrix, spdiag, mul, div, sqrt
from cvxopt import blas, lapack, solvers
import math


def cluster_by_id(subSpace):
    """
    Return the cluster id and cluster size
    """
    allOrbits = subSpace.orbits
    orbitIndices = np.arange(subSpace.num_corr_functions)
    clusterList = [None]*subSpace.num_corr_functions
    clusterSize = np.zeros(subSpace.num_corr_functions)

    for ii, orbit in enumerate(allOrbits):
        first_id = orbit.bit_id

        if ii < len(allOrbits)-1:
            last_id = allOrbits[ii+1].bit_id - 1 # first and last id belongs to the same cluster

        if first_id <= last_id:
            for idx in range(first_id, last_id+1):
                inIdx = idx-first_id
                clusterList[idx] = (orbit.base_cluster, orbit.multiplicity, orbit.bit_combos[inIdx])
                clusterSize[idx] = len(orbit.base_cluster.sites)

        else:
            for idx in range(first_id, subSpace.num_corr_functions):
                inIdx = idx-first_id
                clusterList[idx] = (orbit.base_cluster, orbit.multiplicity, orbit.bit_combos[inIdx])
                clusterSize[idx] = len(orbit.base_cluster.sites)
    return clusterList, clusterSize


def l1regls(A, b):
    """

    Returns the solution of l1-norm regularized least-squares problem

        minimize || A*x - b ||_2^2  + || x ||_1.

    """

    m, n = A.size
    q = matrix(1.0, (2*n,1))
    q[:n] = -2.0 * A.T * b

    def P(u, v, alpha = 1.0, beta = 0.0 ):
        """
            v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v
        """
        v *= beta
        v[:n] += alpha * 2.0 * A.T * (A * u[:n])

    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        """
            v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        v *= beta
        v[:n] += alpha*(u[:n] - u[n:])
        v[n:] += alpha*(-u[:n] - u[n:])

    h = matrix(0.0, (2*n,1))

    # Customized solver for the KKT system
    #
    #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
    #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
    #
    # where D1 = W['di'][:n]**2, D2 = W['di'][:n]**2.
    #
    # We first eliminate zl and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] =
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] +
    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
    #         - (D2-D1)*(D1+D2)^-1 * x[:n]
    #
    #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
    #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m,m))
    Asc = matrix(0.0, (m,n))
    v = matrix(0.0, (m,1))

    def Fkkt(W):
        # Factor
        #
        #     S = A*D^-1*A' + I
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

        d1, d2 = W['di'][:n]**2, W['di'][n:]**2

        # ds is square root of diagonal of D
        ds = math.sqrt(2.0) * div( mul( W['di'][:n], W['di'][n:]),
            sqrt(d1+d2) )
        d3 =  div(d2 - d1, d1 + d2)

        # Asc = A*diag(d)^-1/2
        Asc = A * spdiag(ds**-1)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[::m+1] += 1.0
        lapack.potrf(S)

        def g(x, y, z):
            x[:n] = 0.5 * (x[:n] - mul(d3, x[n:]) +
                           mul(d1, z[:n] + mul(d3, z[:n])) -
                           mul(d2, z[n:] - mul(d3, z[n:])))
            x[:n] = div( x[:n], ds)

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] -
            #         (D2-D1)*(D1+D2)^-1 * bx[n:] +
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )

            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)

            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]
            x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1+d2)\
                - mul( d3, x[:n] )

            # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
            # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
            z[:n] = mul(W['di'][:n], x[:n] - x[n:] - z[:n])
            z[n:] = mul(W['di'][n:], -x[:n] - x[n:] - z[n:])

        return g

    return solvers.coneqp(P, q, G, h, kktsolver=Fkkt)['x'][:n]


def l1_optimize(A, f, mu, weights=None):
    """
    Optimization via L1-regularization by solving a convex optimization
    problem, where the definition of mu is different from the normal expression.

    Standard way: |E - E_DFT|^2 + mu *|ECIs|_1 ,
    Convex solver way: mu * |E - E_DFT|^2 + |ECIs|_1

    Thus, the inverse of the mu parametrized here is equal to the mu in other
    solvers such as LASSO in sklearn.

    Parameters
    ----------
    A: feature matrix as numpy array of shape mxn
    f: scalar property vector as numpy array of shape mx1
    mu: regularization parameter
    weights: scalar weight vector as numpy array of shape mx1

    Returns
    -------
    Fitted ECI values as numpy array of shape A.shape[1]
    """
    mu = 1 / mu

    if weights is None:
        weights = np.ones(len(f))

    A_w = A * weights[:, None] ** 0.5
    f_w = f * weights ** 0.5

    # solvers.options['show_progress'] = False
    A1 = matrix(A)
    b = matrix(f * mu)
    ecis = (np.array(l1regls(A1, b)) / mu).flatten()
    return ecis


def main():

    with open('./dataset_CE/wrangler.json', 'r') as fp:
        wrangler_json = json.load(fp)

    wrangler = StructureWrangler.from_dict(wrangler_json)
    subspace = wrangler.cluster_subspace

    #### initialization if wranger is empty ####
    if len(wrangler.structures) == 0:
        ecis_l1 = np.random.rand(subspace.num_corr_functions + 1)
        with open('./dataset_CE/ecis_L1_new', 'wb') as fp:
            pickle.dump(ecis_l1, fp)
        return 0


    f_GS = np.min(wrangler.get_property_vector('energy'))

    indices = unique_corr_vector_indices(wrangler, property_key='energy', filter_by='min')
    duplicate_indices = list(set(range(len(wrangler.structures)))- set(indices))
    ### fit on the unique indices with the lowest energy ###
    A_valid = wrangler.feature_matrix[indices]
    f_valid = wrangler.get_property_vector('energy')[indices]

    print("min energy = ", np.min(f_valid))

    print("Feature matrix shape: ", A_valid.shape)
    print("Feature matrix rank: ", np.linalg.matrix_rank(A_valid))

    _, clusterSize = cluster_by_id(subspace)

    pointID = np.where(clusterSize <=1)[0]
    pointID = np.append(pointID, -1)

    print(pointID)

    A_point = A_valid[:, pointID]
    A_point[:, 0] = 0
    w_point = l1_optimize(A=A_point, f= f_valid - np.average(f_valid), mu=1e-3)

    w_point[0] = np.average(f_valid)

    A_res = deepcopy(A_valid)
    A_res[:, pointID] = 0

    A_point[:, 0] = 1
    f_point = A_point @ w_point
    f_res = f_valid - f_point
    f_shift = deepcopy(f_point)



    print("Point term rank: ", np.linalg.matrix_rank(A_point))

    print(w_point)
    print("Fitted dielectrict:", 1/w_point[-1])
    print(np.average(f_res ** 2)**0.5 * 1000)



    ecis_l1 = l1_optimize(A=A_res, f= f_res, mu=1e-4)


    ecis_l1[pointID] = w_point


    nonZero = np.sum(np.abs(ecis_l1) > 1e-6)
    rmse = np.average((A_valid @ ecis_l1 - f_valid)**2)**0.5 * 1000
    print("Non-Zero: ", nonZero)
    print("RMSE", rmse)


    with open('./dataset_CE/ecis_L1_new', 'wb') as fp:
        pickle.dump(ecis_l1, fp)


if __name__ == '__main__':
    main()

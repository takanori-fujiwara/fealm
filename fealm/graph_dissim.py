# graph dissimilarity/distane related functions are implemented while referring
# to netrd's implementations (https://github.com/netsiphd/netrd)
# But, each function here is significantly different from netrd's implementation

# netrd's implentation is MIT License and Copyright (c) 2019 NetSI 2019 Collabathon Team

import random
import numpy as np
import scipy as sp
import networkx as nx
import netrd.distance as nd
from collections import deque
from scipy.spatial.distance import directed_hausdorff

from scipy.linalg import expm, eigvalsh, eigvals, eigh, inv, solve
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.stats import entropy
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix, issparse

import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


## updown_linear_approx and eigenvalues_auto are modifed versions from netlsd's (https://github.com/xgfs/NetLSD, released under MIT licence, Copyright (c) 2018 Anton Tsitsulin)
## Modified points are:
## - adding args of [eigsh_tol, eigsh_v0] to control the tolerance
##   to avoid the slow convergence based on the random initial solution and
##   to make the result reporoduciable by indicating the initial solution
## - adding error handling for the no convergence case (see http://jiffyclub.github.io/scipy/release.0.9.0.html).
## - bug-fixing in the selection of sparse and dense arrays based on n_eivals
##
def updown_linear_approx(eigvals_lower, eigvals_upper, nv):
    nal = len(eigvals_lower)
    nau = len(eigvals_upper)
    if nv < nal + nau:
        raise ValueError(
            'Number of supplied eigenvalues ({0} lower and {1} upper) is higher than number of nodes ({2})!'
            .format(nal, nau, nv))
    ret = np.zeros(nv)
    ret[:nal] = eigvals_lower
    ret[-nau:] = eigvals_upper
    ret[nal - 1:-nau + 1] = np.linspace(eigvals_lower[-1], eigvals_upper[0],
                                        nv - nal - nau + 2)

    return ret


def eigenvalues_auto(X, n_eivals='auto', eigsh_tol=0, eigsh_v0=None):
    do_full = True
    n_lower = 150
    n_upper = 150
    nv = X.shape[0]

    if n_eivals == 'auto':
        if X.shape[0] > 1024:
            do_full = False
    if n_eivals == 'full':
        do_full = True
    if isinstance(n_eivals, int):
        n_lower = n_upper = n_eivals
        if n_lower + n_upper < nv:
            do_full = False
    if isinstance(n_eivals, tuple):
        n_lower, n_upper = n_eivals
        if n_lower + n_upper < nv:
            do_full = False

    if do_full and issparse(X):
        X = X.toarray()

    if issparse(X):
        if n_lower == n_upper:
            try:
                tr_eivals = eigsh(X,
                                  2 * n_lower,
                                  which='BE',
                                  return_eigenvectors=False,
                                  tol=eigsh_tol,
                                  v0=eigsh_v0)
            except ArpackNoConvergence as err:
                # this will get partially converged eigenvalues
                tr_eivals = err.eigenvalues

            return updown_linear_approx(tr_eivals[:n_upper],
                                        tr_eivals[n_upper:], nv)
        else:
            try:
                lo_eivals = eigsh(X,
                                  n_lower,
                                  which='SM',
                                  return_eigenvectors=False,
                                  tol=eigsh_tol,
                                  v0=eigsh_v0)[::-1]
            except ArpackNoConvergence as err:
                # this will get partially converged eigenvalues
                lo_eivals = err.eigenvalues

            try:
                up_eivals = eigsh(X,
                                  n_upper,
                                  which='LM',
                                  return_eigenvectors=False,
                                  tol=eigsh_tol,
                                  v0=eigsh_v0)
            except ArpackNoConvergence as err:
                # this will get partially converged eigenvalues
                up_eivals = err.eigenvalues

            return updown_linear_approx(lo_eivals, up_eivals, nv)
    else:
        if do_full:
            return eigvalsh(X)
        else:
            lo_eivals = eigvalsh(X, eigvals=(0, n_lower - 1))
            up_eivals = eigvalsh(X, eigvals=(nv - n_upper - 1, nv - 1))
            return updown_linear_approx(lo_eivals, up_eivals, nv)


def _to_undirected(G, weighted=False):
    if weighted:
        # this is O(n^2)
        undirected_G = G + G.T - G * G.T
    else:
        # this is O(n)
        undirected_G = ((G + G.T) > 0).astype(float)

    return undirected_G


def _js_divergence(P, Q):
    M = 0.5 * (P + Q)
    jsd = 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))

    return 0 if np.isclose(jsd, 0.0) else jsd


def _communicability(G, n_eigvals=10, beta=1, from_networkx_graph=False):
    # by limiting n_eigvals, we can achieve faster computations
    # n_eigvals related to the potential # of communities

    # assume G is unweighted
    if from_networkx_graph:
        nodelist = list(G)
        A = nx.to_numpy_array(G, nodelist)
    else:
        A = G.toarray()
    A = _to_undirected(A)

    n = A.shape[0]
    w, V = eigh(A, subset_by_index=[n - n_eigvals, n - 1])
    expw = np.exp(beta * w)
    C = V @ np.diag(expw) @ V.T

    return C


def _communicability_exp(G, from_networkx_graph=False):
    expA = None
    if from_networkx_graph:
        A = nx.to_numpy_array(G)
    else:
        A = G.toarray()
    A = _to_undirected(A)

    return expm(A)


def _communicability_delta(G, n_eigvals=10, beta=1, from_networkx_graph=False):
    # Equation 25 of the original paper
    # by limiting n_eigvals, we can achieve faster computations
    # n_eigvals related to the potential # of communities

    # assume G is unweighted
    if from_networkx_graph:
        nodelist = list(G)
        A = nx.to_numpy_array(G, nodelist)
    else:
        A = G.toarray()
    A = _to_undirected(A)

    n = A.shape[0]
    # w is ordered from small to large
    w, V = eigh(A, subset_by_index=[n - n_eigvals, n - 1])
    expw = np.exp(beta * w)

    # communicability without the first eigenval and eigenvector
    C = V[:, :-1] @ np.diag(expw[:-1]) @ V[:, :-1].T

    return C


def communicability_dissim(G1,
                           G2,
                           comm_G1=None,
                           comm_G2=None,
                           method='delta',
                           measure='l2',
                           n_eigvals=5,
                           from_networkx_graphs=False):
    '''
    This version is more than 5x faster than nd.CommunicabilityJSD().dist
    when using method='exp' and measure='jsd'

    method: 'normal', 'exp', 'delta'
    measure: 'jsd', 'hausdorff', 'l2'
    '''
    comm_G1 = None
    comm_G2 = None
    communicability_func = None
    if method == 'normal':
        communicability_func = lambda G: _communicability(
            G, n_eigvals=n_eigvals, from_networkx_graph=from_networkx_graphs)
    elif method == 'exp':
        communicability_func = lambda G: _communicability_exp(
            G, from_networkx_graph=from_networkx_graphs)
    elif method == 'delta':
        communicability_func = lambda G: _communicability_delta(
            G, n_eigvals=n_eigvals, from_networkx_graph=from_networkx_graphs)

    if comm_G1 is None:
        comm_G1 = communicability_func(G1)
    if comm_G2 is None:
        comm_G2 = communicability_func(G2)

    dissim = None
    if measure == 'jsd':
        lil_sigma1 = np.triu(comm_G1).flatten()
        lil_sigma2 = np.triu(comm_G2).flatten()

        # not taking nonzero is probably much faster
        big_sigma1 = lil_sigma1.sum()
        big_sigma2 = lil_sigma2.sum()

        P1 = np.sort(lil_sigma1 / big_sigma1)
        P2 = np.sort(lil_sigma2 / big_sigma2)

        dissim = _js_divergence(P1, P2)
    elif measure == 'hausdorff':
        dissim = max(
            directed_hausdorff(comm_G1, comm_G2)[0],
            directed_hausdorff(comm_G2, comm_G1)[0])
    elif measure == 'l2':
        dissim = np.sqrt(np.sum((comm_G1 - comm_G2)**2))

    return dissim


def _degree_matrix(A, return_diags=False):
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = sp.sparse.spdiags(diags, [0], m, n, format="csr")

    if return_diags:
        return D, diags
    else:
        return D


def _laplacian_matrix(A):
    D = _degree_matrix(A)

    return D - A


def _normalized_laplacian_matrix(A):
    D, diags = _degree_matrix(A, return_diags=True)
    L = D - A

    with sp.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0

    n, m = A.shape
    DH = sp.sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
    L = DH @ (L @ DH)

    return L


def _lsd_trace_signature(G,
                         normalization=None,
                         timescales=np.logspace(-2, 2, 256),
                         n_eigvals=10,
                         from_networkx_graph=False,
                         eigsh_tol=0,
                         eigsh_v0=None):
    if from_networkx_graph:
        nodelist = list(G)
        G_ = nx.to_scipy_sparse_matrix(G, nodelist)
    else:
        G_ = G

    L = _normalized_laplacian_matrix(_to_undirected(G_))

    # Note: this is O(n_nodes * n_nodes * n_eigvals)
    # also netlsd library used n_eivals instead of n_eigvals (probably typo)
    # TDOO: maybe we should set tol for this (speed depends on random vec v0)
    w = eigenvalues_auto(L,
                         n_eivals=n_eigvals,
                         eigsh_tol=eigsh_tol,
                         eigsh_v0=eigsh_v0)

    signature = np.sum(np.exp(-timescales[:, np.newaxis] @ w[:, np.newaxis].T),
                       axis=1)

    # normalization
    if normalization == 'empty':
        signature = signature / L.shape[0]
    elif normalization == 'complete':
        n = L.shape[0]
        signature = signature / (1 + (n - 1) * np.exp(-(1 + 1 /
                                                        (n - 1)) * timescales))
    return signature


def netlsd(G1,
           G2,
           sig1=None,
           sig2=None,
           normalization=None,
           timescales=np.logspace(-2, 2, 256),
           n_eigvals='auto',
           from_networkx_graphs=False,
           eigsh_tol=0,
           eigsh_v0=None):
    '''
    This version is currently about 20x faster than nd.NetLSD().dist
    when using from_networkx_graphs=False. When from_networkx_graphs=True, still
    8x faster than nd.NetLSD().dist.

    # this documentation is from https://github.com/xgfs/NetLSD and modifeied for n_eigvals='auto' case
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    n_eigvals : string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised.
            'full' computes all eigenvalues.
            'auto' uses (25, 25) when # of average nodes in G1 and G2 is larger than 50.
            (xgfs's NetLSD sets 150, i.e., (150, 150), when # of nodes is larger than 1024)
        If int, compute n_eigvals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    eigsh_tol, eigsh_v0: eigsh's option used for eigenvalue-based approximation (check SciPy's eigsh).
    '''
    if n_eigvals == 'auto':
        if (G1.shape[0] + G2.shape[0]) / 2 > 50:
            n_eigvals = (25, 25)
        else:
            n_eigvals = 'full'

    if sig1 is None:
        sig1 = _lsd_trace_signature(G1,
                                    normalization=normalization,
                                    timescales=timescales,
                                    n_eigvals=n_eigvals,
                                    from_networkx_graph=from_networkx_graphs,
                                    eigsh_tol=eigsh_tol,
                                    eigsh_v0=eigsh_v0)
    if sig2 is None:
        sig2 = _lsd_trace_signature(G2,
                                    normalization=normalization,
                                    timescales=timescales,
                                    n_eigvals=n_eigvals,
                                    from_networkx_graph=from_networkx_graphs,
                                    eigsh_tol=eigsh_tol,
                                    eigsh_v0=eigsh_v0)

    return np.linalg.norm(sig1 - sig2)


def _lsd_diag_signature(G,
                        timescales=np.logspace(-2, 2, 64),
                        from_networkx_graph=False):
    L = nx.normalized_laplacian_matrix(G).toarray(
    ) if from_networkx_graph else _normalized_laplacian_matrix(G).toarray()

    w, V = eigh(L)

    # probably using iteration is faster (avoiding generating a big tensor)
    signature = np.zeros((V.shape[0], len(timescales)))
    for i, t in enumerate(timescales):
        signature[:, i] = np.diag(V @ np.diag(np.exp(-t * w)) @ V.T)

    return signature


def netlsd_with_node_correspondence(G1,
                                    G2,
                                    timescales=np.logspace(-2, 2, 64),
                                    from_networkx_graphs=False):

    sig1 = _lsd_diag_signature(G1,
                               timescales=timescales,
                               from_networkx_graph=from_networkx_graphs)
    sig2 = _lsd_diag_signature(G2,
                               timescales=timescales,
                               from_networkx_graph=from_networkx_graphs)

    return np.sum((sig1 - sig2)**2)


def _process_deltacon(G, from_networkx_graph):
    if from_networkx_graph:
        A = nx.to_numpy_array(G)
    else:
        A = G.toarray()

    n = A.shape[0]
    D = _degree_matrix(A)
    eps = 1 / (1 + np.max(D))

    return inv(np.eye(n) + (eps**2) * D - eps * A)


def deltacon(G1, G2, S1=None, S2=None, from_networkx_graphs=False):
    if from_networkx_graphs:
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)
    else:
        A1 = G1.toarray()
        A2 = G2.toarray()

    if S1 is None:
        S1 = _process_deltacon(G1, from_networkx_graph=from_networkx_graphs)
    if S2 is None:
        S2 = _process_deltacon(G2, from_networkx_graph=from_networkx_graphs)

    return np.sqrt(np.sum(np.square(np.sqrt(S1) - np.sqrt(S2))))


def _process_approx_deltacon(G, S0, from_networkx_graph):
    if from_networkx_graph:
        A = nx.to_numpy_array(G)
    else:
        A = G.toarray()

    n = A.shape[0]

    D = _degree_matrix(A)
    eps = 1 / (1 + np.max(D))

    S_before_inv = np.eye(n) + (eps**2) * D - eps * A
    S = solve(S_before_inv, S0)

    return S


def approx_deltacon(G1,
                    G2,
                    S1=None,
                    S2=None,
                    n_groups=10,
                    from_networkx_graphs=False):
    # prepare S0
    S0 = None
    if S1 is None or S2 is None:
        n = A1.shape[0]
        random_node_indices = np.random.permutation(n)
        groups = [random_node_indices[i::n_groups] for i in range(n_groups)]

        S0 = np.zeros((n, n_groups))
        for k, group in enumerate(groups):
            S0[group, k] = 1

    if S1 is None:
        S1 = _process_approx_deltacon(G1,
                                      S0=S0,
                                      from_networkx_graph=from_networkx_graphs)
    if S2 is None:
        S2 = _process_approx_deltacon(G2,
                                      S0=S0,
                                      from_networkx_graph=from_networkx_graphs)

    return np.sqrt(np.sum(np.square(np.sqrt(S1) - np.sqrt(S2))))


def frobenius(G1, G2, from_networkx_graphs=False):
    if from_networkx_graphs:
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)
    else:
        A1 = G1.toarray()
        A2 = G2.toarray()

    return np.sqrt(np.sum((A1 - A2)**2))


def jaccard_dist(G1, G2, from_networkx_graphs=False):
    if from_networkx_graphs:
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)
    else:
        A1 = G1.toarray()
        A2 = G2.toarray()
    cup = np.sum(A1 + A2 > 0)
    cap = np.sum(A1 * A2 > 0)

    return 1 - cap / cup


def _shared_neighbor_sim(G,
                         k=None,
                         ratio_to_k=True,
                         zero_diagonal=True,
                         normalize_by_max=False,
                         n_hops=1,
                         symmetrize=False,
                         from_networkx_graph=False):
    if from_networkx_graph:
        A = nx.to_numpy_array(G)
    else:
        A = G.toarray()

    if symmetrize:
        A = _to_undirected(A)

    A_nhop = A
    for i in range(n_hops - 1):
        A_nhop = A_nhop @ A

    S = A_nhop @ A_nhop.T

    # when applying normalize_by_max, this step is redundant
    if (not normalize_by_max) and ratio_to_k:
        if k is None:
            k = np.sum(A) / A.shape[0]  # mean degree
        S /= k**n_hops

    if zero_diagonal:
        np.fill_diagonal(S, 0)

    if normalize_by_max:
        S /= np.max(S)

    return S


def _snn_clust_extract(G, S, start_node_idx, n_walks):
    # G: adjacency matrix (sparse matrix)
    # S: SNN matrix (np.array)
    cluster_member = set()
    cluster_member.add(start_node_idx)
    current_queue = deque([start_node_idx])

    # Note: this does't exactly extract nodes=n_walks
    # (but for gain speeding up, this is acceptable)
    n_visits = 0
    n_trials = n_walks * 20  # to avoid infinite loop
    while n_visits < n_walks and n_trials > 0:
        i = current_queue.popleft()
        neigbors = G[i].nonzero()[1]

        probabilities = 1 - S[i, neigbors]
        dices = np.random.rand(len(neigbors))
        new_members = neigbors[np.where(dices > probabilities)[0]]
        # shuffle to make walk more actually random
        np.random.shuffle(new_members)

        current_queue += deque(new_members)
        for new_mem_idx in new_members:
            cluster_member.add(new_mem_idx)
        n_visits += len(new_members)
        n_trials -= 1

        if not current_queue:
            break

    return np.array(list(cluster_member))


def _extract_cluster(G, S, n_walks):
    # G: adjacency matrix (sparse matrix)
    # S: SNN matrix (np.array)
    n = G.shape[0]
    rand_node_idx = np.random.randint(n)

    not_found_cluster = True
    n_trials = n
    clust_members = []
    while not_found_cluster and n_trials > 0:
        clust_members = _snn_clust_extract(G, S, rand_node_idx, n_walks)
        if clust_members.size > 1:
            not_found_cluster = False
        n_trials -= 1  # to avoid infinite loop

    return clust_members


def _dbscan(D, indices):
    import hdbscan
    related_D = (D[indices].T)[indices]
    np.fill_diagonal(related_D, 0)

    clust = hdbscan.HDBSCAN(metric="precomputed",
                            allow_single_cluster=True).fit(related_D)

    return clust.labels_


def _snc_measure_iter(G_src, S_src, S_target, D_target, max_val, min_val,
                      n_walks, alpha):
    src_space_clust_indices = _extract_cluster(G_src, S_src, n_walks)

    # TODO: this doesn't make sense especially when using kmeans
    # if applying on kmeans on the cluster indices above,
    # always these indices are separated into k-clusters
    target_space_clust_ids = _dbscan(D_target, src_space_clust_indices)

    # construct cluster member matrix (n_clusts x n_indices)
    unq_clust_ids = np.unique(target_space_clust_ids)
    n_clusts = len(unq_clust_ids)
    n_indices = len(src_space_clust_indices)
    clust_member_mat = np.zeros((n_clusts, n_indices)).astype(bool)
    for i, clust_id in enumerate(unq_clust_ids):
        clust_member_mat[i, :] = target_space_clust_ids == clust_id

    # construct weight matrix
    n_members_for_each_clust = np.sum(clust_member_mat, axis=1)[:, np.newaxis]
    W = n_members_for_each_clust @ n_members_for_each_clust.T

    # construct distortion matrix
    M = np.zeros((n_clusts, n_clusts))
    for i in range(n_clusts):
        indices_i = src_space_clust_indices[clust_member_mat[i, :]]
        for j in range(i):
            indices_j = src_space_clust_indices[clust_member_mat[j, :]]
            sim_src = np.sum((S_src[indices_i].T)[indices_j]) / W[i, j]
            sim_target = np.sum((S_target[indices_i].T)[indices_j]) / W[i, j]
            dist_src = 1 / (sim_src + alpha)
            dist_target = 1 / (sim_target + alpha)
            M[i, j] = dist_target - dist_src

    distortions = (M[M > 0] - min_val) / (max_val - min_val)
    weights = W[M > 0]
    part_distort_sum = np.sum(distortions * weights)
    part_weight_sum = np.sum(weights)

    return part_distort_sum, part_weight_sum


def _snc_measure(G_src,
                 S_src,
                 S_target,
                 D_target,
                 max_val,
                 min_val,
                 n_iter,
                 n_walks,
                 alpha,
                 from_networkx_graph=False):
    if from_networkx_graph:
        G_src_ = csr_matrix(nx.to_numpy_array(G).tocsr())
    else:
        G_src_ = G_src

    distort_sum = 0
    weight_sum = 0
    for _ in range(n_iter):
        part_distort_sum, part_weight_sum = _snc_measure_iter(
            G_src=G_src_,
            S_src=S_src,
            S_target=S_target,
            D_target=D_target,
            max_val=max_val,
            min_val=min_val,
            n_walks=n_walks,
            alpha=alpha)
        distort_sum += part_distort_sum
        weight_sum += part_weight_sum

    # When graphs are (close to) the same, distortion/weight_sum will be 0
    score = 1 if weight_sum == 0 else 1 - distort_sum / weight_sum

    return score


### Matrix computation version of SnC implementation
def snc(G1,
        G2,
        S1=None,
        S2=None,
        D1=None,
        D2=None,
        n_iter=None,
        alpha=0.1,
        walk_ratio=0.4,
        from_networkx_graphs=False):

    n = G1.shape[0]

    # Make this faster to allow S1, S2 precomputed
    # compute snn matrix
    if S1 is None:
        S1 = _shared_neighbor_sim(G1,
                                  normalize_by_max=True,
                                  from_networkx_graph=from_networkx_graphs)
    if S2 is None:
        S2 = _shared_neighbor_sim(G2,
                                  normalize_by_max=True,
                                  from_networkx_graph=from_networkx_graphs)

    # convert snn matrix to dist matrix
    if D1 is None:
        D1 = 1 / (S1 + alpha)
    if D2 is None:
        D2 = 1 / (S2 + alpha)

    D_diff = D2 - D1  # when > 0, strecth. when < 0 compress

    max_D_diff = np.max(D_diff)
    min_D_diff = np.min(D_diff)
    max_stretch = max(0, max_D_diff)
    min_stretch = max(0, min_D_diff)
    max_compress = max(0, -min_D_diff)
    min_compress = max(0, -max_D_diff)

    n_iter = n_iter if n_iter else max(int(n / 50), 20)
    n_walks = int(n * walk_ratio)

    # steadiness
    steadiness = _snc_measure(G_src=G2,
                              S_src=S2,
                              S_target=S1,
                              D_target=D1,
                              max_val=max_compress,
                              min_val=min_compress,
                              n_iter=n_iter,
                              n_walks=n_walks,
                              alpha=alpha,
                              from_networkx_graph=from_networkx_graphs)
    # cohesiveness
    cohesiveness = _snc_measure(G_src=G1,
                                S_src=S1,
                                S_target=S2,
                                D_target=D2,
                                max_val=max_stretch,
                                min_val=min_stretch,
                                n_iter=n_iter,
                                n_walks=n_walks,
                                alpha=alpha,
                                from_networkx_graph=from_networkx_graphs)

    return steadiness, cohesiveness


def snc_dissim(G1,
               G2,
               S1=None,
               S2=None,
               D1=None,
               D2=None,
               n_iter=None,
               alpha=0.1,
               walk_ratio=0.4,
               snc_score_aggregation=np.min,
               from_networkx_graphs=False):
    # NOTE: SnC is  unstable because of its random walk
    # also, DBSCAN is only applied to a subset of noded (i.e., visited nodes)
    # (should find a better dissimilarity measure)
    s, c = snc(G1,
               G2,
               S1=S1,
               S2=S2,
               D1=D1,
               D2=D2,
               n_iter=n_iter,
               alpha=alpha,
               walk_ratio=walk_ratio,
               from_networkx_graphs=from_networkx_graphs)

    return 1 - snc_score_aggregation([s, c])


def snn_dissim(G1,
               G2,
               S1=None,
               S2=None,
               method='default',
               fixed_degree=None,
               n_hops=1,
               symmetrize=True,
               from_networkx_graphs=False):
    if S1 is None:
        S1 = _shared_neighbor_sim(G1,
                                  k=fixed_degree,
                                  n_hops=n_hops,
                                  symmetrize=symmetrize,
                                  from_networkx_graph=from_networkx_graphs)
    if S2 is None:
        S2 = _shared_neighbor_sim(G2,
                                  k=fixed_degree,
                                  n_hops=n_hops,
                                  symmetrize=symmetrize,
                                  from_networkx_graph=from_networkx_graphs)

    dissim = None
    if method == 'l2':
        dissim = np.sqrt(np.sum((S1 - S2)**2))
    elif method == 'default':
        D = S1 - S2
        D_plus = D[D > 0]
        D_minus = D[D < 0]
        dissim_plus = np.sqrt(np.sum(D_plus**2))
        dissim_minus = np.sqrt(np.sum(D_minus**2))
        dissim = max(dissim_plus, dissim_minus)

    return dissim


def nsd(G1,
        G2,
        beta=1,
        fixed_degree=None,
        S1=None,
        S2=None,
        sig1=None,
        sig2=None,
        snn_nhops=1,
        snn_symmetrize=False,
        lsd_normalization=None,
        lsd_timescales=np.logspace(-2, 2, 256),
        lsd_n_eigvals='auto',
        take_log_for_lsd=True,
        from_networkx_graphs=False):
    '''
    Neighbor & Shape Dissimilaity (NSD):
    '''
    snn_dissim_val = snn_dissim(G1,
                                G2,
                                S1=S1,
                                S2=S2,
                                method='default',
                                fixed_degree=fixed_degree,
                                n_hops=snn_nhops,
                                symmetrize=snn_symmetrize,
                                from_networkx_graphs=from_networkx_graphs)
    netlsd_dissim_val = netlsd(G1,
                               G2,
                               sig1=sig1,
                               sig2=sig2,
                               normalization=lsd_normalization,
                               timescales=lsd_timescales,
                               n_eigvals=lsd_n_eigvals,
                               from_networkx_graphs=from_networkx_graphs)

    # taking log because netlsd is following exponential difference
    if take_log_for_lsd:
        netlsd_dissim_val = np.log(1 + netlsd_dissim_val)
    nsd = snn_dissim_val**beta * netlsd_dissim_val

    return nsd


####
# for netlsd, communicability, and deltacon, frobenius, and jaccard_dist, faster versions are implemented above
####

####
# Undirected, Unweighted
####
# communicability_jsd = nd.CommunicabilityJSD().dist
dk_series = lambda G1, G2: nd.dkSeries().dist(G1, G2, d=2)
# netlsd = lambda G1, G2: nd.NetLSD().dist(
#     G1, G2, normalization='complete', timescales=None)
# normalization is not needed for comparison of the same size networks
# netlsd = lambda G1, G2: _netlsd(G1, G2, normalization=None)
onion_divergence = lambda G1, G2: nd.OnionDivergence().dist(
    G1, G2, dist='lccm')
quantum_jsd = lambda G1, G2: nd.QuantumJSD().dist(G1, G2, beta=0.1, q=None)

####
# Undirected
####
degree_divergence = nd.DegreeDivergence().dist
d_measure = lambda G1, G2: nd.DMeasure().dist(
    G1, G2, w1=0.45, w2=0.45, w3=0.1, niter=50)
# d-measure doesn't work for nonconnected graph
graph_diffusion = lambda G1, G2: nd.GraphDiffusion().dist(
    G1, G2, thresh=1e-08, resolution=1000)
netsimile = nd.NetSimile().dist
resistance_perturbation = lambda G1, G2: nd.ResistancePerturbation().dist(
    G1, G2, p=2)
# resistance_perturbation doesn't work for nonconnected graph

####
# Unweighted
####
distributional_nbd = lambda G1, G2: nd.DistributionalNBD().dist(
    G1,
    G2,
    sparse=False,
    shave=True,
    keep_evals=True,
    k=None,
    vector_distance='euclidean')
hamming = nd.Hamming().dist
hamming_ipsen_mikhailov = lambda G1, G2: nd.HammingIpsenMikhailov().dist(
    G1, G2, combination_factor=1)
ipsen_mikhailov = lambda G1, G2: nd.IpsenMikhailov().dist(G1, G2, hwhm=0.08)
# jaccard_distance = nd.JaccardDistance().dist
laplacian_spectral = lambda G1, G2: nd.LaplacianSpectral().dist(
    G1,
    G2,
    normed=True,
    kernel='normal',
    hwhm=0.011775,
    measure='jensen-shannon',
    k=None,
    which='LM')
non_backtracking_spectral = lambda G1, G2: nd.NonBacktrackingSpectral().dist(
    G1, G2, topk='automatic', ignore_negative_evals=True, batch=50, tol=1e-03)
polynomial_dissimilarity = lambda G1, G2: nd.PolynomialDissimilarity().dist(
    G1, G2, k=5, alpha=1)

####
# Any
####
# deltacon = lambda G1, G2: nd.DeltaCon().dist(G1, G2, exact=True, g=None)
# deltacon's exact=False is not supported by the library yet
# frobenius = nd.Frobenius().dist
portrait_divergence = lambda G1, G2: nd.PortraitDivergence().dist(
    G1, G2, bins=5, binedges=None)
# portrait_divergence is slow. Probably need to finda good bin size (in pecentile)


def symmetric_hausdorff(G1,
                        G2,
                        normalization=False,
                        from_networkx_graphs=False):
    A1 = nx.to_numpy_array(G1) if from_networkx_graphs else G1.toarray()
    A2 = nx.to_numpy_array(G2) if from_networkx_graphs else G2.toarray()

    if normalization:
        A1 = A1 / np.max(A1)
        A2 = A2 / np.max(A2)

    return max(directed_hausdorff(A1, A2)[0], directed_hausdorff(A1, A2)[0])

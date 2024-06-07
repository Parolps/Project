from scipy.linalg import svd
import numpy as np

#### Nearest Neighbors utils ####


def jaccard_similarity(id1, id2, mol_bits, mol_ids):
    set1 = set(mol_bits[mol_ids[id1]])
    set2 = set(mol_bits[mol_ids[id2]])
    return len(set1 & set2) / len(set1 | set2)


def get_all_neighbors(id, buckets, mol_bits, mol_ids, test_ids=set()):
    neighbors = set()
    for (b, buck), mols in buckets.items():
        if id in mols:
            neighbors.update(mols)
    neighbors.remove(id)
    neighbors.difference_update(test_ids)
    return neighbors


def k_neighbors(mol_idx, buckets, mol_bits, mol_ids, k=5, test_ids=set()):
    neighbors = get_all_neighbors(
        mol_idx, buckets, mol_bits, mol_ids, test_ids=test_ids
    )

    neighbor_similarities = [
        (neighbor, jaccard_similarity(mol_idx, neighbor, mol_bits, mol_ids))
        for neighbor in neighbors
    ]

    sorted_neighbors = sorted(neighbor_similarities, key=lambda x: x[1], reverse=True)

    return sorted_neighbors[:k]


#### Latent Factors utils ####


def get_factors(U, S, V, k=10):
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]
    P = U_k.copy()
    Q = S_k @ V_k
    return P, Q


def masked_MSE(R, R_hat):
    mask = (R > 0) * 1
    return np.sum((mask * (R - R_hat)) ** 2) / np.sum(mask)


def Reg_Epoch(positions, R, P, Q, seed=42, LR=0.0001, Lp=1, Lq=1):
    np.random.seed(seed)
    np.random.shuffle(positions)
    for pos in positions:
        R_ui = R[pos[0], pos[1]]
        delta = 2 * (R_ui - P[pos[0], :] @ Q[:, pos[1]])
        Q[:, pos[1]] += LR * (delta * P[pos[0], :] - Lq * Q[:, pos[1]])
        P[pos[0], :] += LR * (delta * Q[:, pos[1]] - Lp * P[pos[0], :])
    return P, Q


def Reg_SGD(R, epochs=10, k=10, seed=42, LR=0.001, Lp=1, Lq=1):
    U, S, V = svd(R, full_matrices=False)
    P, Q = get_factors(U, S, V, k=k)
    locations = np.argwhere(R > 0)
    mses = []
    for _ in range(epochs):
        P, Q = Reg_Epoch(locations, R, P, Q, seed=seed, LR=LR, Lp=Lp, Lq=Lq)
        mses.append(masked_MSE(R, P @ Q))
    return P, Q, mses

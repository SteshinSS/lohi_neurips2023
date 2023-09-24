from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import numpy as np
from numpy.random import default_rng
import pandas as pd


def get_nearest_mols(lhs: list[str], rhs: list[str], return_idx=False) -> list[float]:
    """
    Gets two lists of smiles. Returns list of the nearest distances from lhs to the rhs molecules.
    """
    lhs_mols = []
    for smiles in lhs:
        lhs_mols.append(Chem.MolFromSmiles(smiles))
    lhs_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in lhs_mols]
    
    rhs_mols = []
    for smiles in rhs:
        rhs_mols.append(Chem.MolFromSmiles(smiles))
    rhs_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in rhs_mols]

    nearest_dist = []
    nearest_idx = []
    for lhs in lhs_fps:
        sims = DataStructs.BulkTanimotoSimilarity(lhs, rhs_fps)
        nearest_dist.append(max(sims))
        nearest_idx.append(np.argmax(sims))
    if return_idx:
        result = (nearest_dist, nearest_idx)
    else:
        result = nearest_dist
    return result


def butina_split(smiles: list[str], cutoff: float, seed: int, frac_train=0.8):
    """
    Select distinct molecules to train/test. Returns indices of the molecules in the smiles list.
    Adapted from DeepChem (https://deepchem.io/), but random seed is added.
    """

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    scaffold_sets = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    rng = default_rng(seed)
    rng.shuffle(scaffold_sets)

    train_cutoff = frac_train * len(smiles)
    train_inds = []
    test_inds = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            test_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, test_inds


def get_hi_split(data: pd.DataFrame, threshold: float, seed: float, cutoff: float = 0.5):
    """
    Takes dataset and split it for the Hit Identification task.
    """
    train_idx, test_idx = butina_split(data['smiles'], cutoff, seed)
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    
    sim = get_nearest_mols(test['smiles'], train['smiles'])
    diverse_test_idx = []
    for i in range(len(sim)):
        if sim[i] < threshold:
            diverse_test_idx.append(test_idx[i])
    valid = data.iloc[diverse_test_idx]
    sim = get_nearest_mols(valid['smiles'], train['smiles'])
    return train, valid, sim


def select_distinct_clusters(smiles: list[str], threshold: float, min_cluster_size: int, max_clusters: int, values: list[int], std_threshold: float):
    """
    A greedy algorithm to select independed clusters from datasets. A part of the Lo splitter.
    """

    clusters = []

    while len(clusters) < max_clusters:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        all_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
        total_neighbours = []
        stds = []

        for fps in all_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fps, all_fps)
            neighbors_idx = np.array(sims) > threshold
            total_neighbours.append(neighbors_idx.sum())
            stds.append(values[neighbors_idx].std())
        
        total_neighbours = np.array(total_neighbours)
        stds = np.array(stds)

        # Find the most distant cluster
        central_idx = None
        least_neighbours = max(total_neighbours)
        for idx, n_neighbours in enumerate(total_neighbours):
            if n_neighbours > min_cluster_size:
                if n_neighbours < least_neighbours:
                    if stds[idx] > std_threshold:
                        least_neighbours = n_neighbours
                        central_idx = idx

        if central_idx is None:
            break # there are no clusters 
        
        sims = DataStructs.BulkTanimotoSimilarity(all_fps[central_idx], all_fps)
        is_neighbour = np.array(sims) > threshold

        # Add them into cluster
        cluster_smiles = []
        for idx, value in enumerate(is_neighbour):
            if value:
                if idx != central_idx:  # we add the central molecule at the end of the list
                    cluster_smiles.append(smiles[idx])
        cluster_smiles.append(smiles[central_idx])
        clusters.append(cluster_smiles)

        # Remove neighbours of neighbours from the rest of smiles
        nearest_dist = get_nearest_mols(smiles, cluster_smiles)
        rest_idx = []
        for idx, dist in enumerate(nearest_dist):
            if dist < threshold:
                rest_idx.append(idx)
        smiles = smiles[rest_idx]
        values = values[rest_idx]
    
    return clusters, smiles


def get_lo_split(smiles: list[str], threshold: float, min_cluster_size: int, max_clusters: int, values: list[float], std_threshold: float):
    cluster_smiles, train_smiles = select_distinct_clusters(smiles, threshold, min_cluster_size, max_clusters, values, std_threshold)
    train_smiles = list(train_smiles)
    # Move one molecule from each test cluster to the train
    leave_one_clusters = []
    for cluster in cluster_smiles:
        train_smiles.append(cluster[-1])
        leave_one_clusters.append(cluster[:-1])

    return leave_one_clusters, train_smiles


def set_cluster_columns(data: pd.DataFrame, cluster_smiles: list[list[str]], train_smiles: list[str]):
    data = data.copy()
    data['cluster'] = -1
    is_train = data['smiles'].isin(train_smiles)
    data.loc[is_train, ['cluster']] = 0

    for i, cluster in enumerate(cluster_smiles):
        is_cluster = data['smiles'].isin(cluster)
        data.loc[is_cluster, ['cluster']] = i + 1
    
    is_in_cluster = data['cluster'] != -1
    return data[is_in_cluster]
from collections import defaultdict
import logging
import random
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np
# from numba import jit
from time import time
import pdb

from .data import MoleculeDataset
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem
from rdkit import DataStructs


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

from concurrent.futures import ProcessPoolExecutor, as_completed

def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """

    n_mols = len(mols)
    print ("Computing scaffolds for", n_mols, "molecule(s)")
    scaffolds = defaultdict(set)

    # n_proc = 5
    # print ("Using", n_proc, "process(es)")

    # with tqdm(total=n_mols) as pbar:

    #     with ProcessPoolExecutor(max_workers=n_proc) as p:

    #         tasks = {}

    #         for i, mol in enumerate(mols):

    #             task = p.submit(
    #                 generate_scaffold,
    #                 mol,
    #             )
    #             tasks[task] = (i, mol)

    #         for task in as_completed(tasks):
                
    #             i, mol = tasks[task]
    #             scaffold = task.result()
    #             del tasks[task]

    #             if use_indices:
    #                 scaffolds[scaffold].add(i)
    #             else:
    #                 scaffolds[scaffold].add(mol)
                
    #             pbar.update(1)

    for i, mol in tqdm(enumerate(mols), total=n_mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0,
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index (set) in the data
    scaffold_to_indices = scaffold_to_smiles(data.mols(), use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  
        # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            # train += index_set
            train_scaffold_count += 1
            set_to_add_index_set = train
     
        elif len(val) + len(index_set) <= val_size:
            # val += index_set
            val_scaffold_count += 1
            set_to_add_index_set = val

        else:
            # test += index_set
            test_scaffold_count += 1
            set_to_add_index_set = test

        for i in index_set:
            # set_to_add_index_set.append(i)
            set_to_add_index_set.append(data[i])
            data[i] = None
       
        # add set as element
        # set_to_add_index_set.append(index_set)

    # if logger is not None:
    #     logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
    #                  f'train scaffolds = {train_scaffold_count:,} | '
    #                  f'val scaffolds = {val_scaffold_count:,} | '
    #                  f'test scaffolds = {test_scaffold_count:,}')
    
    # log_scaffold_stats(data, index_sets, logger=logger)
    
    # raise Exception

    # Map from indices to data
    # print ("Mapping indexes back to data")
    # train = [data[i] for i in train]
    # val = [data[i] for i in val]
    # test = [data[i] for i in test]

    # train = [data[j] for i in train for j in i]
    # val = [data[j] for i in val for j in i]
    # test = [data[j] for i in test for j in i]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A Logger.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first num_labels labels and a list of the number of non-zero values for
    the first num_scaffolds scaffolds, sorted in decreasing order of scaffold frequency.
    """
    # print some statistics about scaffolds
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=np.float)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_scaffolds, len(target_avgs)))]

    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels: {stats}')

    return stats


def cluster_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0,
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    """
    Split a dataset by cluster so that no molecules in a same cluster are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of clusters in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_cluster_count, val_cluster_count, test_cluster_count = 0, 0, 0

    # Map from cluster to index in the data
    fingerprints = [AllChem.GetMorganFingerprint(m,2) for m in data.mols()]
    dists = []
    nfps = len(fingerprints)
    for i in tqdm(range(nfps)):
        if i == 0:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        dists.extend([(1 - x) for x in sims])
    cluster_to_indices = Butina.ClusterData(dists, nfps, 0.4, isDistData=True)
    cluster_to_indices_list = [list(x) for x in cluster_to_indices]

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = cluster_to_indices_list
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest cluster sets
        index_sets = sorted(list(cluster_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_cluster_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_cluster_count += 1
        else:
            test += index_set
            test_cluster_count += 1

    if logger is not None:
        logger.debug(f'Total clusters = {len(cluster_to_indices):,} | '
                     f'train clusters = {train_cluster_count:,} | '
                     f'val clusters = {val_cluster_count:,} | '
                     f'test clusters = {test_cluster_count:,}')
        
    log_cluster_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def log_cluster_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_clusters: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular clusters.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_clusters: The number of clusters about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A Logger.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first num_labels labels and a list of the number of non-zero values for
    the first num_clusters clusters, sorted in decreasing order of cluster frequency.
    """
    # print some statistics about clusters
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=np.float)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_clusters, len(target_avgs)))]

    if logger is not None:
        logger.debug('Label averages per cluster, in decreasing order of cluster frequency,'
                     f'capped at {num_clusters} clusters and {num_labels} labels: {stats}')

    return stats
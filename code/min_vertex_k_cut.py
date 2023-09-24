import networkx as nx
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import mip


def get_neighborhood_graph(smiles, threshold):
    """
    Builds a neighborhood graph from smiles list. 
    Neighborhood graph is a graph, which nodes corresponds to the smiles, 
    and two nodes are connected iff the corresponding smiles have Tanimoto similarity > threshold.  
    """
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    similarity_matrix = []
    for fp in fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
        similarity_matrix.append(sims)

    G = nx.Graph()
    for i in range(len(mols)):
        for j in range(i, len(mols)):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                G.add_edge(i, j, weight=similarity)
    return G

def get_main_component(G):
    """
    Return the main component of the graph and list of small components.
    """
    components = [x for x in nx.connected_components(G)]
    biggest_component_idx = np.argmax([len(x) for x in components])
    biggest_component = components[biggest_component_idx]
    S = G.subgraph(biggest_component).copy()

    small_components = []
    for i, component in enumerate(components):
        if i == biggest_component_idx:
            continue
        small_components.append(component)
    return S, small_components


def coarse_graph(graph, threshold):
    """
    Cluster some nodes with large number of neighbors and return condensed graph.
    """
    # For each node, calculate number of neighbors with more than threshold similarity
    n_neighbors = []
    for node in graph.nodes():
        edges = graph.edges(node, data=True)

        total_neighbors = 0
        for edge in edges:
            if edge[2]['weight'] > threshold:
                total_neighbors += 1
        n_neighbors.append((total_neighbors, len(n_neighbors)))
    
    # Starting with the nodes with the biggerst number of neighbors, cluster them
    n_neighbors = sorted(n_neighbors, key=lambda x: -x[0])
    node_to_cluster = [-1] * len(graph)
    current_cluster = 0
    for _, node in n_neighbors:
        # Skip this node if it is already in a cluster
        if node_to_cluster[node] != -1:
            continue

        # Assign this node and not-assigned neighbors to a new cluster
        node_to_cluster[node] = current_cluster
        edges = graph.edges(node, data=True)
        for edge in edges:
            if edge[2]['weight'] > threshold:
                adjacent_node = edge[1]
                if node_to_cluster[adjacent_node] == -1:
                    node_to_cluster[adjacent_node] = current_cluster
        current_cluster += 1
    node_to_cluster = np.array(node_to_cluster)
    
    # For cluster i, cluster_size[i] = size of the i'th cluster
    cluster_size = np.unique(node_to_cluster, return_counts=True)[1]

    # Build new coarsed graph
    result = nx.Graph()

    # Add nodes
    for cluster in range(current_cluster):
        result.add_node(cluster, weight=cluster_size[cluster])
    
    # Add edges
    for cluster in range(current_cluster):
        # Find to which clusters this cluster is connected to
        connected_clusters = set()
        this_cluster_indices = np.where(node_to_cluster == cluster)[0]
        for node in this_cluster_indices:
            edges = graph.edges(node, data=True)
            for edge in edges:
                connected_clusters.add(node_to_cluster[edge[1]])
        for connected_cluster in connected_clusters:
            result.add_edge(cluster, connected_cluster)
    return result, node_to_cluster

def train_test_split_connected_graph(S, train_min_fraq, test_min_fraq, max_mip_gap=0.1):
    k = 2
    total_weight = 0
    for node in S.nodes():
        total_weight += S.nodes[node]['weight']
    print('Total molecules:', total_weight)
        
    min_train_size = total_weight * train_min_fraq
    print("Min train size", int(min_train_size))
    min_test_size = total_weight * test_min_fraq
    print("Min test size", int(min_test_size))

    m = mip.Model(sense=mip.MAXIMIZE)

    x = []
    for i in range(len(S)):
        per_node_x = []
        for j in range(k):
            per_node_x.append(m.add_var(var_type=mip.BINARY))
        x.append(per_node_x)

    w = []
    for i in range(len(S)):
        node = S.nodes[i]
        per_node_w = []
        for k_i in range(k):
            per_node_w.append(node['weight'])
        w.append(per_node_w)
    
    objective = []
    for i in range(len(x)):
        for k_i in range(k):
            objective.append(w[i][k_i] * x[i][k_i]) 
    m.objective = mip.xsum(objective)

    # Each node in one particion only
    for x_k in x:
        m += mip.xsum(x_k) <= 1

    # No edges between partitions
    for edge in S.edges:
        i, j = edge
        for k_1 in range(k):
            for k_2 in range(k):
                if k_1 == k_2:
                    continue
                m += x[i][k_1] + x[j][k_2] <= 1

    # Partitions are balanced
    train_weight = []
    test_weight = []
    for i in range(len(x)):
        train_weight.append(w[i][0] * x[i][0])
        test_weight.append(w[i][1] * x[i][1])
    m += mip.xsum(train_weight) >= min_train_size
    m += mip.xsum(test_weight) >= min_test_size

    m.max_mip_gap = max_mip_gap
    m.threads = -1
    m.emphasis = 2
    m.optimize()
    return m

def process_bisect_results(model, coarsed_S, S, node_to_cluster):
    result = np.array([a.x for a in model.vars])

    coarsed_S_split = [-1] * len(coarsed_S)
    for i in range(len(coarsed_S)):
        if result[i*2] > 0:
            coarsed_S_split[i] = 0
        if result[i*2 + 1] > 0:
            coarsed_S_split[i] = 1

    S_split = []
    for node in S.nodes():
        coarsed_id = node_to_cluster[node]
        S_split.append(coarsed_S_split[coarsed_id])

    in_lost, in_train, in_test = np.unique(S_split, return_counts=True)[1]
    print('Molecules in train:', in_train)
    print('Molecules in test:', in_test)
    print('Molecules lost:', in_lost)
    return S_split

def process_trisect_results(model, coarsed_S, S, node_to_cluster):
    result = np.array([a.x for a in model.vars])

    coarsed_S_split = [-1] * len(coarsed_S)
    for i in range(len(coarsed_S)):
        if result[i*3] > 0:
            coarsed_S_split[i] = 0
        if result[i*3 + 1] > 0:
            coarsed_S_split[i] = 1
        if result[i*3 + 2] > 0:
            coarsed_S_split[i] = 2

    S_split = []
    for node in S.nodes():
        coarsed_id = node_to_cluster[node]
        S_split.append(coarsed_S_split[coarsed_id])

    print(np.unique(S_split, return_counts=True)[1])
    return S_split


def trisect_connected_graph(S, part_min_frac, emphasis=2, max_mip_gap=0.1):
    k = 3
    total_weight = 0
    for node in S.nodes():
        total_weight += S.nodes[node]['weight']
    print('Total molecules:', total_weight)
        
    lower_bound = int(total_weight * part_min_frac)
    print('Min size of a partition:', lower_bound)

    m = mip.Model(sense=mip.MAXIMIZE)

    x = []
    for i in range(len(S)):
        per_node_x = []
        for j in range(k):
            per_node_x.append(m.add_var(var_type=mip.BINARY))
        x.append(per_node_x)

    w = []
    for i in range(len(S)):
        node = S.nodes[i]
        per_node_w = []
        for k_i in range(k):
            per_node_w.append(node['weight'])
        w.append(per_node_w)
    
    objective = []
    for i in range(len(x)):
        for k_i in range(k):
            objective.append(w[i][k_i] * x[i][k_i]) 
    m.objective = mip.xsum(objective)

    # Each node in one particion only
    for x_k in x:
        m += mip.xsum(x_k) <= 1

    # No edges between partitions
    for edge in S.edges:
        i, j = edge
        for k_1 in range(k):
            for k_2 in range(k):
                if k_1 == k_2:
                    continue
                m += x[i][k_1] + x[j][k_2] <= 1

    # Partitions are balanced
    for k_i in range(k):
        partition_weight = []
        for i in range(len(x)):
            partition_weight.append(w[i][k_i] * x[i][k_i]) 
        m += mip.xsum(partition_weight) >= lower_bound


    m.max_mip_gap = max_mip_gap
    m.threads = -1
    m.emphasis = emphasis
    m.optimize()
    return m



def test_split(train_smiles, test_smiles, threshold):
    train_mols = [Chem.MolFromSmiles(smile) for smile in train_smiles]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_mols]

    test_mols = [Chem.MolFromSmiles(smile) for smile in test_smiles]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in test_mols]

    for train_fp in train_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(train_fp, test_fps))
        is_bad = sims > threshold
        assert is_bad.sum() == 0

    

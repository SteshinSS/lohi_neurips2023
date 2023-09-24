from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import pandas as pd
import numpy as np


def calculate_circles_quick(smiles: list[str], threshold: float = 0.5) -> int:
    """
    Takes list of smiles and calculates molecular diversity measure "#Circles".

    Adapted from "How Much Space Has Been Explored? Measuring the Chemical 
    Space Covered by Databases and Machine-Generated Molecules" by Xie et al. 2023
    See Algorithm 3 in the appendix H.
    """
    circle_smiles = [smiles[0]]
    circle_mols = [Chem.MolFromSmiles(smiles[0])]
    circle_fps = [AllChem.GetMorganFingerprintAsBitVect(circle_mols[0], 2, 1024)]

    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        sims = DataStructs.BulkTanimotoSimilarity(fps, circle_fps)
        if max(sims) < threshold:
            circle_smiles.append(smile)
            circle_mols.append(mol)
            circle_fps.append(fps)
    return len(circle_smiles)


def binarize_log_data(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Takes chembl slice and binarize log activity values.
    Note: It is for log values, but relations in the chembl are set for normal values.
    Be sure you understand it before use.
    """
    final_smiles = []
    label = []
    for i in range(len(data)):
        row = data.iloc[i, :]
        if row["standard_relation"] == "=":
            final_smiles.append(row["canonical_smiles"])
            label.append(row["standard_value"] > threshold)
        if row["standard_relation"] == ">":
            if row["standard_value"] < threshold:
                final_smiles.append(row["canonical_smiles"])
                label.append(False)
        if row["standard_relation"] == "<":
            if row["standard_value"] > threshold:
                final_smiles.append(row["canonical_smiles"])
                label.append(True)
    result = pd.DataFrame({"smiles": final_smiles, "label": label})
    return result


def remove_ambiguous_row(data):
    """
    Takes pd.DataFrame with binary activity values 'label' and duplicated smiles.
    Takes median label for each smile and remove smiles with values that not 0 or 1 (ambiguous).
    """
    data_group = data.groupby(["smiles"]).median()
    labels = data_group["label"]
    is_almost_one = np.isclose(labels, 1.0)
    is_almost_zero = np.isclose(labels, 0.0)
    is_good = is_almost_one | is_almost_zero
    return data_group[is_good]


def clean_continuous(data):
    """
    Takes a slice of Chembl and select rows with:
    - Exact activity values
    - Values in range of [5, 9]
    - For duplicated smiles difference in value < 1.0

    Then groups by smiles and calculates median activity value
    """
    continuous = data[data["standard_relation"] == "="]
    is_big_enough = continuous["standard_value"] > 5
    is_small_enough = continuous["standard_value"] < 9
    continuous = continuous[is_big_enough & is_small_enough]

    group = continuous.groupby(["canonical_smiles"])
    min_values = group.min()
    max_values = group.max()
    difference = max_values["standard_value"] - min_values["standard_value"]
    is_difference_ok = difference < 1.0

    continuous_clean = group.median()
    continuous_clean = continuous_clean[is_difference_ok]
    return pd.DataFrame(
        {
            "canonical_smiles": continuous_clean.index.to_list(),
            "standard_value": continuous_clean["standard_value"],
        }
    ).reset_index(drop=True)


def chemprop_prepare_df(original_data):
    result = pd.DataFrame({
        'smiles': original_data['smiles'],
        'targets': original_data['value'].astype(float)
    })
    return result

def chemprop_process_folder(input_path, output_path):
    files = ['train_1.csv', 'train_2.csv', 'train_3.csv', 'test_1.csv', 'test_2.csv', 'test_3.csv']
    for file in files:
        input_data = pd.read_csv(input_path + file)
        output_data = chemprop_prepare_df(input_data)
        output_data.to_csv(output_path + file, index=False)
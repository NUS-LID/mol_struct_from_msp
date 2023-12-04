import os
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 


def read_jdx_file(file_path):
    jdx_file = open(file_path, "r")
    j_temp = jdx_file.read()
    jdx_file.close()
    return j_temp


def get_permuted_adjacency_matrix(mol):
    Chem.Kekulize(mol)
    bonds = mol.GetBonds()
    list_bonds = list()

    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    for bond in bonds:
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_type = int(bond.GetBondTypeAsDouble())

        # Populate adjacency matrix
        adjacency_matrix[begin_atom_idx, end_atom_idx] = bond_type
        adjacency_matrix[end_atom_idx, begin_atom_idx] = bond_type

        # Find index of atoms of each bond
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        # print(f"Bond between atoms {begin_atom.GetSymbol()}({begin_atom_idx}) and {end_atom.GetSymbol()}({end_atom_idx}), Bond type: {bond_type}")
        # list_bonds.append((begin_atom_idx, end_atom_idx, bond_type, begin_atom, end_atom))

    # Get the atom types and sort them
    atom_types = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    atom_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    sorted_indices = np.argsort(atom_nums)

    # Permute the adjacency matrix based on the sorted atom indices
    permuted_matrix = adjacency_matrix[sorted_indices][:, sorted_indices]

    return list_bonds, permuted_matrix, atom_types[sorted_indices]


def check_ionized_or_not(mol):
    # Check if the molecule has any non-zero charges
    ionized = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
    return ionized


def get_properties_from_symbol(atom_symbol):
    atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol)
    atomic_weight = Chem.GetPeriodicTable().GetAtomicWeight(atom_symbol)
    valency = Chem.GetPeriodicTable().GetDefaultValence(atom_symbol)
    num_outer_elec = Chem.GetPeriodicTable().GetNOuterElecs(atom_symbol)
    return atomic_number, atomic_weight, valency, num_outer_elec


def load_data():
    path = "../../nist_database/"
    mol_files = [file for file in os.listdir(path+"mol/") if file.endswith(".mol")]
    jdx_files = [file for file in os.listdir(path+"jdx/") if file.endswith(".jdx")]
    file_names = [file.split(".mol")[0] for file in mol_files]
    # print(len(mol_files), len(jdx_files))

    data_dicts = [{"file_names": file_name,
                   "mol": "",
                   "smiles": "",
                   "smiles": "",
                   "msp_seq": []} for file_name in file_names]
    for i in range(len(file_names)):
        data_dicts[i]["mol"] = Chem.MolFromMolFile(path+"mol/"+file_names[i] + ".mol")

    failed_index = list()
    for i in range(len(data_dicts)):
        try: 
            data_dicts[i]["smiles"] = Chem.MolToSmiles(data_dicts[i]["mol"])
        except:
            failed_index.append(i)

    print("Number of failed indices", len(failed_index))

    j_temp = read_jdx_file(path+"jdx/"+jdx_files[0])
    j_temp = j_temp.split("##XYDATA=(XY..XY) 1 \n")[-1].splitlines()
    j_temp = np.array([[int(y) for y in x.split()] for x in j_temp])
    j_disp = np.zeros(1600, dtype=int)
    for x in j_temp:
        j_disp[x[0]] = x[1]

    for i in range(len(data_dicts)):
        j_temp = read_jdx_file(path+"jdx/"+jdx_files[i])
        j_temp = j_temp.split("##XYDATA=(XY..XY) 1 \n")[-1].splitlines()
        j_temp = np.array([[int(y) for y in x.split()] for x in j_temp])
        data_dicts[i]["msp_seq"] = np.zeros(1600)
        for x in j_temp:
            data_dicts[i]["msp_seq"][x[0]] = x[1]

    failed_atomic_count_index = list()
    for i in range(len(data_dicts)):
        atomic_count = dict()
        try:
            for atom in data_dicts[i]["mol"].GetAtoms():
                try:
                    atomic_count[atom.GetAtomicNum()] += 1
                except:
                    atomic_count[atom.GetAtomicNum()] = 1
        except:
            failed_atomic_count_index.append(i)
        data_dicts[i]["atomic_count"] = atomic_count

    data_df = pd.DataFrame(data_dicts)
    data_df = data_df.drop(index=failed_index)
    data_df = data_df.reset_index()

    # filter molecules in dataset
    # filtered_molecules_idx = []
    # max_atoms = 13
    # # Iterate through the list of input molecules
    # for i in range(len(data_df)):
    #     mol = data_df["mol"][i]
        
    #     # Check if the molecule contains only carbon and oxygen atoms
    #     if mol is not None and mol.GetNumAtoms() <= max_atoms and all(atom.GetSymbol() in ('C', 'O') for atom in mol.GetAtoms()):
    #         num_carbon_atoms = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
    #         if num_carbon_atoms/mol.GetNumAtoms() < 0.8:
    #             filtered_molecules_idx.append(i)
    # print("Filtered molecules:", len(filtered_molecules_idx))
    # data_df = data_df.iloc[filtered_molecules_idx, ].reset_index(drop=True) 

    df_atom_counts = pd.DataFrame(list(data_df["atomic_count"])).fillna(0)

    adjacency_matrix_list = list()
    mol_atom_list = list()
    for i in range(len(data_df)):
        list_bonds, adjacency_matrix, atom_list = get_permuted_adjacency_matrix(data_df["mol"][i])
        adjacency_matrix_list.append(adjacency_matrix)
        mol_atom_list.append(atom_list)
    data_df["Adjacency_Matrix"] = adjacency_matrix_list
    data_df["Atom_list"] = mol_atom_list
    
    element_list = list(set(data_df["Atom_list"].explode()))
    k = 0
    element_properties = dict()
    for element in element_list:
        element_properties[element] = dict()
        atomic_number, atomic_weight, valency, num_outer_elec = get_properties_from_symbol(element)
        element_properties[element]["alias_num"] = int(k)
        element_properties[element]["atomic_number"] = atomic_number
        element_properties[element]["atomic_weight"] = atomic_weight
        element_properties[element]["valency"] = valency
        element_properties[element]["num_outer_elec"] = num_outer_elec
        k += 1

    element_properties = pd.DataFrame(element_properties).T
    element_properties["alias_num"] = element_properties["alias_num"].astype(int)
    element_properties["abs_valency"] = element_properties["valency"].abs()
    # print(element_properties)
    mol_atom_list, mol_atom_node_feats = list(), list()
    mol_atom_valency, adjacency_matrices = list(), list()
    for i in range(len(data_df)):
        # mol_atom_list.append(torch.tensor(element_properties["alias_num"].iloc[data_df["Atom_list"].iloc[i]]))
        # temp_mol_nodes = torch.tensor(np.array(element_properties.loc[data_df["Atom_list"].iloc[i]][["atomic_number", "atomic_weight", "abs_valency", "valency", "num_outer_elec"]]))
        mol = data_df["mol"][i]

        temp_atom_indices = torch.tensor([atom.GetIdx() for atom in mol.GetAtoms()])
        temp_atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        temp_mol_nodes = torch.tensor(np.array(element_properties.loc[temp_atom_symbols][["atomic_number", "atomic_weight", "abs_valency", "valency", "num_outer_elec"]]))
        temp_mol_nodes = torch.cat([temp_mol_nodes, temp_atom_indices.unsqueeze(-1)], dim=-1)

        mol_atom_node_feats.append(temp_mol_nodes)
        mol_atom_valency.append(torch.tensor(np.array(element_properties.loc[data_df["Atom_list"].iloc[i]][["abs_valency"]])))
        adjacency_matrices.append(torch.tensor(data_df["Adjacency_Matrix"].iloc[i]))

    # mol_atom_list = np.array(mol_atom_list, dtype=object)
    mol_atom_node_feats = np.array(mol_atom_node_feats, dtype=object)
    mol_atom_valency = np.array(mol_atom_valency, dtype=object)
    adjacency_matrices = np.array(adjacency_matrices, dtype=object)
    print("mol_atom_node_feats:", mol_atom_node_feats.shape[0], mol_atom_node_feats[0].shape)
    print("mol_atom_valency:", mol_atom_valency.shape[0], mol_atom_valency[0].shape)
    print("adjacency_matrices:", adjacency_matrices.shape[0])

    return data_df, element_properties, mol_atom_node_feats, mol_atom_valency, adjacency_matrices


# data_df, element_properties, mol_atom_node_feats, mol_atom_valency, adjacency_matrices = load_data()
# # print("mol_atom_list:", mol_atom_list.shape[0], mol_atom_list[0].shape)
# print("mol_atom_node_feats:", mol_atom_node_feats.shape[0], mol_atom_node_feats[0].shape)
# print("mol_atom_valency:", mol_atom_valency.shape[0], mol_atom_valency[0].shape)
# print("adjacency_matrices:", adjacency_matrices.shape[0])

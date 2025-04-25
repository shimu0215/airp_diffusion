import os
import re
import glob
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def nbatch_transform(indices):
    nbatch = torch.tensor([(indices[i + 1] - indices[i]).item() for i in range(len(indices) - 1)])

    return nbatch

def batch_transform(indices):
    batch = torch.tensor([i for i in range(len(indices) - 1) for _ in range(indices[i + 1] - indices[i])])

    return batch

def generate_edge_mask(atom_number, max_atom_number):
    A = torch.ones(atom_number, atom_number)
    B = torch.zeros(max_atom_number, max_atom_number)

    # 将 A 复制到 B 的左上角区域
    B[:atom_number, :atom_number] = A

    return B

def generate_atom_mask(atom_number, max_atom_number):
    A = torch.ones(atom_number, 1)
    B = torch.zeros(max_atom_number, 1)
    B[:atom_number] = A
    return B

def extend_atom_3_feature(pos, max_atom_number):
    n_atoms = pos.shape[0]
    pos_ext = torch.zeros(max_atom_number, 3)
    pos_ext[:n_atoms] = pos

    return pos_ext

def extend_atom_feature(feature, max_atom_number):
    n_atoms = feature.shape[0]
    new_feature = torch.zeros(max_atom_number)
    new_feature[:n_atoms] = feature

    return new_feature

def extend_atomic_numbers_one_hot(atomic_numbers, max_atom_number):
    n_atoms = atomic_numbers.shape[0]
    atomic_numbers_ext = torch.zeros(max_atom_number, 10)
    atomic_numbers_ext[:n_atoms] = atomic_numbers

    return atomic_numbers_ext

def generate_a_sample_dataset(loaded_tensor, species_name, max_atom_number):

    database = loaded_tensor[0]

    n_atoms = len(database.z)
    n_molecules = len(database.y)

    species = -1
    if species_name == "anion":
        species = 0
    elif species_name == "cation":
        species = 1
    elif species_name == "neutral":
        species = 2
    elif species_name == "radical":
        species = 3

    pos = database.pos
    energy_grad = database.energy_grad
    z = database.z
    original_batch = batch_transform(loaded_tensor[1]['z'])
    natoms = nbatch_transform(loaded_tensor[1]['z'])
    energy = database.energy
    npa_charges = database.npa_charges

    data_list = []
    indices = loaded_tensor[1]['z']

    for i in range(n_molecules):
        start, end = indices[i], indices[i + 1]

        data = Data(
        pos=extend_atom_3_feature(pos[start:end], max_atom_number), # atom
        atomic_numbers=extend_atom_feature(z[start:end], max_atom_number), # atom
        atomic_numbers_one_hot = extend_atomic_numbers_one_hot(F.one_hot(z[start:end], num_classes=10), max_atom_number), # atom
        atom_mask = generate_atom_mask(z[start:end].shape[0], max_atom_number), # atom
        edge_mask = generate_edge_mask(z[start:end].shape[0], max_atom_number),
        # original_batch=original_batch[start:end], # atom
        batch=extend_atom_feature(original_batch[start:end], max_atom_number),  # atom
        natoms = natoms[i], # molecule
        energy = energy[i], # molecule
        energy_grad = extend_atom_3_feature(energy_grad[start:end], max_atom_number), # atom
        npa_charges = extend_atom_feature(npa_charges[start:end], max_atom_number), # atom
        species = extend_atom_feature(torch.full((end-start,), species), max_atom_number) # atom
        )

        # data = Data(
        #     pos=pos[start:end],  # atom
        #     atomic_numbers=z[start:end],  # atom
        #     atomic_numbers_one_hot=F.one_hot(z[start:end], num_classes=10),  # atom
        #     atom_mask=z[start:end].shape[0],  # atom
        #     # edge_mask=z[start:end].shape[0],
        #     # original_batch=original_batch[start:end], # atom
        #     batch=original_batch[start:end],  # atom
        #     natoms=natoms[i],  # molecule
        #     energy=energy[i],  # molecule
        #     energy_grad=energy_grad[start:end],  # atom
        #     npa_charges=npa_charges[start:end],  # atom
        #     species=torch.full((end - start,), species)  # atom
        # )

        data_list.append(data)

    return data_list

def split_dataset(data, train_ratio=0.6, val_ratio=0.3):

    train_set, temp_set = train_test_split(data, train_size=train_ratio, random_state=42)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    return train_set, val_set, test_set

def calculate_max_node_number(loaded_tensor):
    diff = torch.abs(loaded_tensor[1:] - loaded_tensor[:-1])
    max_diff = torch.max(diff)

    return max_diff.item()


def read_dataset(path, max_atom_number):

    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))
    pattern = re.compile(r"qm9star_(.+?)_chunk\d+_processed\.pt")

    data_list = []
    for file in file_list:
        match = pattern.search(file)
        name = match.group(1)

        loaded_tensor = torch.load(file, weights_only=False)
        dataset = generate_a_sample_dataset(loaded_tensor, name, max_atom_number)

        data_list += dataset

    train_list, val_list, test_list = split_dataset(data_list, train_ratio=0.9, val_ratio=0.0005)

    return train_list, val_list, test_list

def calculate_n_nodes(loaded_tensor_z):

    values, counts = torch.unique(loaded_tensor_z, return_counts=True)

    freq_dict = {int(v): int(c) for v, c in zip(values, counts)}
    freq_dict = dict(sorted(freq_dict.items()))

    pretty = ", ".join(f"{k}: {v}" for k, v in freq_dict.items())
    print(pretty)

    return freq_dict


def calculate_datasets_config(path):
    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))
    pattern = re.compile(r"qm9star_(.+?)_chunk\d+_processed\.pt")

    data_list = []
    n_nodes = {}
    for file in file_list:
        match = pattern.search(file)
        name = match.group(1)

        loaded_tensor = torch.load(file, weights_only=False)

        curr_n_nodes = calculate_n_nodes(loaded_tensor[0].z)
        n_nodes.update(curr_n_nodes)
        print(curr_n_nodes)


if __name__ == "__main__":

    # calculate_datasets_config("data/processed")

    max_atom_number = 28

    # load data set
    train_list, val_list, test_list = read_dataset("data/processed", max_atom_number = max_atom_number)
    #
    train_loader = DataLoader(train_list, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_list, batch_size=32)
    test_loader = DataLoader(test_list)

    # for batch_idx, batch in enumerate(test_loader):
    #     # batch = batch.to(device)  # 移动到 GPU（如果有）
    #
    #     # 检查节点特征 x
    #
    #     if batch.x is not None and (torch.isnan(batch.pos).any() or torch.isinf(batch.pos).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
    #     if batch.x is not None and (torch.isnan(batch.atomic_numbers).any() or torch.isinf(batch.atomic_numbers).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
    #     if batch.x is not None and (torch.isnan(batch.batch).any() or torch.isinf(batch.batch).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
    #     if batch.x is not None and (torch.isnan(batch.natoms).any() or torch.isinf(batch.natoms).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
    #     if batch.x is not None and (torch.isnan(batch.energy).any() or torch.isinf(batch.energy).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
    #     if batch.x is not None and (torch.isnan(batch.energy_grad).any() or torch.isinf(batch.energy_grad).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")
    #     if batch.x is not None and (torch.isnan(batch.npa_charges).any() or torch.isinf(batch.npa_charges).any()):
    #         print(f"🚨 Found NaN/Inf in batch {batch_idx}: x")


    # print(train_loader)
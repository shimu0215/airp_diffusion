import torch
import os
from AIRP_read_data import read_dataset
from diffusion import set_seed

def save_large_object(obj_list, save_dir, chunk_size_gb=3):
    """
    将一个很大的 list (比如train_list) 分块保存到 save_dir，每块大小大约为 chunk_size_gb GB。
    """
    os.makedirs(save_dir, exist_ok=True)

    current_chunk = []
    current_size = 0
    chunk_idx = 0

    def estimate_size(obj):
        size = 0
        for attr in obj.__dict__['_store'].values():
            if hasattr(attr, 'element_size') and hasattr(attr, 'nelement'):
                size += attr.element_size() * attr.nelement()
        return size / 1024 / 1024


    for i, obj in enumerate(obj_list):
        obj_mb = estimate_size(obj)
        current_size += obj_mb
        current_chunk.append(obj)

        if current_size >= chunk_size_gb * 1024:  # 超过设定大小，保存一块
            torch.save(current_chunk, os.path.join(save_dir, f'chunk_{chunk_idx}.pt'))
            print(f"Saved chunk {chunk_idx} with ~{current_size:.2f} MB")
            current_chunk = []
            current_size = 0
            chunk_idx += 1

    # 保存最后一块
    if current_chunk:
        torch.save(current_chunk, os.path.join(save_dir, f'chunk_{chunk_idx}.pt'))
        print(f"Saved final chunk {chunk_idx} with ~{current_size:.2f} MB")

import glob

def load_large_object(load_dir, map_location='cpu'):
    """
    加载 save_large_object 保存的所有小块，并合并成一个大的 list。
    """
    all_files = sorted(glob.glob(os.path.join(load_dir, 'chunk_*.pt')))
    all_data = []

    for file in all_files:
        chunk = torch.load(file, map_location=map_location)
        all_data.extend(chunk)
        print(f"Loaded {file}, total loaded: {len(all_data)} objects")

    return all_data

if __name__ == '__main__':

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    batch_size = 4096
    data_path = "../airp/processed-real"
    max_atom_number = 29
    max_atom_id = 10

    T = torch.tensor(1000)

    train_list, val_list, test_list = read_dataset(data_path, max_atom_number=max_atom_number, max_atom_id = max_atom_id, train=0.8, val=0.1)


    # save_path = "/scratch/wzhao20/airp_saved_data"
    # torch.save(train_list, '/scratch/wzhao20/airp_saved_data/train_list.pt')
    # torch.save(val_list, '/scratch/wzhao20/airp_saved_data//val_list.pt')
    # torch.save(test_list, '/scratch/wzhao20/airp_saved_data//test_list.pt')

    save_large_object(train_list, save_dir='/scratch/wzhao20/airp_saved_data/train_chunks', chunk_size_gb=1)
    save_large_object(train_list, save_dir='/scratch/wzhao20/airp_saved_data/val_chunks', chunk_size_gb=1)
    save_large_object(train_list, save_dir='/scratch/wzhao20/airp_saved_data/test_chunks', chunk_size_gb=1)


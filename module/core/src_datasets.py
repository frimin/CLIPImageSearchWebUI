IMAGE_EXT = (".png", ".jpg", ".jpeg",".webp")

import os
import torch
from tqdm import tqdm

class SrcDataset(torch.utils.data.Dataset):
    def __init__(self, root, ext=None, endswith_str=None):
        self.files = []

        if ext is None:
            self.ext = IMAGE_EXT
        else:
            self.ext = ext

        if root[-1] == '\\' or root[-1] == '/':
            root = root[:-1]

        if endswith_str is None:
            for dir_path, dir_names, filenames in tqdm(os.walk(root), desc=f"Get files: {root}"):
                self.files += [ os.path.join(dir_path, i) for i in filenames if os.path.splitext(i)[1] in self.ext ] 
        else:
            for dir_path, dir_names, filenames in tqdm(os.walk(root), desc=f"Get files: {root}"):
                self.files += [ os.path.join(dir_path, i) for i in filenames if i.endswith(endswith_str) ] 

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.files[idx]

    def get_files(self):
        return self.files
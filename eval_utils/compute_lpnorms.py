import os
import torch
import argparse
import numpy as np
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data

class CFDataset():
    def __init__(self, path):

        self.images = []
        self.path = path
        for bucket_folder in sorted(glob.glob(self.path + "/bucket*")):
            self.images += [(original, counterfactual) for original, counterfactual in zip(sorted(glob.glob(bucket_folder + "/original/*.png")), sorted(glob.glob(bucket_folder + "/counterfactual/*.png")))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original_path, counterfactual_path = self.images[idx]

        original = self.load_img(original_path)
        counterfactual = self.load_img(counterfactual_path)

        return original, counterfactual

    def load_img(self, path):
        img = Image.open(os.path.join(path))
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)

    def transform(self, img):
        img = img.astype(np.float32) / 255
        img = torch.from_numpy(img).float()
        img = img.permute((2, 0, 1))  # C x H x W
        return img

def compute_lp_norms(args):
    dataset = CFDataset(args.output_path)
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=16, pin_memory=True)
    l1, l2 = [], []
    for orig, cf in tqdm(loader, leave=False):
        diff = orig.view(orig.shape[0], -1) - cf.view(cf.shape[0], -1) 
        l1 += list(torch.norm(diff, p=1, dim=-1).cpu().numpy())
        l2 += list(torch.norm(diff, p=2, dim=-1).cpu().numpy())
    return np.mean(l1), np.mean(l2)

def arguments():
    parser = argparse.ArgumentParser(description="LP norm evaluation")
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--batch-size', required=False, type=int, default=15)
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    device = torch.device("cuda")
    
    l1, l2 = compute_lp_norms(args)

    print(f"L1: {l1}")
    print(f"L2: {l2}")

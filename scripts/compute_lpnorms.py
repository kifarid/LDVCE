import os
import torch
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm

def load_img(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img

def arguments():
    parser = argparse.ArgumentParser(description="LP norm evaluation")
    parser.add_argument('--cf_dir', required=True, type=str)
    parser.add_argument('--orig_dir', required=True, type=str)

    return parser.parse_args()

def compute_lp_norms(counterfactual_dir: str, original_dir: str):
    l1_criterion, l2_criterion = torch.nn.L1Loss(), torch.nn.MSELoss()
    l1, l2 = [], []
    for cf_example in tqdm(os.listdir(counterfactual_dir)):
        counterfactual = load_img(os.path.join(counterfactual_dir, cf_example))
        raise NotImplementedError
        original = load_img(os.path.join(original_dir, cf_example)) # TODO
        l1.append(l1_criterion(counterfactual, original).item())
        l2.append(l2_criterion(counterfactual, original).item())
    return np.mean(l1), np.mean(l2)

if __name__ == '__main__':
    args = arguments()
    device = torch.device("cuda")
    
    l1, l2 = compute_lp_norms(counterfactual_dir=args.cf_dir, original_dir=args.orig_dir)

    print(f"L1: {l1}")
    print(f"L2: {l2}")

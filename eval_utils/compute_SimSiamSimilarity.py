import os
import torch
import torch.nn as nn
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from utils.simsiam import get_simsiam_dist

# create dataset to read the counterfactual results images
class CFDataset():
    def __init__(self, path):

        self.images = []
        self.path = path
        for bucket_folder in glob.glob(self.path + "/bucket*"):
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
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        return img



@torch.inference_mode()
def _compute_S3(oracle,
                path,
                batch_size,
                device):

    dataset = CFDataset(path)
    dists = []
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False,
                             num_workers=16, pin_memory=True)

    for cl, cf in tqdm(loader, leave=False):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)
        dists.append(oracle(cl, cf).cpu().numpy())

    return np.concatenate(dists)

def compute_s3(args):
    device = torch.device('cuda')
    oracle = get_simsiam_dist(args.weights_path)
    oracle.to(device)
    oracle.eval()
    return _compute_S3(oracle, args.output_path, args.batch_size, device)
    

def arguments():
    parser = argparse.ArgumentParser(description='S^3 arguments.')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--weights-path', default='pretrained_models/checkpoint_0099.pth.tar', type=str,
                        help='ResNet50 SimSiam model weights')
    parser.add_argument('--batch-size', default=15, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    results = compute_s3(args)
    print('SimSiam Similarity: {:>4f}'.format(np.mean(results).item()))
    

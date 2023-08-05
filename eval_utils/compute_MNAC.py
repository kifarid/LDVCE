import os
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp

from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils import data
from torchvision import transforms
import glob

from utils.preprocessor import Normalizer

#from eval_utils.oracle_celeba_metrics import OracleMetrics
from utils.oracle_celebahq_metrics import OracleResnet

BINARYDATASET = ['CelebA', 'CelebAHQ', 'CelebAMV', 'BDD']
MULTICLASSDATASETS = ['ImageNet']


# create dataset to read the counterfactual results images
class CFDataset():
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    def __init__(self, path):
        self.images = []
        self.path = path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
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
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transform(img)


@torch.inference_mode()
def compute_MNAC(oracle,
                 path,
                 batch_size,
                 device):

    dataset = CFDataset(path)
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    MNACS = []
    dists = []
    for cl, cf in tqdm(loader, leave=False):
        d_cl = oracle(cl.to(device, dtype=torch.float))
        d_cf = oracle(cf.to(device, dtype=torch.float))
        MNACS.append(((d_cl > 0.5) != (d_cf > 0.5)).sum(dim=1).cpu().numpy())
        dists.append([d_cl.cpu().numpy(), d_cf.cpu().numpy()])

    return np.concatenate(MNACS), np.concatenate([d[0] for d in dists]), np.concatenate([d[1] for d in dists])


class CelebaOracle():
    def __init__(self, weights_path, device):
        self.oracle = OracleMetrics(weights_path=weights_path,
                                    device=device)
        self.oracle.eval()

    def __call__(self, x):
        return torch.sigmoid(self.oracle.oracle(x)[1])


class CelebaHQOracle():
    def __init__(self, weights_path, device):
        oracle = OracleResnet(weights_path=None,
                                   freeze_layers=True)
        oracle.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state_dict'])
        self.oracle = Normalizer(oracle, [0.5] * 3, [0.5] * 3)
        self.oracle.to(device)
        self.oracle.eval()

    def __call__(self, x):
        return self.oracle(x)

def compute_mnac(args):
    # load oracle trained on vggface2 and fine-tuned on CelebA
    device = torch.device('cuda')
    
    if args.dataset == 'CelebA':
        oracle = CelebaOracle(weights_path=args.oracle_path,
                              device=device)
    elif args.dataset == 'CelebAHQ':
        oracle = CelebaHQOracle(weights_path=args.oracle_path,
                                device=device)
    
    results = compute_MNAC(oracle,
                           args.output_path,
                           args.batch_size,
                           device)
    return results

def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--oracle-path', default='models/oracle.pth', type=str,
                        help='Oracle path')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--dataset', required=True, type=str,
                        choices=BINARYDATASET + MULTICLASSDATASETS,
                        help='Dataset to evaluate')
    parser.add_argument('--batch-size', default=15, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    
    results = compute_mnac(args)

    print('MNAC:', np.mean(results[0]))
    print('MNAC:', np.std(results[0]))

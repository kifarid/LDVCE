import os
import torch
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp

from tqdm import tqdm
from torch.utils import data
import glob
from torchvision import transforms
from PIL import Image

from utils.resnet50_facevgg2_FVA import resnet50, load_state_dict

class CFDataset():
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    def __init__(self, path, transforms=None):
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
        img = transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

@torch.inference_mode()
def compute_FVA(oracle,
                path,
                batch_size,
                device):

    dataset = CFDataset(path)
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    cosine_similarity = torch.nn.CosineSimilarity()
    FVAS = []
    dists = []
    for cl, cf in tqdm(loader, leave=False):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)
        cl_feat = oracle(cl)
        cf_feat = oracle(cf)
        dist = cosine_similarity(cl_feat, cf_feat)
        FVAS.append((dist > 0.5).cpu().numpy())
        dists.append(dist.cpu().numpy())

    return np.concatenate(FVAS), np.concatenate(dists)

def compute_fva(args):
    device = torch.device('cuda')
    oracle = resnet50(num_classes=8631, include_top=False).to(device)
    load_state_dict(oracle, args.weights_path)
    oracle.eval()

    results = compute_FVA(oracle,
                          args.output_path,
                          args.batch_size,
                          device)
    return results

def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--weights-path', default='pretrained_models/resnet50_ft_weight.pkl', type=str,
                        help='ResNet50 VGGFace2 model weights')
    parser.add_argument('--batch-size', default=15, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    
    results = compute_fva(args)

    print('FVA', np.mean(results[0]))
    print('FVA (STD)', np.std(results[0]))
    print('mean dist', np.mean(results[1]))
    print('std dist', np.std(results[1]))

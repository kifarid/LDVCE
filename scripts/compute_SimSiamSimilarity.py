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


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        self.criterion = nn.CosineSimilarity(dim=1)

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        # p1 = self.predictor(z1) # NxC
        # p2 = self.predictor(z2) # NxC

        # dist = (self.criterion(p1, z2) + self.criterion(p2, z1)) * 0.5
        dist = self.criterion(z1, z2)

        return dist


def get_simsiam_dist(weights_path):
    import torchvision.models as models
    model = SimSiam(models.resnet50, dim=2048, pred_dim=512)
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    model.load_state_dict(
        {k[7:]: v for k, v in state_dict.items()}
    )
    return model


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
def compute_S3(oracle,
                path,
                batch_size):

    dataset = CFDataset(path)
    dists = []
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=False,
                             num_workers=16, pin_memory=True)

    for cl, cf in tqdm(loader):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)
        dists.append(oracle(cl, cf).cpu().numpy())

    return np.concatenate(dists)


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
    device = torch.device('cuda')
    oracle = get_simsiam_dist(args.weights_path)
    oracle.to(device)
    oracle.eval()

    results = compute_S3(oracle, args.output_path, args.batch_size)

    print('SimSiam Similarity: {:>4f}'.format(np.mean(results).item()))

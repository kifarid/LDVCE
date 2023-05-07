# code from https://github.com/guillaumejs2403/ACE

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
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
from torchvision import transforms

try:
    from models.dive.densenet import DiVEDenseNet121
    from models.steex.DecisionDensenetModel import DecisionDensenetModel
    from models.mnist import Net
except:
    pass

try:
    from guided_diffusion.image_datasets import BINARYDATASET, MULTICLASSDATASETS
except:
    pass

class Normalizer(torch.nn.Module):
    '''
    normalizing module. Useful for computing the gradient
    to a x image (x in [0, 1]) when using a classifier with
    different normalization inputs (i.e. f((x - mu) / sigma))
    '''
    def __init__(self, classifier,
                 mu=[0.485, 0.456, 0.406],
                 sigma=[0.229, 0.224, 0.225]):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor(mu).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma).view(1, -1, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.classifier(x)

def gen_masks(inputs, targets, mode='abs'):
    """
        generates a difference masks give two images (inputs and targets).
    :param inputs:
    :param targets:
    :param mode:
    :return:
    """
    masks = targets - inputs
    masks = masks.view(inputs.size(0), 3, -1)

    if mode == 'abs':
        masks = masks.abs()
        masks = masks.sum(dim=1)
        # normalize 0 to 1
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    elif mode == "mse":
        masks = masks ** 2
        masks = masks.sum(dim=1)
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    else:
        raise ValueError("mode value is not valid!")

    return masks.view(inputs.size(0), 1, inputs.size(2), inputs.size(3))


@torch.no_grad()
def evaluate(label, target, classifier, loader, device, binary):
    """
        evaluates loss values and metrics.
    :param encoder:
    :param maps:
    :param generator:
    :param discriminator:
    :param classifier:
    :param dataloader:
    :param writer:
    :param epoch:
    :return:
    """
    # eval params
    cout_num_steps = 50

    # init scores
    cout = 0
    total_samples = 0
    plot_data = {'c_curve': [],
                 'c_prime_curve': []}

    with torch.no_grad():
        for i, (img, cf) in enumerate(tqdm(loader)):
            img = img.to(device, dtype=torch.float)
            cf = cf.to(device, dtype=torch.float)

            # calculate metrics
            cout_score, plot_info = calculate_cout(
                img,
                cf,
                gen_masks(img, cf, mode='abs'),
                classifier,
                label,
                target,
                max(1, (img.size(2) * img.size(3)) // cout_num_steps),
                binary
            )
            plot_data['c_curve'].append([d.cpu() for d in plot_info[0]])
            plot_data['c_prime_curve'].append([d.cpu() for d in plot_info[1]])
            cout += cout_score

            # update total sample number
            total_samples += img.shape[0]

    # process plot info
    curves = torch.zeros(2, len(plot_info[2]))
    for idx, curve_name in enumerate(['c_curve', 'c_prime_curve']):
        for data_points in plot_data[curve_name]:
            data_points = torch.cat([d.unsqueeze(dim=1) for d in data_points], dim=1)
            curves[idx, :] += data_points.sum(dim=0)
    curves /= total_samples
    cout /= total_samples
    print(f"\nEVAL [COUT: {cout:.4f}]")
    return cout, {'indexes': plot_info[2], 'c_curve': curves[0, :].numpy(), 'c_prime_curve': curves[1, :].numpy()}


@torch.no_grad()
def get_probs(label, target, img, model, binary):
    '''
        Computes the probabilities of the target/label classes
    '''
    if binary:
        # for the binary classification, the target is irrelevant since it is 1 - label
        output = model(img)
        pos = (label == 1).float()
        c_curve = torch.sigmoid(pos * output - (1 - pos) * output)
        c_prime_curve = 1 - c_curve
    else:
        output =  F.softmax(model(img))
        c_curve = output[:, label]
        c_prime_curve = output[:, target]

    return c_curve, c_prime_curve


@torch.no_grad()
def calculate_cout(imgs, cfs, masks, model, label, target, step, binary):
    """
        calculates the counterfactual transition (cout) score.
        Produce the results solely for correctly classified images
    :param imgs:
    :param cfs:
    :param masks:
    :param model:
    :param cls_1:
    :param cls_2:
    :param step:
    :return:
    """

    # The dimensions for the image
    img_size = imgs.shape[-2:]
    mask_size = masks.shape[-2:]

    # Compute the total number of pixels in a mask
    num_pixels = torch.prod(torch.tensor(masks.shape[1:])).item()
    l = torch.arange(imgs.shape[0])

    if binary:
        label = (model(imgs) > 0.0)

    # Initial values for the curves
    c_curve, c_prime_curve = get_probs(label, target, imgs, model, binary)
    c_curve = [c_curve]
    c_prime_curve = [c_prime_curve]
    index = [0.]

    # init upsampler
    up_sample = torch.nn.UpsamplingBilinear2d(size=img_size).to(imgs.device)

    # updating mask and the ordering
    cur_mask = torch.zeros((masks.shape[0], num_pixels)).to(imgs.device)
    elements = torch.argsort(masks.view(masks.shape[0], -1), dim=1, descending=True)

    for pixels in range(0, num_pixels, step):
        # Get the indices used in this iteration
        indices = elements[l, pixels:pixels + step].squeeze().view(imgs.shape[0], -1)

        # Set those indices to 1
        cur_mask[l, indices.permute(1, 0)] = 1
        up_masks = up_sample(cur_mask.view(-1, 1, *mask_size))

        # perturb the image using cur mask and calculate scores
        perturbed = phi(cfs, imgs, up_masks)
        score_c, score_c_prime = get_probs(label, target, perturbed, model, binary)

        # obtain the scores
        c_curve.append(score_c)
        c_prime_curve.append(score_c_prime)
        index.append((pixels + step) / num_pixels)

    auc_c, auc_c_prime = auc(c_curve), auc(c_prime_curve)
    auc_c *= step / (mask_size[0] * mask_size[1])
    auc_c_prime *= step / (mask_size[0] * mask_size[1])
    cout = auc_c_prime.sum().item() - auc_c.sum().item()

    return cout, (c_curve, c_prime_curve, index)


def phi(img, baseline, mask):
    """
        composes an image from img and baseline according to the mask values.
    :param img:
    :param baseline:
    :param mask:
    :return:
    """
    return img.mul(mask) + baseline.mul(1-mask)


def auc(curve):
    """
        calculates the area under the curve
    :param curve:
    :return:
    """
    return curve[0]/2 + sum(curve[1:-1]) + curve[-1]/2

# create dataset to read the counterfactual results images
class CFDataset():
    def __init__(self, path, exp_name):

        self.images = []
        self.path = path
        self.exp_name = exp_name
        for CL, CF in itertools.product(['CC'], ['CCF', 'ICF']):
            self.images += [(CL, CF, I) for I in os.listdir(osp.join(path, 'Results', self.exp_name, CL, CF, 'CF'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        CL, CF, I = self.images[idx]
        # get paths
        cl_path = osp.join(self.path, 'Original', 'Correct' if CL == 'CC' else 'Incorrect', I)
        cf_path = osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF', I)

        cl = self.load_img(cl_path)
        cf = self.load_img(cf_path)

        return cl, cf

    def load_img(self, path):
        img = Image.open(os.path.join(path))
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)

    def transform(self, img):
        img = img.astype(np.float32) / 255
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img


def arguments():
    parser = argparse.ArgumentParser(description='COUT arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--weights-path', required=True, type=str,
                        help='Classification model weights')
    parser.add_argument('--dataset', required=True, type=str,
                        choices=BINARYDATASET + MULTICLASSDATASETS + ["ImageNet"],
                        help='Dataset to evaluate')
    parser.add_argument('--batch-size', default=10, type=int,
                        help='Batch size')
    parser.add_argument('--query-label', required=True, type=int)
    parser.add_argument('--target-label', required=True, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    device = torch.device('cuda:' + args.gpu)

    dataset = CFDataset(args.output_path, args.exp_name)

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    print('Loading Classifier')

    ql = args.query_label
    if args.dataset in ['CelebA', 'CelebAMV']:
        classifier = Normalizer(
            DiVEDenseNet121(args.weights_path, args.query_label),
            [0.5] * 3, [0.5] * 3
        ).to(device)

    elif args.dataset == 'CelebAHQ':
        assert args.query_label in [20, 31, 39], 'Query label MUST be 20 (Gender), 31 (Smile), or 39 (Gender) for CelebAHQ'
        ql = 0
        if args.query_label in [31, 39]:
            ql = 1 if args.query_label == 31 else 2
        classifier = DecisionDensenetModel(3, pretrained=False,
                                           query_label=ql)
        classifier.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model_state_dict'])
        classifier = Normalizer(
            classifier,
            [0.5] * 3, [0.5] * 3
        ).to(device)

    elif 'BDD' in args.dataset:
        classifier = DecisionDensenetModel(4, pretrained=False,
                                           query_label=args.query_label)
        classifier.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model_state_dict'])
        classifier = Normalizer(
            classifier,
            [0.5] * 3, [0.5] * 3
        ).to(device)

    else:
        classifier = Normalizer(
            models.resnet50(pretrained=True)
        ).to(device)
    
    classifier.eval()

    results = evaluate(ql,
                       args.target_label,
                       classifier,
                       loader,
                       device,
                       args.dataset in BINARYDATASET)

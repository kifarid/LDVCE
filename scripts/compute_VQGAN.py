import argparse
import glob
from PIL import Image
import os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

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
        img = 2. * img - 1 # normalize to [-1, +1]
        return img
    
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    #model#.to(device)  # .cuda()
    model.eval()
    return model

def get_model(cfg_path="configs/latent-diffusion/v1-inference.yaml", ckpt_path="models/ldm/stable-diffusion-v1/model.ckpt"):
    config = OmegaConf.load(cfg_path)
    model = load_model_from_config(config, ckpt_path)
    return model

@torch.inference_mode()
def compute_vqgan(args):
    device = torch.device("cuda")
    
    dataset = CFDataset(args.output_path)
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=16, pin_memory=True)
    
    model = get_model(cfg_path="configs/stable-diffusion/v1-inference.yaml", ckpt_path = args.diffusion_ckpt).to(device).eval()
    
    l1, l2 = [], []
    for original, counterfactual in tqdm(loader, leave=False):
        # move to latent space
        original_latent = model.get_first_stage_encoding(
            model.encode_first_stage(original.to(device)))
        counterfactual_latent = model.get_first_stage_encoding(
            model.encode_first_stage(counterfactual.to(device)))  
        
        # compute lp norms
        diff = original_latent.view(original_latent.shape[0], -1) - counterfactual_latent.view(counterfactual_latent.shape[0], -1) 
        l1 += list(torch.norm(diff, p=1, dim=-1).cpu().numpy())
        l2 += list(torch.norm(diff, p=2, dim=-1).cpu().numpy())
    
    return np.mean(l1), np.mean(l2)

def arguments():
    parser = argparse.ArgumentParser(description="VQGAN perceptual similarity")
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--diffusion-ckpt', required=True, type=str,
                        help='Results Path')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    return parser.parse_args()

if __name__ == "__main__":
    args = arguments()

    l1, l2 = compute_vqgan(args)

    print(args.output_path)
    print(f"L1: {l1}")
    print(f"L2: {l2}")
import argparse
import os
import yaml
import copy

import matplotlib.pyplot as plt
import numpy as np


import torch
from contextlib import nullcontext
from torch import autocast

from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torchvision
from torchvision import transforms, datasets

from src.clipseg.models.clipseg import CLIPDensePredT
from sampling_helpers import disabled_train, get_model, _unmap_img, generate_samples


import sys
import regex as re
from ldm import *
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler

# sys.path.append(".")
# sys.path.append('./taming-transformers')
LMB_USERNAME = "faridk"
os.environ["WANDB_API_KEY"] = 'cff06ca1fa10f98d7fde3bf619ee5ec8550aba11'
os.environ['WANDB_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.wandb"
os.environ['WANDB_DATA_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals"
os.environ['WANDB_CACHE_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.cache/wandb"
WANDB_ENTITY = "kifarid"
WANDB_ENABLED = True

i2h = dict()
with open('data/imagenet_clsidx_to_label.txt', "r") as f:
    lines = f.read().splitlines()
    assert len(lines) == 1000

    for line in lines:
        key, value = line.split(":")
        i2h[int(key)] = re.sub(r"^'|',?$", "", value.strip()) #value.strip().strip("'").strip(",").strip("\"")


@hydra.main(version_base=None, config_path="../configs/dvce", config_name="v4")
def main(cfg : DictConfig) -> None:
    # load model
    config = {}
    config.update(OmegaConf.to_container(cfg, resolve=True))

    run = wandb.init(entity=WANDB_ENTITY, project="cdiff", dir = os.environ['WANDB_DATA_DIR'], config=config, mode="enabled" if WANDB_ENABLED else "disabled")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if cfg.seg_model is not None:
        print("### Loading segmentation model ###")
        model_seg = CLIPDensePredT(version=cfg.seg_model.version, reduce_dim=64) #int(cfg.seg_model.version.split('/')[-1]
        model_seg.eval()
        model_seg.load_state_dict(torch.load(cfg.seg_model.path, map_location=torch.device('cpu')), strict=False)

    model = get_model(cfg_path=cfg.diffusion_model.cfg_path, ckpt_path = cfg.diffusion_model.ckpt_path).to(device)
    classifier_name = cfg.classifier_model.name
    classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True).to(device)
    classifier_model = classifier_model.eval()
    classifier_model.train = disabled_train

    torch.autograd.set_detect_anomaly(True)
    ddim_steps = cfg.ddim_steps
    ddim_eta = cfg.ddim_eta
    scale = cfg.scale #for unconditional guidance
    strength = cfg.strength #for unconditional guidance
    sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg.to(device) if cfg.seg_model is not None else None, **cfg.sampler)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    n_samples_per_class = cfg.n_samples_per_class
    #precision = "autocast" #"full"
    #precision_scope = autocast if precision == "autocast" else nullcontext
      
    print(config)

    data_path = cfg.data_path
    out_size = 256
    transform_list = [
        transforms.Resize((out_size, out_size)),
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    dataset = datasets.ImageFolder(data_path,  transform=transform)
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    with open('data/synset_closest_idx.yaml', 'r') as file:
        synset_closest_idx = yaml.safe_load(file)
        
    my_table = wandb.Table(columns=["id", "image", "source", "target", "lp1", "lp2", *[f"gen_image_{i}" for i in range(n_samples_per_class)], "class_prediction", "video", "mask"])
    
    #for i, sample in enumerate(dataset, 1000):
    #    image, label = dataset[i]
    for i, batch in enumerate(data_loader):
        image, label = batch
        image = image.squeeze().to(device)
        label = label.squeeze().to(device).item()

        tgt_classes = synset_closest_idx[label]
        print(f"converting {i} from : {i2h[label]} to: {[i2h[tgt_class] for tgt_class in tgt_classes]}")
        init_image = image.repeat(n_samples_per_class, 1, 1, 1).to(device)
        sampler.init_images = init_image.to(device)
        sampler.init_labels = n_samples_per_class * [label]
        if isinstance(cfg.sampler.lp_custom, str) and "dino_" in cfg.sampler.lp_custom:
            if device != next(sampler.distance_criterion.dino.parameters()).device:
                sampler.distance_criterion.dino = sampler.distance_criterion.dino.to(device)
            sampler.dino_init_features = sampler.get_dino_features(sampler.init_images, device=device).clone()
        mapped_image = _unmap_img(init_image)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(_unmap_img(init_image)))  # move to latent space

        out = generate_samples(model, sampler, tgt_classes, n_samples_per_class, ddim_steps, scale, init_latent=init_latent.to(device),
                               t_enc=t_enc, init_image=init_image.to(device), ccdddim=True, latent_t_0=cfg.get("latent_t_0", False))

        all_samples = out["samples"]
        all_videos = out["videos"] 
        all_probs = out["probs"]
        all_masks = out["masks"] 

        # Loop through your data and update the table incrementally
        for j in range(len(all_probs)):
            # Generate data for the current row
            src_image = copy.deepcopy(all_samples[j][0])
            src_image = wandb.Image(src_image)
            gen_images = []
            for k in range(n_samples_per_class):
                gen_image = copy.deepcopy(all_samples[j][k + 1])
                gen_images.append(wandb.Image(gen_image))
                
            class_prediction = copy.deepcopy(all_probs[j])
            source = i2h[label]
            target = i2h[tgt_classes[j]]

            diff =  (init_image - all_samples[j][1:])
            diff = diff.view(diff.shape[0], -1)
            lp1 = int(torch.norm(diff, p=1, dim=-1).mean().cpu().numpy())
            lp2 = int(torch.norm(diff, p=2, dim=-1).mean().cpu().numpy())
            #print(f"lp1: {lp1}, lp2: {lp2}")

            video = wandb.Video((255. * all_videos[j]).to(torch.uint8).cpu(), fps=10, format="gif")
            mask = wandb.Image(all_masks[j]) if all_masks is not None else None
            #print("added data to table")
            my_table.add_data(i, src_image, source, target, lp1, lp2, *gen_images, class_prediction, video, mask)

        if i % 2 == 0:
            print(f"logging {i} with {len(my_table.data)} rows")
            table_name = f"dvce_video" #_{i}"
            run.log({table_name: copy.deepcopy(my_table)})

    #wandb.log({"dvce_video_complete": my_table})
    return None

if __name__ == "__main__":
    main()

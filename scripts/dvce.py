import argparse
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np


import torch
from contextlib import nullcontext
from torch import autocast

from omegaconf import OmegaConf
import wandb
import torchvision
from torchvision import transforms, datasets

from src.clipseg.models.clipseg import CLIPDensePredT
from sampling_helpers import disabled_train, get_model, _unmap_img, generate_samples


import sys
import regex as re
from ldm import *
from ldm.models.diffusion.cc_ddim import CCDDIMSampler

# sys.path.append(".")
# sys.path.append('./taming-transformers')

os.environ["WANDB_API_KEY"] = 'cff06ca1fa10f98d7fde3bf619ee5ec8550aba11'
run = wandb.init(entity="kifarid", project="cdiff")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

i2h = dict()
with open('data/imagenet_clsidx_to_label.txt', "r") as f:
        lines = f.read().splitlines()
        assert len(lines) == 1000

        for line in lines:
            key, value = line.split(":")
            i2h[int(key)] = re.sub(r"^'|',?$", "", value.strip()) #value.strip().strip("'").strip(",").strip("\"")




# load model
model_seg = CLIPDensePredT(version='ViT-B/16', reduce_dim=16)
model_seg.eval();
model_seg.load_state_dict(torch.load('src/clipseg/weights/rd16-uni.pth', map_location=torch.device('cpu')), strict=False);


model = get_model().to(device)
classifier_name = "efficientnet_b0"
classifier_model=getattr(torchvision.models, classifier_name)(pretrained=True).to(device)
classifier_model = classifier_model.eval()
classifier_model.train = disabled_train

torch.autograd.set_detect_anomaly(True)
ddim_steps = 500
ddim_eta = 0.0
scale = 3.0   # for unconditional guidance


strength = 0.3#0.28#48#46#"strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image"
sampler = CCDDIMSampler(model, classifier_model)
sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
t_enc = int(strength * ddim_steps)
precision = "autocast" #"full"
precision_scope = autocast if precision == "autocast" else nullcontext
n_samples_per_class = 1

sampler.enforce_same_norms = True
sampler.guidance = "projected"
sampler.classifier_lambda = 2 #.2 #1.8 #2 final #1.5 #2.5 # 5.5 best
sampler.dist_lambda = 0.6 # 0.15
sampler.masked_dist = False
sampler.masked_guidance = False
sampler.backprop_diffusion = True
#sampler.seg_model = model_seg

sampler.lp_custom = 1

#log config
config = run.config
config.strength = strength
config.ddim_steps = ddim_steps
config.ddim_eta = ddim_eta
config.scale = scale
config.t_enc = t_enc
config.precision = precision
config.n_samples_per_class = n_samples_per_class
config.enforce_same_norms = sampler.enforce_same_norms
config.guidance = sampler.guidance
config.classifier_lambda = sampler.classifier_lambda
config.dist_lambda = sampler.dist_lambda
config.masked_dist = sampler.masked_dist
config.masked_guidance = sampler.masked_guidance
config.lp_custom = sampler.lp_custom
config.backprop_diffusion = sampler.backprop_diffusion
config.classifier_name = classifier_name



data_path = '/misc/scratchSSD2/datasets/ILSVRC2012/val'
out_size = 256
transform_list = [
    transforms.Resize((out_size,out_size)),
    transforms.ToTensor()
]
transform = transforms.Compose(transform_list)

config.data_path = data_path

dataset = datasets.ImageFolder(data_path,  transform=transform)
with open('data/synset_closest_idx.yaml', 'r') as file:
    synset_closest_idx = yaml.safe_load(file)


my_table = wandb.Table(columns = ["image", "source", "target", *[f"gen_image_{i}" for i in range(n_samples_per_class)], "class_prediction", "closness_1", "closness_2", "video"])

for i, sample in enumerate(dataset, 0):
    image, label = dataset[i]
    tgt_classes = synset_closest_idx[label]
    print(f"converting {i} from : {i2h[label]} to: {[i2h[tgt_class] for tgt_class in tgt_classes]}")
    init_image = image.repeat(n_samples_per_class, 1, 1, 1).to(device)
    plt.imshow(init_image[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()
    sampler.init_images = init_image
    sampler.init_labels = n_samples_per_class * [label]
    mapped_image = _unmap_img(init_image)
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(_unmap_img(init_image)))  # move to latent space

    out = generate_samples(model, sampler, tgt_classes, n_samples_per_class, ddim_steps, scale, init_latent=init_latent,
                           t_enc=t_enc, init_image=init_image, ccdddim=True)

    all_samples = out["samples"]
    all_videos = out["videos"]
    all_probs = out["probs"]

    # Loop through your data and update the table incrementally
    for j in range(len(all_probs)):
        # Generate data for the current row
        src_image = all_samples[j][0]
        src_image = wandb.Image(src_image)
        # my_table.update_column("image", [wandb.Image(src_image)], row_idx=i)
        gen_images = []
        for k in range(n_samples_per_class):
            gen_image = all_samples[j][k + 1]
            gen_images.append(wandb.Image(gen_image))
            # my_table.update_column(f"gen_image_{j}", [wandb.Image(gen_image)], row_idx=i)

        class_prediction = all_probs[j]
        source = i2h[label]
        target = i2h[tgt_classes[j]]

        diff = 255. * (init_image - all_samples[j][1:]).permute(0, 2, 3, 1).detach().cpu().numpy()
        closeness_2 = int(np.linalg.norm(diff.astype(np.uint8), axis=-1).mean())
        closeness_1 = int(np.abs(diff).sum(axis=-1).mean())

        video = wandb.Video((255. * all_videos[j]).to(torch.uint8).cpu(), fps=4, format="gif")
        my_table.add_data(src_image, source, target, *gen_images, class_prediction, closeness_1, closeness_2, video)
        
    if i %5 == 0:
        run.log({"dvce_video": my_table})

run.log({"dvce_video_complete": my_table})

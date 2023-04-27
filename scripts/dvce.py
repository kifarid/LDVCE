import argparse
import os
import yaml
import copy
import random

import matplotlib.pyplot as plt
import numpy as np


import torch
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
from contextlib import nullcontext
from torch import autocast

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import torchvision
from torchvision import transforms, datasets

from src.clipseg.models.clipseg import CLIPDensePredT
from segment_anything import build_sam, SamPredictor
from sampling_helpers import disabled_train, get_model, _unmap_img, generate_samples
from sampling_helpers import load_model_hf


import sys
import regex as re
from ldm import *
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler

from imagenet_classnames import name_map, folder_label_map

# sys.path.append(".")
# sys.path.append('./taming-transformers')
LMB_USERNAME = "faridk"
#check if directories exist
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.cache/wandb", exist_ok=True)
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.wandb", exist_ok=True)
os.makedirs(f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals", exist_ok=True)

os.environ["WANDB_API_KEY"] = 'cff06ca1fa10f98d7fde3bf619ee5ec8550aba11'
os.environ['WANDB_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.wandb"
os.environ['WANDB_DATA_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals"
os.environ['WANDB_CACHE_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.cache/wandb"

os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
os.makedirs(os.environ['WANDB_DATA_DIR'], exist_ok=True)
os.makedirs(os.environ['WANDB_CACHE_DIR'], exist_ok=True)

WANDB_ENTITY = "kifarid"

if "faridk" == LMB_USERNAME:
    torch.hub.set_dir(f'/misc/lmbraid21/{LMB_USERNAME}/torch')

i2h = dict()
with open('data/imagenet_clsidx_to_label.txt', "r") as f:
    lines = f.read().splitlines()
    assert len(lines) == 1000

    for line in lines:
        key, value = line.split(":")
        i2h[int(key)] = re.sub(r"^'|',?$", "", value.strip()) #value.strip().strip("'").strip(",").strip("\"")

class ImageNet(datasets.ImageFolder):
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(
            self, 
            root:str, 
            split:str="val", 
            transform=None, 
            target_transform=None, 
            class_idcs=None, 
            start_sample: int = 0, 
            end_sample: int = 50000//1000,
            return_tgt_cls: bool = False,
            idx_to_tgt_cls_path = None, 
            **kwargs
    ):
        _ = kwargs  # Just for consistency with other datasets.
        assert split in ["train", "val"]
        assert start_sample < end_sample and start_sample >= 0 and end_sample <= 50000//1000
        path = root if root[-3:] == "val" or root[-5:] == "train" else os.path.join(root, split)
        super().__init__(path, transform=transform, target_transform=target_transform)
        
        with open(idx_to_tgt_cls_path, 'r') as file:
            idx_to_tgt_cls = yaml.safe_load(file)
            if isinstance(idx_to_tgt_cls, dict):
                idx_to_tgt_cls = [idx_to_tgt_cls[i] for i in range(len(idx_to_tgt_cls))]
        self.idx_to_tgt_cls = idx_to_tgt_cls

        self.return_tgt_cls = return_tgt_cls

        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            samples = []
            idx_to_tgt_cls = []
            for i, (p, t) in enumerate(self.samples):
                if t in tgt_to_tgt_map:
                    samples.append((p, tgt_to_tgt_map[t]))
                    idx_to_tgt_cls.append(self.idx_to_tgt_cls[i])
            
            self.idx_to_tgt_cls = idx_to_tgt_cls
            #self.samples = [(p, tgt_to_tgt_map[t]) for i, (p, t) in enumerate(self.samples) if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        if "val" == split: # reorder
            new_samples = []
            idx_to_tgt_cls = []
            for idx in range(50000//1000):
                new_samples.extend(self.samples[idx::50000//1000])
                idx_to_tgt_cls.extend(self.idx_to_tgt_cls[idx::50000//1000])
            self.samples = new_samples[start_sample*1000:end_sample*1000]
            self.idx_to_tgt_cls = idx_to_tgt_cls[start_sample*1000:end_sample*1000]

        else:
            raise NotImplementedError

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.return_tgt_cls:
            return *sample, self.idx_to_tgt_cls[index]
        else:
            return sample

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path="../configs/dvce", config_name="v4")
def main(cfg : DictConfig) -> None:    
    # load model
    config = {}
    config.update(OmegaConf.to_container(cfg, resolve=True))
    run = wandb.init(entity=WANDB_ENTITY, project="cdiff", config=config, mode="online" if cfg.wandb.enabled else "offline") # dir = os.environ['WANDB_DATA_DIR']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # there seems to be a CUDA/autograd instability in gradient computation
    print(f"using device: {device}")

    if "seg_model" in cfg and cfg.seg_model is not None:
        print("### Loading segmentation model ###")
        if "name" in cfg.seg_model and cfg.seg_model.name == "clipseg":
            model_seg = CLIPDensePredT(version=cfg.seg_model.version, reduce_dim=64) #int(cfg.seg_model.version.split('/')[-1]
            model_seg.eval()
            model_seg.load_state_dict(torch.load(cfg.seg_model.path, map_location=torch.device('cpu')), strict=False)
        elif "name" in cfg.seg_model and cfg.seg_model.name == "GD_SAM":
            detect_model = load_model_hf(repo_id=cfg.seg_model.dino.repo_id, filename= cfg.seg_model.dino.filename, dir = cfg.seg_model.dino.dir, ckpt_config_filename = cfg.seg_model.dino.ckpt_config_filename, device=device)
            sam_checkpoint = os.path.join(cfg.pretrained_models_dir, 'sam_vit_h_4b8939.pth')
            model_seg = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    model = get_model(cfg_path=cfg.diffusion_model.cfg_path, ckpt_path = cfg.diffusion_model.ckpt_path).to(device).eval()
    classifier_name = cfg.classifier_model.name
    classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True).to(device)
    classifier_model = classifier_model.eval()
    classifier_model.train = disabled_train

    ddim_steps = cfg.ddim_steps
    ddim_eta = cfg.ddim_eta
    scale = cfg.scale #for unconditional guidance
    strength = cfg.strength #for unconditional guidance

    if "seg_model" not in cfg or cfg.seg_model is None or "name" not in cfg.seg_model:
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= None, **cfg.sampler)
    elif cfg.seg_model.name == "clipseg":
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg, **cfg.sampler)
    else:
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg, detect_model = detect_model, **cfg.sampler)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    n_samples_per_class = cfg.n_samples_per_class
    batch_size = cfg.data.batch_size
    #precision = "autocast" #"full"
    #precision_scope = autocast if precision == "autocast" else nullcontext
      
    print(config)

    #data_path = cfg.data_path
    out_size = 256
    transform_list = [
        transforms.Resize((out_size, out_size)),
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    #dataset = datasets.ImageFolder(data_path,  transform=transform)
    #dataset = ImageNet(data_path, transform=transform)
    dataset = instantiate(cfg.data, transform=transform)
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    with open('data/synset_closest_idx.yaml', 'r') as file:
        synset_closest_idx = yaml.safe_load(file)

    
        
    #my_table = wandb.Table(columns=["id", "image", "source", "target", "lp1", "lp2", *[f"gen_image_{i}" for i in range(n_samples_per_class)], "class_prediction", "video", "mask"])
    my_table = wandb.Table(columns = ["image", "source", "target", "gen_image",  "class_pred", "closness_1", "closness_2", "video"])
    #for i, sample in enumerate(dataset, 1000):
    #    image, label = dataset[i]
    for i, batch in enumerate(data_loader):

        set_seed(seed=cfg.seed if "seed" in cfg else 0)

        if cfg.data.return_tgt_cls:
            image, label, tgt_classes = batch
            tgt_classes = tgt_classes.squeeze().to(device)
        else:
            image, label = batch
            tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)

        image = image.squeeze().to(device)
        label = label.squeeze().to(device) #.item()
        #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #tgt_classes = synset_closest_idx[label]
        #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #shuffle tgt_classes
        #random.shuffle(tgt_classes)
        for j, l in enumerate(label):
            print(f"converting {i} from : {i2h[l.item()]} to: {i2h[tgt_classes[j].item()]}")
        

        init_image = image.clone() #image.repeat(n_samples_per_class, 1, 1, 1).to(device)
        sampler.init_images = init_image.to(device)
        sampler.init_labels = label # n_samples_per_class * [label]
        if isinstance(cfg.sampler.lp_custom, str) and "dino_" in cfg.sampler.lp_custom:
            if device != next(sampler.distance_criterion.dino.parameters()).device:
                sampler.distance_criterion.dino = sampler.distance_criterion.dino.to(device)
            sampler.dino_init_features = sampler.get_dino_features(sampler.init_images, device=device).clone()
        #mapped_image = _unmap_img(init_image)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(_unmap_img(init_image)))  # move to latent space

        out = generate_samples(model, sampler, tgt_classes, ddim_steps, scale, init_latent=init_latent.to(device),
                               t_enc=t_enc, init_image=init_image.to(device), ccdddim=True, latent_t_0=cfg.get("latent_t_0", False))

        all_samples = out["samples"]
        all_videos = out["videos"] 
        all_probs = out["probs"]
        all_masks = out["masks"] 

        # Loop through your data and update the table incrementally
        for j in range(batch_size):
            # Generate data for the current row
            src_image = copy.deepcopy(sampler.init_images[j].cpu()) #all_samples[j][0])
            src_image = wandb.Image(src_image)
            # gen_images = []
            # for k in range(n_samples_per_class):
            #     gen_image = copy.deepcopy(all_samples[j][k + 1])
            #     gen_images.append(wandb.Image(gen_image))

            gen_image = wandb.Image(copy.deepcopy(all_samples[0][j].cpu()))
                
            class_prediction = copy.deepcopy(all_probs[0][j])  # all_probs[j]
            source = i2h[label[j].item()]
            target = i2h[tgt_classes[j].item()]

            #diff =  (init_image - all_samples[j][1:])
            diff = sampler.init_images[j]-all_samples[0][j]   
            diff = diff.view(diff.shape[0], -1)
            lp1 = int(torch.norm(diff, p=1, dim=-1).mean().cpu().numpy())
            lp2 = int(torch.norm(diff, p=2, dim=-1).mean().cpu().numpy())
            #print(f"lp1: {lp1}, lp2: {lp2}")

            video = wandb.Video((255. * all_videos[0][j]).to(torch.uint8).cpu(), fps=10, format="gif")
            mask = wandb.Image(all_masks[j]) if all_masks is not None else None
            #print("added data to table")
            #my_table.add_data(i, src_image, source, target, lp1, lp2, *gen_images, class_prediction, video, mask)
            my_table.add_data(src_image, source, target, gen_image, class_prediction, lp1, lp2, video)

        if i % 2 == 0:
            print(f"logging {i} with {len(my_table.data)} rows")
            table_name = f"dvce_video" #_{i}"
            run.log({table_name: copy.deepcopy(my_table)})

    #wandb.log({"dvce_video_complete": my_table})
    return None

if __name__ == "__main__":
    main()

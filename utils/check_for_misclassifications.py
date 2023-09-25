from torchvision import transforms
from data.datasets import ImageNet
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import torchvision

from utils.preprocessor import CropAndNormalizer

save_path = "data/misclassifications_imagenet_resnet50.txt"
device = torch.device("cuda")

root = "/misc/scratchSSD2/datasets/ILSVRC2012/val"
out_size = 256
transform_list = [
    transforms.Resize((out_size, out_size)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
transform = transforms.Compose(transform_list)
dataset = ImageNet(root=root,transform=transform,idx_to_tgt_cls_path='data/image_idx_to_tgt.yaml')
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=1,
    pin_memory=False,
    shuffle=False,
)

model = getattr(torchvision.models, "resnet50")(pretrained=True)
# model = CropAndNormalizer(model)
model.eval()
model.to(device)

misclassification_indices = []
with torch.inference_mode():
    for blob in tqdm(loader, total=len(loader)):
        img, label, unique_idx = blob
        logits = model(img.to(device)).cpu()
        pred = logits.argmax(dim=1)
        idx_map = label==pred
        misclassification_indices += [idx.item() for idx in unique_idx[~idx_map]]

with open(save_path, "w", encoding="utf-8") as f:
    f.write('\n'.join(str(idx) for idx in misclassification_indices))

print(1-len(misclassification_indices)/len(dataset))
import torchvision
from torchvision import transforms
from data.datasets import ImageNet
import cv2
import os
import shutil
from tqdm import trange
import numpy as np
import torch
from torch import nn

out_dir = "/misc/lmbraid21/faridk/tmp/imgs"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

# bell pepper 945 -> 22
# coral reef 973 -> 30, 43
# buckeye 990 -> 8
# goldfish 1 -> 32
# toyshop 865 -> 20
# bubble 971 -> 33
# tabby 281 -> 6
# egyptian 285 -> 29
idx=774

root = "/misc/scratchSSD2/datasets/ILSVRC2012/val"
out_size = 256
transform_list = [
    transforms.Resize((out_size, out_size)),
    transforms.ToTensor()
]
transform = transforms.Compose(transform_list)
dataset = ImageNet(root=root,transform=transform,class_idcs=[idx],idx_to_tgt_cls_path='data/image_idx_to_tgt.yaml')

imgs = []
model_transform_list = [
    torchvision.transforms.CenterCrop(224),
    lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
]
model_transform = transforms.Compose(model_transform_list)
for i in trange(50):
    blob = dataset.__getitem__(i)
    img, label, _ = blob
    imgs.append(model_transform(torch.clone(img)))
    img = img.permute((1,2,0)).numpy()
    filename = os.path.join(out_dir, f"{str(i).zfill(2)}.jpg")
    cv2.imwrite(filename, cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))


device = torch.device("cuda")
model = getattr(torchvision.models, "resnet50")(pretrained=True).eval().to(device)
imgs = torch.cat(imgs, dim=0).to(device)

with torch.inference_mode():
    logits = model(imgs)

pred = logits.argmax(dim=1).cpu()
print((pred!=idx).nonzero())

print(torch.sum(pred==idx)/len(dataset))
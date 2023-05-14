import torch
import open_clip
import torchvision
from tqdm import tqdm
import json
import os
from torchvision import transforms

from utils.vision_language_wrapper import VisionLanguageWrapper
from utils.preprocessor import GenericPreprocessing
from typing import Callable

data_dir = "/misc/lmbraid21/schrodi/datasets"

device = torch.device("cuda")

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

with open("data/pets_idx_to_label.json", "r") as f:
    pets_idx_to_classname = json.load(f)
prompts = [f"a photo of a {label}, a type of pet." for label in pets_idx_to_classname.values()]
model = VisionLanguageWrapper(model, tokenizer, prompts)
transforms_list = [preprocess.transforms[1], preprocess.transforms[4]] # CenterCrop(224, 224), Normalize
model = GenericPreprocessing(model, transforms.Compose(transforms_list))


def _convert_to_rgb(image):
    return image.convert('RGB')
out_size = 256
transform_list = [
    transforms.Resize((out_size, out_size)),
    # transforms.CenterCrop(out_size),
    _convert_to_rgb,
    transforms.ToTensor(),
]
transform = transforms.Compose(transform_list)
dataset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="test", target_types="category", transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)

with open("/misc/lmbraid21/schrodi/datasets/oxford-iiit-pet/annotations/test.txt", "r") as f:
    data = f.read()

if not os.path.isfile("data/pets_idx_to_label.json"):
    idx_to_classname = {}
    for row in data.split("\n"):
        if row == "":
            continue
        class_name, idx, _, _ = row.split(" ")
        if int(idx) not in idx_to_classname:
            tmp = ""
            for x in class_name.split("_"):
                if x.isdigit():
                    break
                tmp += f" {x}"
            idx_to_classname[int(idx)-1] = tmp[1:].lower()

    with open("data/pets_idx_to_label.json", "w") as f:
        json.dump(idx_to_classname, f, indent=4)

correct = 0
with torch.inference_mode():
    for image, label in tqdm(data_loader, leave=False):
        logits = model(image.to(device))
        text_probs = logits.softmax(dim=-1)
        pred = text_probs.argmax(dim=-1).cpu()
        correct += torch.sum(pred == label)

acc = correct / len(dataset)
print("Acc:", acc.item() * 100) # 90.62
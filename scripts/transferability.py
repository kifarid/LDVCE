from typing import Any
import torch
import os
import torchvision
from torch.utils.data import Dataset
import torchmetrics
from PIL import Image
import timm
import wandb
import pandas as pd
import re


LMB_USERNAME = "schrodi"
WANDB_ENTITY = "kifarid"
os.environ["WANDB_API_KEY"] = 'cff06ca1fa10f98d7fde3bf619ee5ec8550aba11'
os.environ['WANDB_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.wandb"
os.environ['WANDB_DATA_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/counterfactuals"
os.environ['WANDB_CACHE_DIR'] = f"/misc/lmbraid21/{LMB_USERNAME}/tmp/.cache/wandb"

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
imagenet_transform = torchvision.transforms.Compose(
    [
        #torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

class WandBImages(Dataset):
    def __init__(self, artifact: str, transform) -> None:
        super().__init__()
        self.artifact = artifact
        self.transform = transform
        run = wandb.init(entity=WANDB_ENTITY, project="cdiff", mode="online")
        artifact = run.use_artifact(self.artifact, type='run_table')
        self.df_data = pd.DataFrame(data=artifact.get("dvce_video").data, columns=artifact.get("dvce_video").columns)

        self.label_to_idx = {}
        with open("data/imagenet_clsidx_to_label.txt") as file:
            for line in file.readlines():     # readlines should split by '/n' for you
                (key, val) = line.split(':')  # change delimiter to space, not default "," delimited
                self.label_to_idx[val[2:-3]] = int(key)  

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index: Any) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.df_data.iloc[index]
        counterfactual_image = row.gen_image.image # PIL image
        transformed_img = self.transform(counterfactual_image)

        source_label = row.source
        if source_label in self.label_to_idx:
            source_idx = self.label_to_idx[source_label]
        else:
            raise Exception(f"{source_label} not found!")
        
        target_label = row.target
        if target_label in self.label_to_idx:
            target_idx = self.label_to_idx[target_label]
        else:
            raise Exception(f"{target_label} not found!")

        return transformed_img, source_idx, target_idx

@torch.inference_mode()
def evaluate(model: torch.nn.Module, data_loader) -> float:
    model.eval()
    device = next(model.parameters()).device
    source_acc = torchmetrics.Accuracy(num_classes=1000)
    target_acc = torchmetrics.Accuracy(num_classes=1000)
    for input, source, target in data_loader:
        input = input.to(device)
        logits = model(input)
        preds = torch.argmax(logits, dim=-1).cpu()
        source_acc(preds, target)
        target_acc(preds, target)
    return source_acc.compute(), target_acc.compute()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transferability of CEs")
    parser.add_argument('--artifact', required=True, type=str)
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument("--timm", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.timm:
        model = timm.create_model(args.model, pretrained=True, in_chans=3).to(device)
    else:
        model = getattr(torchvision.models, args.model)(pretrained=True).to(device)

    dataset = WandBImages(artifact=args.artifact, transform=imagenet_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)

    source_acc, target_acc = evaluate(model, data_loader)
    print(source_acc, target_acc)
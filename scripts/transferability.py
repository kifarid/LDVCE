from typing import Any
import torch
import os
import torchvision
from torch.utils.data import Dataset
import torchmetrics
from PIL import Image

imagenet_mean = (0.485, 0.456, 0.406),
iamgenet_std = (0.229, 0.224, 0.225),
imagenet_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=imagenet_mean, std=iamgenet_std),
    ]
)

class ImageFolder(Dataset):
    def __init__(self, root: str, transform) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        raise NotImplementedError("Needs to be adapted to WandB design")
        self.data = os.listdir(self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: Any) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()
        filepath = os.path.join(self.root, self.data[index])
        with open(filepath, "rb") as f:
            img = Image.open(f)
        img = img.convert("RGB")
        transformed_img = self.transform(img)
        return transformed_img

@torch.inference_mode()
def evaluate(model: torch.nn.Module, data_loader) -> float:
    device = next(model.parameters()).device
    acc = torchmetrics.Accuracy(num_classes=1000)
    for input, target in data_loader:
        input = input.to(device)
        logits = model(input)
        preds = torch.argmax(logits, dim=-1).cpu()
        acc(preds, target)
    return acc.compute()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transferability of CEs")
    parser.add_argument('--cf_dir', required=True, type=str)
    parser.add_argument('--model', required=True, type=str, default="resnet50")
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(torchvision.models, args.model)(pretrained=True).to(device)

    dataset = ImageFolder(root=args.cf_dir, transform=imagenet_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    acc = evaluate(model, data_loader)
    print(acc)
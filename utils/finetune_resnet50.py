from torchvision import transforms
import torch
from PIL import Image
import hashlib
import glob
from typing import Any
from torch.utils.data import Dataset
from data.datasets import ImageNet
import torchvision
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import os

device = torch.device("cuda")
BATCH_SIZE = 1024
EPOCHS = 16
LR = 0.1

class ImageFolder(Dataset):
    def __init__(self, root, transform, label_idx, train: bool = True, oversampling_factor: int = 1) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.label_idx = label_idx

        self.data = [img for img in sorted(glob.glob(self.root + "/*.jpg"))]
        if train:
            self.data = self.data[:25]
            assert isinstance(oversampling_factor, int) and oversampling_factor >= 1
            tmp_copy = deepcopy(self.data)
            for _ in range(oversampling_factor-1):
                self.data += deepcopy(tmp_copy)
        else:
            self.data = self.data[25:]

        # check for duplicates
        if not train:
            hashs = [hashlib.md5(Image.open(path).tobytes()).hexdigest() for path in self.data]
            assert len(set(hashs)) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: Any) -> Any:
        img = Image.open(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label_idx

def train(model, loader, optimizer, criterion):
    model.train()
    for blob in tqdm(loader, total=len(loader), leave=False):
        optimizer.zero_grad()
        input, label = blob
        input = input.to(device)
        label = label.to(device)
        with torch.cuda.amp.autocast():
            logits = model(input)
            loss = criterion(logits, label)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correctly_classified = 0
    for blob in loader:
        input, label = blob
        logits = model(input.to(device)).cpu()
        pred = logits.argmax(dim=1)
        correctly_classified += torch.sum(label==pred).item()
    return correctly_classified / len(loader.dataset)


model = getattr(torchvision.models, "resnet50")(pretrained=True)
# model = CropAndNormalizer(model)
model.train()

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True 

model.to(device)

out_size = 256
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transform_list = [
    transforms.Resize((out_size, out_size)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
val_transform = transforms.Compose(val_transform_list)

imagenet_root = "/misc/scratchSSD2/datasets/ILSVRC2012/train"
full_dataset = ImageNet(root=imagenet_root,split="train",transform=train_transform,idx_to_tgt_cls_path='data/image_idx_to_tgt.yaml')

# failure_mode_root = "/misc/lmbraid21/faridk/scraped_images/bald_eagle"
# label_idx = 22
# failure_mode_root = "/misc/lmbraid21/faridk/scraped_images/spoon_engravings"
# label_idx = 910
failure_mode_root = "/misc/lmbraid21/faridk/scraped_images/sandals"
label_idx = 774
train_dataset = ImageFolder(failure_mode_root, train_transform, label_idx, train=True, oversampling_factor=4)
val_dataset = ImageFolder(failure_mode_root, val_transform, label_idx, train=False)

train_set = torch.utils.data.ConcatDataset([full_dataset, train_dataset])
train_loader = DataLoader(
    train_set,
    batch_size=512,
    num_workers=16,
    pin_memory=False,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=1,
    pin_memory=False,
    shuffle=False,
)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
criterion = torch.nn.CrossEntropyLoss()

accs = []
acc = evaluate(model, val_loader)
print(acc)
with open(f"{os.path.basename(failure_mode_root)}.txt", "w") as f:
    f.write(f"{acc}\n")
for _ in range(EPOCHS):
    train(model, train_loader, optimizer, criterion)
    scheduler.step()
    acc = evaluate(model, val_loader)
    print(acc)
    with open(f"{os.path.basename(failure_mode_root)}.txt", "a") as f:
        f.write(f"{acc}\n")
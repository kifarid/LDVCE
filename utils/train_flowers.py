import torch
from torch import nn
import torchvision
from torchvision import transforms
from tqdm import tqdm, trange
import os
import numpy as np
import random

from utils.preprocessor import GenericPreprocessing
from utils.dino_linear import DINOLinear, LinearClassifier

also_train = False
data_dir = "/misc/lmbraid21/schrodi/datasets"
model_dir = "/misc/lmbraid21/schrodi/pretrained_models"
num_labels = 102
lr = 1e-3
epochs = 30
n_last_blocks = 1 #4
device = torch.device("cuda")

if also_train:
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    def train(model, linear_classifier, loss_fn, optimizer, loader, n):
        linear_classifier.train()
        for input, target in tqdm(loader, leave=False):
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                #intermediate_output = model.get_intermediate_layers(input, n)
                #output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                output = model(input)
            logits = linear_classifier(output)

            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    @torch.inference_mode()
    def validate(model, linear_classifier, loader, n):
        linear_classifier.eval()
        acc = 0
        for input, target in tqdm(loader, leave=False):
            input = input.to(device)
            target = target.to(device)
            intermediate_output = model.get_intermediate_layers(input, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            logits = linear_classifier(output)
            pred = logits.argmax(dim=-1)
            acc += torch.sum(pred == target).cpu().item()
        return acc / len(loader.dataset)


    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()
    dim = dino.embed_dim
    #model = DINOLinear(dino, dim=dim*n_last_blocks, num_labels=num_labels).to(device)
    #dino = CropAndNormalizer(dino, crop_size=224)

    linear_classifier = LinearClassifier(dim*n_last_blocks, num_labels).train().to(device)

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    target_transform = lambda x: x-1
    train_dataset = torchvision.datasets.Flowers102(root=data_dir, split="train", transform=transform, target_transform=target_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            num_workers=8,
            pin_memory=False,
            shuffle=True,
        )
    test_dataset = torchvision.datasets.Flowers102(root=data_dir, split="test", transform=transform, target_transform=target_transform, download=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            num_workers=16,
            pin_memory=False,
            shuffle=True
        )

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr * 256 / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0.0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in trange(epochs, leave=False):
        train(dino, linear_classifier, loss_fn, optimizer, train_loader, n_last_blocks)
        scheduler.step()
    acc = validate(dino, linear_classifier, test_loader, n_last_blocks)
    print(epoch, acc)
    torch.save(linear_classifier.state_dict(), os.path.join(model_dir, "dino_flowers_linear.pth"))
    print("Saved!")


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
target_transform = lambda x: x-1
dataset = torchvision.datasets.Flowers102(root=data_dir, split="test", transform=transform, target_transform=target_transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)

dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()
dim = dino.embed_dim
linear_classifier = LinearClassifier(dim*n_last_blocks, num_labels)
linear_classifier.load_state_dict(torch.load(os.path.join(model_dir, "dino_flowers_linear.pth"), map_location="cpu"), strict=True)
linear_classifier = linear_classifier.eval().to(device)
classifier_model = DINOLinear(dino, linear_classifier)
transforms_list = [transforms.CenterCrop(224), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
classifier_model = GenericPreprocessing(classifier_model, transforms.Compose(transforms_list))

acc = 0
with torch.inference_mode():
    for batch in tqdm(loader, leave=False):
        image, label = batch
        logits = classifier_model(image.to(device))
        pred = torch.argmax(logits, dim=-1).cpu()
        acc += torch.sum(pred == label)

final_acc = acc/len(dataset)
print(round(final_acc.item()*100, 2))